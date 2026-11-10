// Meeting Notes summary proxy (Cloudflare Worker).
// The app sends only transcript TEXT (never audio). This worker holds the Anthropic API key,
// wraps the transcript in a fixed prompt (so it cannot be used as a general-purpose LLM), and
// enforces the free tier: FREE_MEETINGS distinct meetings per device.
// It never logs or stores transcripts. KV holds only counters keyed by device id / IP / day.
//
// Routes:  GET /v1/quota      -> { used, limit, remaining }
//          POST /v1/summarize -> { text, usage, quota }   body: { meeting_id, transcript, hint_title?, language? }
// Headers: X-Device-Id (uuid v4, required), X-Invite-Code (required only when INVITE_CODES is set)

import { SYSTEM_PROMPT, JSON_SPEC } from "./prompts.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const DAY = 2 * 24 * 3600; // counters expire after two days

function num(v, d) {
  const n = Number(v);
  return v !== undefined && v !== "" && Number.isFinite(n) ? n : d;
}

export function limits(env) {
  return {
    free: num(env.FREE_MEETINGS, 5),
    maxCallsPerMeeting: num(env.MAX_CALLS_PER_MEETING, 5),
    maxChars: num(env.MAX_TRANSCRIPT_CHARS, 250000),
    perIpDay: num(env.MAX_REQUESTS_PER_IP_DAY, 40),
    newDevicesPerIpDay: num(env.MAX_NEW_DEVICES_PER_IP_DAY, 10),
    globalDay: num(env.MAX_REQUESTS_PER_DAY, 1000),
    model: env.MODEL || "claude-sonnet-5-5",
  };
}

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
const err = (status, code, message, extra = {}) => json({ error: code, message, ...extra }, status);

async function getJSON(kv, key) {
  const raw = await kv.get(key);
  return raw ? JSON.parse(raw) : null;
}
const putJSON = (kv, key, value, ttl) =>
  kv.put(key, JSON.stringify(value), ttl ? { expirationTtl: ttl } : undefined);

async function bump(kv, key) {
  const n = Number((await kv.get(key)) || 0) + 1;
  await kv.put(key, String(n), { expirationTtl: DAY });
  return n;
}
async function peek(kv, key) {
  return Number((await kv.get(key)) || 0);
}

function quotaOf(device, lim) {
  const used = Object.keys((device && device.meetings) || {}).length;
  return { used, limit: lim.free, remaining: Math.max(0, lim.free - used) };
}

export function buildPrompt({ transcript, hint_title, language }) {
  const hint = hint_title ? `The user named this meeting: ${String(hint_title).slice(0, 200)}\n\n` : "";
  const lang = /^[\p{L}\p{N} \-()]{1,40}$/u.test(language || "") ? language : "English";
  return `${hint}Write all notes in ${lang}. Keep technical terms and names as spoken.\n\n${JSON_SPEC}\n\n<transcript>\n${transcript}\n</transcript>`;
}

async function callClaude(env, lim, prompt, fetchFn) {
  const res = await fetchFn("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: lim.model,
      max_tokens: 4000,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) return null;
  const data = await res.json();
  const text = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
  return { text, usage: { input_tokens: data.usage?.input_tokens || 0, output_tokens: data.usage?.output_tokens || 0 } };
}

export async function handle(request, env, fetchFn = fetch) {
  const url = new URL(request.url);
  const lim = limits(env);
  const kv = env.KV;

  if (request.method === "GET" && url.pathname === "/") return json({ ok: true });

  const isQuota = request.method === "GET" && url.pathname === "/v1/quota";
  const isSummarize = request.method === "POST" && url.pathname === "/v1/summarize";
  if (!isQuota && !isSummarize) return err(404, "not_found", "Not found");

  if (env.INVITE_CODES) {
    const allowed = env.INVITE_CODES.split(",").map((s) => s.trim()).filter(Boolean);
    if (!allowed.includes(request.headers.get("X-Invite-Code") || "")) {
      return err(401, "invite_required", "A valid invite code is required for this beta.");
    }
  }
  const deviceId = (request.headers.get("X-Device-Id") || "").toLowerCase();
  if (!UUID_RE.test(deviceId)) return err(400, "bad_device", "Missing or invalid device id.");

  const devKey = `dev:${deviceId}`;
  let device = await getJSON(kv, devKey);

  if (isQuota) return json(quotaOf(device, lim));

  // ---- summarize ----
  const raw = await request.text();
  if (raw.length > lim.maxChars * 4 + 4096) return err(413, "too_large", "Transcript is too long.");
  let body;
  try { body = JSON.parse(raw); } catch { return err(400, "bad_request", "Body must be JSON."); }
  const { transcript, meeting_id: meetingId } = body || {};
  if (typeof transcript !== "string" || !transcript.trim()) return err(400, "bad_request", "transcript is required.");
  if (typeof meetingId !== "string" || !meetingId || meetingId.length > 120) return err(400, "bad_request", "meeting_id is required.");
  if (transcript.length > lim.maxChars) return err(413, "transcript_too_long", "This transcript is too long for cloud summaries.");

  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  const day = new Date().toISOString().slice(0, 10);
  if ((await peek(kv, `g:${day}`)) >= lim.globalDay) return err(503, "busy", "The service is busy. Try again later.");
  if ((await peek(kv, `ip:${ip}:${day}`)) >= lim.perIpDay) return err(429, "rate_limited", "Too many requests from this network today.");

  if (!device) {
    if ((await peek(kv, `ipnew:${ip}:${day}`)) >= lim.newDevicesPerIpDay) {
      return err(429, "rate_limited", "Too many new installs from this network today.");
    }
    device = { created: new Date().toISOString(), meetings: {} };
  }
  const calls = device.meetings[meetingId] || 0;
  const isNew = !(meetingId in device.meetings);
  if (isNew && Object.keys(device.meetings).length >= lim.free) {
    return err(429, "quota_exceeded", `You have used your ${lim.free} free cloud summaries.`, quotaOf(device, lim));
  }
  if (calls >= lim.maxCallsPerMeeting) return err(429, "rate_limited", "Too many re-summaries of this meeting.");

  let result;
  try {
    result = await callClaude(env, lim, buildPrompt(body), fetchFn);
  } catch {
    result = null;
  }
  if (!result) return err(502, "upstream", "The summary service failed. You were not charged a free meeting.");

  // charge only after success
  device.meetings[meetingId] = calls + 1;
  if (isNew && Object.keys(device.meetings).length === 1) await bump(kv, `ipnew:${ip}:${day}`);
  await putJSON(kv, devKey, device);
  await bump(kv, `g:${day}`);
  await bump(kv, `ip:${ip}:${day}`);
  return json({ text: result.text, usage: result.usage, quota: quotaOf(device, lim) });
}

export default { fetch: (request, env) => handle(request, env) };
