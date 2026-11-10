"""Turn a transcript into summary, decisions and action items using
a local model via Ollama (default) or Claude (Anthropic API)."""

import json
import os
import re
import threading
import time

import requests

# ---------------- Performance stats (read by the UI) ----------------

_stats_lock = threading.Lock()
_llm = {"active": False, "provider": None, "model": None, "call": 0, "started": 0.0,
        "first_token_at": None, "tokens": 0, "prompt_tokens": 0, "total_tokens": 0, "last": None}


def llm_stats() -> dict:
    """Snapshot of the summariser's speed. tok_s is live while generating."""
    with _stats_lock:
        s = dict(_llm)
    now = time.time()
    live = None
    if s["active"] and s["first_token_at"] and s["tokens"] > 1:
        live = round(s["tokens"] / max(now - s["first_token_at"], 1e-3), 1)
    return {"active": s["active"], "provider": s["provider"], "model": s["model"], "call": s["call"],
            "tokens": s["tokens"], "total_tokens": s["total_tokens"], "tok_s": live,
            "elapsed": round(now - s["started"], 1) if s["active"] else None,
            "waiting_first_token": bool(s["active"] and not s["first_token_at"]),
            "last": s["last"]}


def _llm_begin(provider: str, model: str):
    with _stats_lock:
        _llm.update(active=True, provider=provider, model=model, call=_llm["call"] + 1,
                    started=time.time(), first_token_at=None, tokens=0, prompt_tokens=0)


def _llm_token():
    with _stats_lock:
        if _llm["first_token_at"] is None:
            _llm["first_token_at"] = time.time()
        _llm["tokens"] += 1


def _llm_end(last: dict | None):
    with _stats_lock:
        _llm["active"] = False
        if last:
            _llm["total_tokens"] += last.get("output_tokens", 0)
            _llm["last"] = last
    run = getattr(_run, "usage", None)
    if run is not None and last:
        run["input_tokens"] += last.get("prompt_tokens") or 0
        run["output_tokens"] += last.get("output_tokens") or 0
        run["calls"] += 1


# Token usage of the summarise() run on this thread (long meetings make several LLM calls).
_run = threading.local()

# Anthropic list prices, USD per 1M tokens (input, output). Unknown models show tokens only.
CLAUDE_PRICES = {
    "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-fable-5-1": (10.0, 50.0),
}


def last_run_usage(provider: str | None = None, model: str | None = None) -> dict:
    """Tokens used by the most recent summarize() on this thread, plus an estimated cost for Claude."""
    u = dict(getattr(_run, "usage", None) or {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    price = CLAUDE_PRICES.get(model or "") if provider == "claude" else None
    if price and u["calls"]:
        usd = (u["input_tokens"] * price[0] + u["output_tokens"] * price[1]) / 1e6
        u["est_cost_usd"] = round(usd, 4)
        u["est_cost_aud"] = round(usd * float(os.getenv("USD_TO_AUD", "1.55")), 4)   # fixed rate from .env
    return u

SYSTEM_PROMPT = """You are an expert meeting note-taker for an engineering team.
You receive an automatic transcript (may contain recognition errors). Some lines start with
a speaker label: "Me:" is the person who recorded the meeting and owns these notes. "Others:"
is everyone else in the meeting together, and "Speaker 1:", "Speaker 2:", ... tell apart the
individual voices among them (the numbers are arbitrary and are not names, so never guess a name
for a speaker; a person who is named in the conversation may be matched to a speaker only when it
is clear). Lines without a label have an unknown speaker.
Produce accurate, concise meeting notes. Never invent facts that are not in the transcript.
Only assign an owner to an action item when the transcript makes it clear who will do it;
otherwise leave owner empty."""

JSON_SPEC = """Respond with ONLY a JSON object, no prose, in this shape:
{
  "title": "short descriptive meeting title",
  "summary": "2-4 sentence overview of purpose and outcome",
  "topics": [{"heading": "topic name", "points": ["key point", "..."]}],
  "decisions": ["decision that was agreed", "..."],
  "action_items": [{"task": "what needs to be done", "owner": "name or empty string", "due": "date/timeframe or empty string"}],
  "open_questions": ["unresolved question", "..."]
}"""


def _language_line() -> str:
    lang = os.getenv("NOTES_LANGUAGE", "English")
    return f"Write all notes in {lang}. Keep technical terms and names as spoken."


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model did not return JSON:\n" + text[:500])
    return json.loads(text[start : end + 1])


def _user_prompt(transcript: str, hint_title: str | None) -> str:
    hint = f"The user named this meeting: {hint_title}\n\n" if hint_title else ""
    return f"{hint}{_language_line()}\n\n{JSON_SPEC}\n\n<transcript>\n{transcript}\n</transcript>"


# ---------------- Claude ----------------

def _claude(prompt: str, max_tokens: int = 4000) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    _llm_begin("claude", model)
    t0, last = time.time(), None
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        took = time.time() - t0
        out = msg.usage.output_tokens
        last = {"output_tokens": out, "prompt_tokens": msg.usage.input_tokens,
                "gen_tok_s": round(out / took, 1) if took else None, "total_sec": round(took, 1)}
    finally:
        _llm_end(last)
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


# ---------------- Cloud proxy ----------------

def _cloud(transcript: str, hint_title: str | None, meeting_id: str | None) -> str:
    import cloud

    _llm_begin("cloud", "proxy")
    t0, last = time.time(), None
    try:
        text, usage = cloud.summarize(transcript, meeting_id or "unknown", hint_title,
                                      os.getenv("NOTES_LANGUAGE", "English"))
        took = time.time() - t0
        out = usage.get("output_tokens", 0)
        last = {"output_tokens": out, "prompt_tokens": usage.get("input_tokens", 0),
                "gen_tok_s": round(out / took, 1) if took else None, "total_sec": round(took, 1)}
    finally:
        _llm_end(last)
    return text


# ---------------- Ollama ----------------

def _ollama(prompt: str, as_json: bool = True, model: str | None = None) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    body = {
        "model": model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "stream": True,  # streamed so the UI can show tokens/s while it generates
        # How long Ollama keeps the model (~10 GB for a 14B) in memory after this request. Its own
        # default is 5 minutes; a short value frees the RAM soon after the notes are written while
        # still covering the back-to-back chunk requests of one long meeting.
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30s"),
        "options": {"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")), "temperature": 0.2},
    }
    if as_json:
        body["format"] = "json"
    _llm_begin("ollama", body["model"])
    parts, final, last = [], {}, None
    try:
        with requests.post(f"{host}/api/chat", json=body, stream=True, timeout=(10, 1800)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(d["error"])
                piece = d.get("message", {}).get("content", "")
                if piece:
                    parts.append(piece)
                    _llm_token()
                if d.get("done"):
                    final = d
        ns = 1e9
        out, pin = final.get("eval_count", 0), final.get("prompt_eval_count", 0)
        last = {"output_tokens": out, "prompt_tokens": pin,
                "gen_tok_s": round(out / (final["eval_duration"] / ns), 1) if final.get("eval_duration") else None,
                "prompt_tok_s": round(pin / (final["prompt_eval_duration"] / ns), 1) if final.get("prompt_eval_duration") else None,
                "load_sec": round(final.get("load_duration", 0) / ns, 1),
                "total_sec": round(final.get("total_duration", 0) / ns, 1)}
    finally:
        _llm_end(last)
    return "".join(parts)


def _chunks(text: str, size: int):
    lines, buf, n = text.splitlines(), [], 0
    for line in lines:
        if n + len(line) > size and buf:
            yield "\n".join(buf)
            buf, n = [], 0
        buf.append(line)
        n += len(line) + 1
    if buf:
        yield "\n".join(buf)


def _ollama_summarize(transcript: str, hint_title: str | None, model: str | None = None) -> dict:
    """Small local models have short context, so long meetings are summarised
    in parts first (map) and then merged (reduce)."""
    size = int(os.getenv("OLLAMA_CHUNK_CHARS", "20000"))
    if len(transcript) <= size:
        return _parse_json(_ollama(_user_prompt(transcript, hint_title), model=model))

    parts = list(_chunks(transcript, size))
    notes = []
    for i, chunk in enumerate(parts, 1):
        p = (f"This is part {i} of {len(parts)} of a meeting transcript. {_language_line()}\n"
             "Write detailed bullet notes of everything discussed, every decision, "
             "and every action item (with owner if clear). Keep who said what when it matters "
             "(Me / Others / Speaker N). Plain text bullets only.\n\n"
             f"<transcript_part>\n{chunk}\n</transcript_part>")
        notes.append(f"--- Part {i} ---\n" + _ollama(p, as_json=False, model=model))
    merged = "\n\n".join(notes)
    hint = f"The user named this meeting: {hint_title}\n\n" if hint_title else ""
    p = (f"{hint}Below are notes taken from consecutive parts of one meeting. "
         f"Merge them into final meeting notes, removing duplicates. {_language_line()}\n\n"
         f"{JSON_SPEC}\n\n<notes>\n{merged}\n</notes>")
    return _parse_json(_ollama(p, model=model))


# ---------------- Public API ----------------

def summarize(transcript: str, provider: str = "ollama", hint_title: str | None = None,
              model: str | None = None, meeting_id: str | None = None) -> dict:
    """model only applies to Ollama; None uses OLLAMA_MODEL from .env. meeting_id is what the cloud
    proxy counts against the free tier (re-summarising the same id is free).
    Token usage of the run is available afterwards from last_run_usage()."""
    _run.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    if not transcript.strip():
        return {"title": hint_title or "Empty meeting", "summary": "No speech was detected.",
                "topics": [], "decisions": [], "action_items": [], "open_questions": []}
    if provider == "ollama":
        data = _ollama_summarize(transcript, hint_title, model)
    elif provider == "cloud":
        data = _parse_json(_cloud(transcript, hint_title, meeting_id))
    else:
        data = _parse_json(_claude(_user_prompt(transcript, hint_title)))
    for key in ("topics", "decisions", "action_items", "open_questions"):
        data.setdefault(key, [])
    data.setdefault("summary", "")
    data.setdefault("title", hint_title or "Meeting")
    return data


def _txt(x) -> str:
    """Models sometimes return null or a non-string where we expect text."""
    return str(x).strip() if x else ""


def render_markdown(notes: dict, meta: dict, transcript_text: str) -> str:
    out = [f"# {notes.get('title') or 'Meeting'}", ""]
    out.append(f"*{meta.get('date', '')} · {meta.get('duration', '')} · "
               f"summarised by {meta.get('provider', '')}*")
    u = meta.get("usage") or {}
    if u.get("calls"):
        line = f"*Tokens: {u['input_tokens']:,} in · {u['output_tokens']:,} out"
        if u.get("est_cost_aud") is not None:
            line += f" · est. A${u['est_cost_aud']:.3f}"
        out += ["", line + "*"]
    out += ["", "## Summary", "", _txt(notes.get("summary")), ""]

    if notes.get("action_items"):
        out += ["## Action Items", ""]
        for a in notes["action_items"]:
            if not isinstance(a, dict):
                out.append(f"- [ ] {_txt(a)}")
                continue
            line = f"- [ ] {_txt(a.get('task'))}"
            if _txt(a.get("owner")):
                line += f" — **{_txt(a['owner'])}**"
            if _txt(a.get("due")):
                line += f" _(due: {_txt(a['due'])})_"
            out.append(line)
        out.append("")

    if notes.get("decisions"):
        out += ["## Decisions", ""] + [f"- {d}" for d in notes["decisions"]] + [""]

    topics = [t for t in notes.get("topics", []) if isinstance(t, dict)]
    if topics:
        out += ["## Discussion", ""]
    for t in topics:
        if True:
            out += [f"### {t.get('heading', '')}", ""] + [f"- {p}" for p in t.get("points", [])] + [""]

    if notes.get("open_questions"):
        out += ["## Open Questions", ""] + [f"- {q}" for q in notes["open_questions"]] + [""]

    out += ["## Transcript", "", "<details>", "<summary>Show full transcript</summary>", "",
            transcript_text, "", "</details>", ""]
    return "\n".join(out)
