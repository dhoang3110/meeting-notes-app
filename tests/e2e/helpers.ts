import { expect, type Page, type Route } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

export const NOTES = path.join(__dirname, '.data', 'notes');
export const TRASH = path.join(__dirname, '.data', 'home', '.Trash');

const pad = (n: number) => String(n).padStart(2, '0');
const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

let seq = 0;
/** Unique, sortable meeting id. daysAgo 0 = Today, 1 = Yesterday, 2+ = Earlier. */
export function meetingId(slug: string, daysAgo = 0) {
  const d = new Date(Date.now() - daysAgo * 864e5);
  seq += 1;
  const time = `${pad(10 + (process.pid % 10))}${pad(seq % 60)}${pad(Math.floor(Math.random() * 60))}`;
  return `${ymd(d)}_${time}_${slug}-${process.pid}-${seq}`;
}

export const NOTES_MD = `# Roadmap sync

*Thu 06 Nov 2026, 10:00 · 12 min*

## Summary
We agreed to ship the **beta** next week.

## Action items
- [ ] Dong: write the release notes
- [x] Cam: book the demo

<script>window.__pwned = true</script>
<img src=x onerror="window.__pwned = true">
`;

export interface SeedOpts {
  title?: string;
  daysAgo?: number;
  notes?: string | null;       // markdown; null = no notes.md
  duration_sec?: number;
  transcript?: boolean;
}

/** Writes a meeting folder the way the app does, and returns its id. */
export function seedMeeting(slug: string, o: SeedOpts = {}) {
  const id = meetingId(slug, o.daysAgo ?? 0);
  const dir = path.join(NOTES, id);
  fs.mkdirSync(dir, { recursive: true });
  const meta: Record<string, unknown> = { date: 'Thu 06 Nov 2026, 10:00', source: 'recording' };
  if (o.title) meta.title = o.title;
  if (o.duration_sec) meta.duration_sec = o.duration_sec;
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta));
  const notes = o.notes === undefined ? NOTES_MD : o.notes;
  if (notes !== null) fs.writeFileSync(path.join(dir, 'notes.md'), notes);
  if (o.transcript) fs.writeFileSync(path.join(dir, 'transcript.json'), '[]');
  return id;
}

export const DEVICES = [
  { index: 0, name: 'MacBook Pro Microphone' },
  { index: 1, name: 'BlackHole 2ch' },
  { index: 2, name: 'USB Headset' },
];

export const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

export interface MockOpts {
  devices?: typeof DEVICES;
  ollama?: { running: boolean; models: { name: string; size_gb: number }[] };
  metrics?: Record<string, unknown>;
}

export const OLLAMA_OK = { running: true, models: [{ name: 'qwen2.5:14b', size_gb: 9 }, { name: 'llama3.1:8b', size_gb: 4.7 }] };

export const METRICS = {
  recording: false,
  system: { cpu: 42, ram_used_gb: 10.5, ram_total_gb: 16, swap_used_gb: 0.4, swap_mb_s: 0, mem_pressure: 'normal', mem_status: 'ok', app_rss_gb: 0.3, thermal: 'normal',
            gpu: { util: 18 }, battery: { percent: 80, plugged: true } },
  whisper: null,
  llm: { active: false, last: null },
  ollama: null,
  audio: [],
};

/**
 * Stubs what needs real hardware or a running Ollama, and serves marked/DOMPurify locally so the
 * tests do not depend on a CDN. Everything else (config, meetings, notes, delete) hits the real server.
 */
export async function mockEnvironment(page: Page, o: MockOpts = {}) {
  await page.route('**/cdnjs.cloudflare.com/**/marked.min.js', r =>
    r.fulfill({ path: path.join(__dirname, 'node_modules/marked/marked.min.js'), contentType: 'text/javascript' }));
  await page.route('**/cdnjs.cloudflare.com/**/purify.min.js', r =>
    r.fulfill({ path: path.join(__dirname, 'node_modules/dompurify/dist/purify.min.js'), contentType: 'text/javascript' }));
  await page.route('**/api/devices', r => json(r, o.devices ?? DEVICES));
  await page.route('**/api/ollama/models', r => json(r, o.ollama ?? OLLAMA_OK));
  await page.route('**/api/metrics', r => json(r, o.metrics ?? METRICS));
}

/** Opens the app and waits for the first load (config, devices, providers, list) to finish. */
export async function openApp(page: Page, hash = '', opts: { devices?: boolean } = {}) {
  await page.goto('/' + hash);
  await expect(page.locator('#provider')).not.toHaveValue('');
  if (opts.devices !== false) await expect(page.locator('#mic option').first()).toBeAttached();
}

export const meetingItem = (page: Page, id: string) => page.locator(`.item[data-id="${id}"]`);

/** Accepts the next alert/confirm inside the handler (no race with the page) and resolves with its message. */
export function nextDialog(page: Page): Promise<string> {
  return new Promise(resolve => page.once('dialog', d => { resolve(d.message()); void d.accept(); }));
}
