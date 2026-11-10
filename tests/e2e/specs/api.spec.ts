import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { NOTES, NOTES_MD, TRASH, seedMeeting } from '../helpers';

// Contract tests against the real FastAPI app (isolated notes folder, HOME and .env: see server.py).
// Only endpoints that never touch audio hardware, Ollama, the network or the real .env are exercised.

test.describe('API', () => {
  test('GET / serves the single-page UI', async ({ request }) => {
    const r = await request.get('/');
    expect(r.ok()).toBe(true);
    expect(r.headers()['content-type']).toContain('text/html');
    expect(await r.text()).toContain('<title>Meeting Notes</title>');
  });

  test('GET /api/config describes the models and the notes folder', async ({ request }) => {
    const c = await (await request.get('/api/config')).json();
    expect(c.notes_dir).toBe(NOTES);
    expect(c.default_provider).toBe('ollama');
    expect(c.claude_ready).toBe(false);
    expect(c.default_stt).toMatch(/^parakeet:/);
    expect(c.stt_models.map((e: any) => e.id)).toEqual(expect.arrayContaining([c.default_stt]));
    expect(c.ollama_model).toBe('qwen2.5:14b');
  });

  test('GET /api/settings never returns a key', async ({ request }) => {
    const s = await (await request.get('/api/settings')).json();
    expect(s).toEqual({ has_key: false, key_hint: '', claude_model: expect.any(String) });
  });

  test('POST /api/settings rejects bad input before writing anything', async ({ request }) => {
    const badKey = await request.post('/api/settings', { data: { anthropic_api_key: 'not-a-key' } });
    expect(badKey.status()).toBe(400);
    expect((await badKey.json()).detail).toContain('sk-ant-');
    const badModel = await request.post('/api/settings', { data: { claude_model: 'bad model!' } });
    expect(badModel.status()).toBe(400);
    expect((await badModel.json()).detail).toBe('Invalid model id.');
  });

  test('POST /api/settings/test needs a key', async ({ request }) => {
    const r = await request.post('/api/settings/test', { data: {} });
    expect(r.status()).toBe(400);
    expect((await r.json()).detail).toBe('No API key to test.');
  });

  test('GET /api/meetings lists a seeded meeting with its metadata', async ({ request }) => {
    const id = seedMeeting('api-list', { title: 'API list', duration_sec: 3600 * 1.5 });
    const noNotes = seedMeeting('api-nonotes', { notes: null });
    const list = await (await request.get('/api/meetings')).json();
    const m = list.find((x: any) => x.id === id);
    expect(m).toMatchObject({ id, title: 'API list', date: 'Thu 06 Nov 2026, 10:00', duration: '1h 30m',
                              has_notes: true, has_transcript: false, stage: null, error: null });
    expect(list.find((x: any) => x.id === noNotes).has_notes).toBe(false);
    const ids = list.map((x: any) => x.id);
    expect(ids).toEqual([...ids].sort().reverse());   // newest first
  });

  test('GET /api/meetings/{id} returns the Markdown, 404 otherwise', async ({ request }) => {
    const id = seedMeeting('api-md');
    const ok = await request.get(`/api/meetings/${id}`);
    expect(ok.status()).toBe(200);
    expect(ok.headers()['content-type']).toContain('text/plain');
    expect(await ok.text()).toBe(NOTES_MD);
    const none = seedMeeting('api-md-none', { notes: null });
    const r1 = await request.get(`/api/meetings/${none}`);
    expect(r1.status()).toBe(404);
    expect((await r1.json()).detail).toBe('Notes not ready yet');
    const r2 = await request.get('/api/meetings/2000-01-01_000000_missing');
    expect(r2.status()).toBe(404);
    expect((await r2.json()).detail).toBe('Meeting not found');
  });

  test('DELETE /api/meetings/{id} moves the folder to the Trash, and a second delete is a 404', async ({ request }) => {
    const id = seedMeeting('api-del');
    const r = await request.delete(`/api/meetings/${id}`);
    expect(r.status()).toBe(200);
    expect(fs.existsSync(path.join(NOTES, id))).toBe(false);
    expect(fs.existsSync(path.join(TRASH, id, 'notes.md'))).toBe(true);
    expect((await request.delete(`/api/meetings/${id}`)).status()).toBe(404);
  });

  test('re-summarise validates the meeting and the provider first', async ({ request }) => {
    const id = seedMeeting('api-resum', { transcript: true });
    expect((await request.post('/api/meetings/2000-01-01_000000_missing/summarize', { data: { provider: 'ollama' } })).status()).toBe(404);
    const bad = await request.post(`/api/meetings/${id}/summarize`, { data: { provider: 'gpt' } });
    expect(bad.status()).toBe(400);
    expect((await bad.json()).detail).toContain('Unknown summary provider');
  });

  test('POST /api/start validates before touching any audio device', async ({ request }) => {
    const none = await request.post('/api/start', { data: { devices: [] } });
    expect(none.status()).toBe(400);
    expect((await none.json()).detail).toBe('Choose at least one input device');
    const prov = await request.post('/api/start', { data: { devices: [0], provider: 'bogus' } });
    expect(prov.status()).toBe(400);
    const stt = await request.post('/api/start', { data: { devices: [0], stt: 'whisper:evil/model-name' } });
    expect(stt.status()).toBe(400);
    expect((await stt.json()).detail).toContain('Unknown speech-to-text model');
  });

  test('POST /api/upload rejects an unknown provider or model without saving the file', async ({ request }) => {
    const before = fs.readdirSync(NOTES).length;
    const r = await request.post('/api/upload', {
      multipart: { file: { name: 'a.wav', mimeType: 'audio/wav', buffer: Buffer.from('x') }, provider: 'bogus' },
    });
    expect(r.status()).toBe(400);
    const r2 = await request.post('/api/upload', {
      multipart: { file: { name: 'a.wav', mimeType: 'audio/wav', buffer: Buffer.from('x') }, stt: 'nope' },
    });
    expect(r2.status()).toBe(400);
    expect(fs.readdirSync(NOTES).length).toBeGreaterThanOrEqual(before);   // other tests seed in parallel: never fewer
    expect(fs.readdirSync(NOTES).some(n => n.endsWith('_a'))).toBe(false);
  });

  test('stop with no recording running is a 409', async ({ request }) => {
    const r = await request.post('/api/stop');
    expect(r.status()).toBe(409);
    expect((await r.json()).detail).toBe('Not recording');
  });

  test('idle status, live transcript, monitor and voice endpoints', async ({ request }) => {
    expect(await (await request.get('/api/status')).json()).toMatchObject({
      recording: false, elapsed: 0, levels: [], current: null, no_audio: false, session: null, live: null });
    expect(await (await request.get('/api/live?after=0')).json()).toEqual({ segments: [], recording: false, partial: null });
    expect(await (await request.get('/api/monitor')).json()).toEqual({ active: false, seconds: 0, levels: [] });
    const empty = await request.post('/api/monitor/start', { data: { devices: [] } });
    expect(empty.status()).toBe(400);
    expect(await (await request.get('/api/voice')).json()).toMatchObject({ enrolled: false, sampling: null });
    expect((await request.post('/api/voice/stop')).status()).toBe(409);
  });

  test('Ollama is reported as not running when nothing listens', async ({ request }) => {
    expect(await (await request.get('/api/ollama/models')).json()).toEqual({ running: false, models: [] });
  });

  test('GET /api/metrics returns the performance snapshot', async ({ request }) => {
    const m = await (await request.get('/api/metrics')).json();
    expect(m).toHaveProperty('system');
    expect(m).toHaveProperty('llm');
    expect(m.recording).toBe(false);
  });
});
