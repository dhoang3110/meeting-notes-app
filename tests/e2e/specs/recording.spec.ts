import { test, expect, type Page } from '@playwright/test';
import { json, mockEnvironment, nextDialog, openApp, seedMeeting } from '../helpers';

/** A fake recorder behind /api/start, /api/status, /api/live and /api/stop (no microphone involved). */
function fakeRecorder(page: Page, id: string) {
  const s = {
    recording: false, elapsed: 0, no_audio: false, liveOn: true, liveError: null as string | null,
    levels: [{ name: 'MacBook Pro Microphone', level: 0.05 }, { name: 'BlackHole 2ch', level: 0.1 }],
    segments: [] as any[], partial: null as any, startBody: null as any, liveAfter: [] as number[], stops: 0,
  };
  const session = { stt: 'Parakeet TDT 0.6B v2', summary: 'Ollama · qwen2.5:14b' };
  page.route('**/api/start', r => { s.startBody = r.request().postDataJSON(); s.recording = true; return json(r, { id }); });
  page.route('**/api/stop', r => { s.recording = false; s.stops++; return json(r, { id }); });
  page.route('**/api/status', r => json(r, {
    recording: s.recording, elapsed: s.elapsed, levels: s.recording ? s.levels : [], current: s.recording ? id : null,
    jobs: {}, no_audio: s.no_audio, session: s.recording ? session : null,
    live: s.recording && s.liveOn ? { transcribed: s.segments.at(-1)?.end ?? 0, error: s.liveError, chunk_sec: 10 } : null,
  }));
  page.route('**/api/live?*', r => {
    const after = Number(new URL(r.request().url()).searchParams.get('after'));
    s.liveAfter.push(after);
    return json(r, { segments: s.segments.slice(after), recording: s.recording, partial: s.partial });
  });
  return s;
}

test.describe('Recording', () => {
  test('starts with the chosen settings and shows the recording screen', async ({ page }) => {
    const id = seedMeeting('rec', { title: 'Recorded' });
    const rec = fakeRecorder(page, id);
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#title').fill('Quarterly review');
    await page.locator('#mic').selectOption('2');
    await page.locator('#recBtn').click();

    await expect(page.locator('#v-rec')).toBeVisible();
    await expect(page.locator('#v-new')).toBeHidden();
    expect(rec.startBody).toEqual({
      devices: [2, 1], title: 'Quarterly review', provider: 'ollama:qwen2.5:14b',
      stt: 'parakeet:mlx-community/parakeet-tdt-0.6b-v2',
    });
    await expect(page.locator('#title')).toHaveValue('');
    await expect(page.locator('#recChips')).toContainText('Speech to text: Parakeet TDT 0.6B v2');
    await expect(page.locator('#recChips')).toContainText('Summary: Ollama · qwen2.5:14b');
    await expect(page.locator('#meters .meter-row')).toHaveCount(2);
    await expect(page.locator('#meters')).toContainText('BlackHole 2ch');
    await expect(page.locator('#noAudio')).toBeHidden();
    await expect(page).toHaveURL(new RegExp(`#${id}$`));
  });

  test('sends an empty title as null and omits system audio when set to None', async ({ page }) => {
    const rec = fakeRecorder(page, seedMeeting('rec2', { title: 'R2' }));
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#sys').selectOption('');
    await page.locator('#recBtn').click();
    await expect(page.locator('#v-rec')).toBeVisible();
    expect(rec.startBody.devices).toEqual([0]);
    expect(rec.startBody.title).toBeNull();
  });

  test('shows the running timer and the browser tab title', async ({ page }) => {
    const rec = fakeRecorder(page, seedMeeting('rec3', { title: 'R3' }));
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#recBtn').click();
    rec.elapsed = 3725;
    await expect(page.locator('#timer')).toHaveText('1:02:05');
    await expect(page).toHaveTitle('● 1:02:05 · Meeting Notes');
    rec.elapsed = 65;
    await expect(page.locator('#timer')).toHaveText('01:05');
  });

  test('warns when no sound is coming in', async ({ page }) => {
    const rec = fakeRecorder(page, seedMeeting('rec4', { title: 'R4' }));
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#recBtn').click();
    await expect(page.locator('#v-rec')).toBeVisible();
    await expect(page.locator('#noAudio')).toBeHidden();
    rec.no_audio = true;
    await expect(page.locator('#noAudio')).toBeVisible();
    await expect(page.locator('#noAudio')).toContainText('No sound is coming in.');
  });

  test('streams the live transcript: Me/Others labels, a grey draft, and only new segments', async ({ page }) => {
    const rec = fakeRecorder(page, seedMeeting('rec5', { title: 'R5' }));
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#recBtn').click();
    await expect(page.locator('#liveEmpty')).toContainText('Listening');

    rec.segments = [
      { start: 3, end: 8, text: 'Hello <b>team</b>', speaker: 'Me' },
      { start: 9, end: 14, text: 'Hi, can you hear me?', speaker: 'Others' },
    ];
    rec.partial = { start: 15, text: 'draft words here', speaker: 'Others' };
    const live = page.locator('#liveText');
    await expect(live.locator('p:not(#livePartial):not(.live-empty)')).toHaveCount(2);
    await expect(live.locator('.who.me')).toHaveText('Me');
    await expect(live.locator('.who:not(.me)').first()).toHaveText('Others');
    await expect(live.locator('.ts').first()).toHaveText('00:03');
    // The transcript is text, never HTML.
    await expect(live).toContainText('Hello <b>team</b>');
    await expect(live.locator('b')).toHaveCount(0);
    await expect(page.locator('#livePartial')).toContainText('draft words here');
    await expect(page.locator('#liveEmpty')).toHaveCount(0);
    await expect(page.locator('#liveStatus')).toContainText('Final text up to 00:14');

    rec.partial = null;
    await expect(page.locator('#livePartial')).toBeEmpty();
    // The client asks only for segments it has not rendered yet, so none are duplicated.
    await expect.poll(() => rec.liveAfter.includes(2)).toBe(true);
    await expect(live.locator('p:not(#livePartial):not(.live-empty)')).toHaveCount(2);
  });

  test('says so when live transcription is off or has failed', async ({ page }) => {
    const rec = fakeRecorder(page, seedMeeting('rec6', { title: 'R6' }));
    await mockEnvironment(page);
    await openApp(page);
    rec.liveOn = false;
    await page.locator('#recBtn').click();
    await expect(page.locator('#liveStatus')).toContainText('Live transcription is off');
    rec.liveOn = true; rec.liveError = 'model crashed';
    await expect(page.locator('#liveStatus')).toContainText('Live transcription failed');
  });

  test('Stop saves and opens the finished meeting', async ({ page }) => {
    const id = seedMeeting('rec7', { title: 'Finished meeting' });
    const rec = fakeRecorder(page, id);
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#recBtn').click();
    await expect(page.locator('#v-rec')).toBeVisible();
    await page.locator('#stopBtn').click();
    await expect(page.locator('#v-meeting .mtop h1')).toHaveText('Finished meeting');
    await expect(page.locator('.doc')).toBeVisible();
    await expect(page.locator('#v-rec')).toBeHidden();
    await expect(page).toHaveTitle('Meeting Notes');
    expect(rec.stops).toBe(1);
  });

  test('a page reload during a recording returns to the recording screen', async ({ page }) => {
    const rec = fakeRecorder(page, seedMeeting('rec8', { title: 'R8' }));
    rec.recording = true; rec.elapsed = 42;
    await mockEnvironment(page);
    await openApp(page);
    await expect(page.locator('#v-rec')).toBeVisible();
    await expect(page.locator('#timer')).toHaveText('00:42');
    await page.locator('#newBtn').click();
    await expect(page.locator('#v-rec')).toBeVisible();   // "+ New" must not hide a recording in progress
    await expect(page.locator('#v-new')).toBeHidden();
  });

  test('shows the server error when recording cannot start', async ({ page }) => {
    fakeRecorder(page, seedMeeting('rec9', { title: 'R9' }));
    await mockEnvironment(page);
    await page.route('**/api/start', r => json(r, { detail: 'Could not start recording: device busy' }, 500));
    await openApp(page);
    const message = nextDialog(page);
    await page.locator('#recBtn').click();
    expect(await message).toBe('Could not start recording: device busy');
    await expect(page.locator('#v-new')).toBeVisible();
    await expect(page.locator('#recBtn')).toBeEnabled();
  });
});

test.describe('Test audio', () => {
  function fakeMonitor(page: Page) {
    const s = { startBody: null as any, stops: 0, active: true, seconds: 5, heard: [true, true] };
    page.route('**/api/monitor/start', r => { s.startBody = r.request().postDataJSON(); return json(r, { ok: true }); });
    page.route('**/api/monitor/stop', r => { s.stops++; return json(r, { ok: true }); });
    page.route('**/api/monitor', r => json(r, {
      active: s.active, seconds: s.seconds,
      levels: [{ name: 'MacBook Pro Microphone', level: 0.1, heard: s.heard[0] }, { name: 'BlackHole 2ch', level: 0.02, heard: s.heard[1] }],
    }));
    return s;
  }

  test('shows a meter per input and confirms when everything hears sound', async ({ page }) => {
    const mon = fakeMonitor(page);
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#testBtn').click();
    await expect(page.locator('#testBtn')).toHaveText('Stop test');
    expect(mon.startBody).toEqual({ devices: [0, 1] });
    await expect(page.locator('#testMeters .meter-row')).toHaveCount(2);
    await expect(page.locator('#testHint')).toHaveText('✓ Every input hears sound.');
    await page.locator('#testBtn').click();
    await expect(page.locator('#testBtn')).toHaveText('Test audio');
    await expect(page.locator('#testMeters')).toBeEmpty();
    await expect(page.locator('#testHint')).toHaveText('Check both inputs before you start.');
    expect(mon.stops).toBe(1);
  });

  test('names the input that is silent', async ({ page }) => {
    const mon = fakeMonitor(page);
    mon.heard = [true, false];
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#testBtn').click();
    await expect(page.locator('#testHint')).toContainText('No sound yet on BlackHole 2ch');
    await expect(page.locator('#testMeters .bar i.warn')).toHaveCount(1);
  });

  test('resets itself when the server ends the test', async ({ page }) => {
    const mon = fakeMonitor(page);
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#testBtn').click();
    await expect(page.locator('#testBtn')).toHaveText('Stop test');
    mon.active = false;
    await expect(page.locator('#testBtn')).toHaveText('Test audio');
  });

  test('restarts the test with the new device when the selection changes', async ({ page }) => {
    const mon = fakeMonitor(page);
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#testBtn').click();
    await expect(page.locator('#testBtn')).toHaveText('Stop test');
    await page.locator('#mic').selectOption('2');
    await expect.poll(() => mon.startBody).toEqual({ devices: [2, 1] });
  });

  test('shows the error when an input cannot be opened', async ({ page }) => {
    await mockEnvironment(page);
    await page.route('**/api/monitor/start', r => json(r, { detail: 'Could not open the input device: busy' }, 500));
    await openApp(page);
    await page.locator('#testBtn').click();
    await expect(page.locator('#testHint')).toHaveText('Could not open the input device: busy');
    await expect(page.locator('#testBtn')).toHaveText('Test audio');
  });
});

test.describe('Upload', () => {
  test('uploads an audio file with the chosen models and opens the new meeting', async ({ page }) => {
    const id = seedMeeting('upl', { title: 'Uploaded call' });
    let body = '';
    await mockEnvironment(page);
    await page.route('**/api/upload', r => { body = r.request().postData() ?? ''; return json(r, { id }); });
    await openApp(page);
    await page.locator('#title').fill('Customer call');
    await page.locator('#file').setInputFiles({ name: 'call.m4a', mimeType: 'audio/mp4', buffer: Buffer.from('fake-audio') });
    await expect(page.locator('#v-meeting .mtop h1')).toHaveText('Uploaded call');
    expect(body).toContain('filename="call.m4a"');
    expect(body).toMatch(/name="title"\r\n\r\nCustomer call/);
    expect(body).toMatch(/name="provider"\r\n\r\nollama:qwen2.5:14b/);
    expect(body).toMatch(/name="stt"\r\n\r\nparakeet:/);
    await expect(page.locator('#title')).toHaveValue('');
  });

  test('shows the server error for a rejected upload', async ({ page }) => {
    await mockEnvironment(page);
    await page.route('**/api/upload', r => json(r, { detail: "Unknown speech-to-text model: 'x'" }, 400));
    await openApp(page);
    const message = nextDialog(page);
    await page.locator('#file').setInputFiles({ name: 'a.wav', mimeType: 'audio/wav', buffer: Buffer.from('x') });
    expect(await message).toContain('Unknown speech-to-text model');
    await expect(page.locator('#v-new')).toBeVisible();
  });
});
