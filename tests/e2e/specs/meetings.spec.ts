import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { json, meetingItem, mockEnvironment, nextDialog, openApp, seedMeeting, TRASH, NOTES } from '../helpers';

const labelFor = (id: string) => `Meeting ${id.slice(11, 13)}:${id.slice(13, 15)}`;

test.beforeEach(async ({ page }) => { await mockEnvironment(page); });

/** Answers every confirm/alert and records what was shown. */
function autoDialogs(page: Page) {
  const seen: string[] = [];
  page.on('dialog', d => { seen.push(d.message()); void d.accept(); });
  return seen;
}

test.describe('Meetings list', () => {
  test('groups meetings under Today, Yesterday and Earlier', async ({ page }) => {
    const today = seedMeeting('today', { title: 'Standup today' });
    const yest = seedMeeting('yest', { title: 'Standup yesterday', daysAgo: 1 });
    const old = seedMeeting('old', { title: 'Standup last week', daysAgo: 5, duration_sec: 1500 });
    await openApp(page);
    for (const g of ['Today', 'Yesterday', 'Earlier']) await expect(page.locator('.group', { hasText: g })).toBeVisible();
    await expect(meetingItem(page, today)).toContainText('Standup today');
    await expect(meetingItem(page, yest)).toContainText('Standup yesterday');
    // Older meetings show their date, and the duration when known.
    await expect(meetingItem(page, old).locator('.m')).toContainText(/\d\d-\d\d · \d\d:\d\d · 25 min/);
    const count = await page.locator('.item').count();
    await expect(page.locator('#count')).toHaveText(String(count));
  });

  test('falls back to "Meeting HH:MM" when a meeting has no title', async ({ page }) => {
    const id = seedMeeting('untitled');
    await openApp(page);
    await expect(meetingItem(page, id).locator('.t')).toHaveText(labelFor(id));
  });

  test('filters by title and shows a no-match message', async ({ page }) => {
    const a = seedMeeting('a', { title: 'Zebra planning' });
    const b = seedMeeting('b', { title: 'Giraffe retro' });
    await openApp(page);
    await page.locator('#search').fill('zebra');
    await expect(meetingItem(page, a)).toBeVisible();
    await expect(meetingItem(page, b)).toHaveCount(0);
    await page.locator('#search').fill('no-such-meeting-xyz');
    await expect(page.locator('#list')).toContainText('No meetings match your search.');
    await page.locator('#search').fill('');
    await expect(meetingItem(page, b)).toBeVisible();
  });

  test('shows a status pill for each processing stage', async ({ page }) => {
    const base = { date: '', duration: '', has_transcript: false, error: null };
    await page.route('**/api/meetings', r => json(r, [
      { ...base, id: '2026-11-06_100001_a', title: 'Case-rec', has_notes: false, stage: 'recording' },
      { ...base, id: '2026-11-06_100002_b', title: 'Case-save', has_notes: false, stage: 'saving audio' },
      { ...base, id: '2026-11-06_100003_c', title: 'Case-trans', has_notes: false, stage: 'transcribing' },
      { ...base, id: '2026-11-06_100004_d', title: 'Case-summ', has_notes: false, stage: 'summarising with Claude' },
      { ...base, id: '2026-11-06_100005_e', title: 'Case-queue', has_notes: false, stage: 'queued' },
      { ...base, id: '2026-11-06_100006_f', title: 'Case-bad', has_notes: false, stage: 'error', error: 'boom' },
      { ...base, id: '2026-11-06_100007_g', title: 'Case-empty', has_notes: false, stage: null },
      { ...base, id: '2026-11-06_100008_h', title: 'Case-done', has_notes: true, stage: null },
    ]));
    await openApp(page);
    const pill = (t: string) => page.locator('.item', { hasText: t }).locator('.pill');
    await expect(pill('Case-rec')).toHaveText('Recording');
    await expect(pill('Case-save')).toHaveText('Saving');
    await expect(pill('Case-trans')).toHaveText('Transcribing');
    await expect(pill('Case-summ')).toHaveText('Summarising');
    await expect(pill('Case-queue')).toHaveText('Summarising');
    await expect(pill('Case-bad')).toHaveText('Failed');
    await expect(pill('Case-empty')).toHaveText('No notes');
    await expect(page.locator('.item', { hasText: 'Case-done' }).locator('.pill')).toHaveCount(0);
  });
});

test.describe('Viewing a meeting', () => {
  test('opens the notes, renders Markdown and sanitises HTML', async ({ page }) => {
    const id = seedMeeting('notes', { title: 'Roadmap sync', duration_sec: 720 });
    await openApp(page);
    await meetingItem(page, id).click();
    await expect(page).toHaveURL(new RegExp(`#${id}$`));
    await expect(page.locator('#v-meeting .mtop h1')).toHaveText('Roadmap sync');
    await expect(page.locator('#v-meeting .meta')).toHaveText('Thu 06 Nov 2026, 10:00 · 12 min');
    const doc = page.locator('.doc');
    await expect(doc.getByRole('heading', { name: 'Summary' })).toBeVisible();
    await expect(doc.locator('strong')).toHaveText('beta');
    await expect(doc.locator('input[type=checkbox]')).toHaveCount(2);
    await expect(doc.locator('input[type=checkbox]').nth(1)).toBeChecked();
    // Untrusted HTML inside the notes must never run.
    await expect(doc.locator('script')).toHaveCount(0);
    await expect(doc.locator('[onerror]')).toHaveCount(0);   // DOMPurify keeps <img> but strips the handler
    expect(await page.evaluate(() => (window as any).__pwned)).toBeUndefined();
    await expect(meetingItem(page, id)).toHaveClass(/active/);
  });

  test('opens a meeting straight from the URL hash', async ({ page }) => {
    const id = seedMeeting('deeplink', { title: 'Deep link target' });
    await openApp(page, `#${id}`);
    await expect(page.locator('#v-meeting .mtop h1')).toHaveText('Deep link target');
    await expect(page.locator('#v-new')).toBeHidden();
  });

  test('ignores a hash that matches no meeting', async ({ page }) => {
    await openApp(page, '#2000-01-01_000000_missing');
    await expect(page.locator('#v-new')).toBeVisible();
  });

  test('"+ New" returns to the new-meeting page', async ({ page }) => {
    const id = seedMeeting('back', { title: 'Back test' });
    await openApp(page, `#${id}`);
    await expect(page.locator('#v-meeting .mtop h1')).toBeVisible();
    await page.locator('#newBtn').click();
    await expect(page.locator('#v-new')).toBeVisible();
    await expect(page).not.toHaveURL(/#/);
  });

  test('a meeting without notes offers Re-summarise and disables Copy', async ({ page }) => {
    const id = seedMeeting('nonotes', { title: 'Nothing yet', notes: null });
    await openApp(page, `#${id}`);
    await expect(page.locator('#v-meeting')).toContainText('No notes for this meeting yet.');
    await expect(page.locator('#btnCopy')).toBeDisabled();
    await expect(page.locator('#btnResum')).toBeEnabled();
  });

  test('a failed meeting shows the error and keeps Re-summarise available', async ({ page }) => {
    const row = { id: '2026-11-06_110000_fail', title: 'Failed one', date: '', duration: '', has_notes: false,
                  has_transcript: false, stage: 'error', error: 'ConnectionError: Ollama is down' };
    await page.route('**/api/meetings', r => json(r, [row]));
    await openApp(page, `#${row.id}`);
    await expect(page.locator('#v-meeting .alert')).toContainText('Processing failed.');
    await expect(page.locator('#v-meeting .alert')).toContainText('ConnectionError: Ollama is down');
    await expect(page.locator('#btnResum')).toBeVisible();
  });

  test('a meeting being processed shows the pipeline steps', async ({ page }) => {
    const row = { id: '2026-11-06_110001_busy', title: 'Busy one', date: '', duration: '', has_notes: false,
                  has_transcript: false, stage: 'summarising with Ollama (qwen2.5:14b)', error: null };
    await page.route('**/api/meetings', r => json(r, [row]));
    await openApp(page, `#${row.id}`);
    const steps = page.locator('.steps li');
    await expect(steps).toHaveCount(3);
    await expect(steps.nth(0)).toHaveClass(/done/);
    await expect(steps.nth(1)).toHaveClass(/done/);
    await expect(steps.nth(2)).toHaveClass(/active/);
    await expect(steps.nth(2)).toContainText('with Ollama (qwen2.5:14b)');
    await expect(page.locator('#btnResum')).toHaveCount(0);
  });
});

test.describe('Meeting actions', () => {
  test('Re-summarise posts the chosen model and reloads the meeting', async ({ page }) => {
    const id = seedMeeting('resum', { title: 'Resum me' });
    let body: any;
    await page.route(`**/api/meetings/${id}/summarize`, r => { body = r.request().postDataJSON(); return json(r, { ok: true }); });
    await openApp(page, `#${id}`);
    await page.locator('#resumeModel').selectOption('ollama:llama3.1:8b');
    await page.locator('#btnResum').click();
    await expect.poll(() => body).toEqual({ provider: 'ollama:llama3.1:8b' });
    await expect(page.locator('.doc')).toBeVisible();
  });

  test('Re-summarise shows the server error and re-enables the button', async ({ page }) => {
    const id = seedMeeting('resumfail', { title: 'Resum fail' });
    await page.route(`**/api/meetings/${id}/summarize`, r => json(r, { detail: 'Unknown summary provider' }, 400));
    await openApp(page, `#${id}`);
    const message = nextDialog(page);
    await page.locator('#btnResum').click();
    expect(await message).toBe('Unknown summary provider');
    await expect(page.locator('#btnResum')).toBeEnabled();
  });

  test('Copy Markdown puts the raw notes on the clipboard', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    const id = seedMeeting('copy', { title: 'Copy me' });
    await openApp(page, `#${id}`);
    await page.locator('#btnCopy').click();
    await expect(page.locator('#btnCopy')).toHaveText('Copied ✓');
    const clip = await page.evaluate(() => navigator.clipboard.readText());
    expect(clip).toContain('## Summary');
    expect(clip).toContain('We agreed to ship the **beta** next week.');
    await expect(page.locator('#btnCopy')).toHaveText('Copy Markdown', { timeout: 4000 });
  });

  test('Show in Finder calls the reveal endpoint', async ({ page }) => {
    const id = seedMeeting('reveal', { title: 'Reveal me' });
    let hit = false;
    await page.route(`**/api/meetings/${id}/reveal`, r => { hit = true; return json(r, { path: 'x' }); });
    await openApp(page, `#${id}`);
    await page.locator('#btnReveal').click();
    await expect.poll(() => hit).toBe(true);
  });
});

test.describe('Deleting meetings', () => {
  test('the Delete button asks first, then moves the folder to the Trash', async ({ page }) => {
    const id = seedMeeting('del', { title: 'Delete me' });
    const dialogs = autoDialogs(page);
    await openApp(page, `#${id}`);
    await page.locator('#btnDelete').click();
    await expect(meetingItem(page, id)).toHaveCount(0);
    await expect(page.locator('#v-new')).toBeVisible();
    expect(dialogs[0]).toContain('Delete “Delete me”?');
    expect(fs.existsSync(path.join(NOTES, id))).toBe(false);
    expect(fs.existsSync(path.join(TRASH, id, 'notes.md'))).toBe(true);
  });

  test('declining the confirmation keeps the meeting', async ({ page }) => {
    const id = seedMeeting('keep', { title: 'Keep me' });
    page.on('dialog', d => void d.dismiss());
    await openApp(page, `#${id}`);
    await page.locator('#btnDelete').click();
    await expect(meetingItem(page, id)).toBeVisible();
    expect(fs.existsSync(path.join(NOTES, id))).toBe(true);
  });

  test('the trash icon in the list deletes without opening the meeting', async ({ page }) => {
    const id = seedMeeting('rowdel', { title: 'Row delete' });
    autoDialogs(page);
    await openApp(page);
    await meetingItem(page, id).hover();
    await page.locator(`.del[data-del="${id}"]`).click();
    await expect(meetingItem(page, id)).toHaveCount(0);
    expect(fs.existsSync(path.join(TRASH, id))).toBe(true);
  });

  test('select mode deletes several meetings at once', async ({ page }) => {
    const a = seedMeeting('multi-a', { title: 'Multi A' });
    const b = seedMeeting('multi-b', { title: 'Multi B' });
    const keep = seedMeeting('multi-keep', { title: 'Multi keep' });
    const dialogs = autoDialogs(page);
    await openApp(page);
    await page.locator('#selBtn').click();
    await expect(page.locator('#selBtn')).toHaveText('Cancel');
    await expect(page.locator('#selDelete')).toBeDisabled();
    await meetingItem(page, a).click();
    await meetingItem(page, b).click();
    await expect(page.locator('#selCount')).toHaveText('2 selected');
    await expect(page.locator('#selDelete')).toHaveText('Delete (2)');
    await expect(meetingItem(page, a)).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#selDelete').click();
    await expect(meetingItem(page, a)).toHaveCount(0);
    await expect(meetingItem(page, b)).toHaveCount(0);
    await expect(meetingItem(page, keep)).toBeVisible();
    expect(dialogs[0]).toContain('Delete 2 meetings?');
    await expect(page.locator('#selBtn')).toHaveText('Select');
  });

  test('"Select all" toggles every visible meeting and respects the search filter', async ({ page }) => {
    // Mocked list: never "select all" against the real notes folder.
    const mk = (n: string, t: string) => ({ id: `2026-11-06_12000${n}_x`, title: t, date: '', duration: '', has_notes: true,
                                            has_transcript: true, stage: null, error: null });
    await page.route('**/api/meetings', r => json(r, [mk('1', 'Alpha one'), mk('2', 'Alpha two'), mk('3', 'Beta')]));
    await openApp(page);
    await page.locator('#selBtn').click();
    await page.locator('#search').fill('alpha');
    await page.locator('#selAll').click();
    await expect(page.locator('#selCount')).toHaveText('2 selected');
    await expect(page.locator('#selAll')).toHaveText('Clear');
    await page.locator('#selAll').click();
    await expect(page.locator('#selCount')).toHaveText('0 selected');
    await page.locator('#selBtn').click();
    await expect(page.locator('.meetings')).not.toHaveClass(/selecting/);
  });

  test('reports a delete that the server refuses', async ({ page }) => {
    const id = seedMeeting('busy', { title: 'Busy meeting' });
    await page.route(`**/api/meetings/${id}`, r =>
      r.request().method() === 'DELETE' ? json(r, { detail: 'This meeting is still being recorded or processed.' }, 409) : r.continue());
    const dialogs = autoDialogs(page);
    await openApp(page, `#${id}`);
    await page.locator('#btnDelete').click();
    await expect.poll(() => dialogs.length).toBe(2);
    expect(dialogs[1]).toBe('This meeting is still being recorded or processed.');
    await expect(meetingItem(page, id)).toBeVisible();
  });
});
