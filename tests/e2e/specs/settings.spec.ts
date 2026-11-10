import { test, expect, type Page } from '@playwright/test';
import { json, mockEnvironment, openApp } from '../helpers';

/** Stubs the settings endpoints: the real ones write the API key to the project's .env file. */
function fakeSettings(page: Page, init: { has_key?: boolean } = {}) {
  const s = {
    has_key: init.has_key ?? false, key_hint: init.has_key ? 'sk-ant-…abcd' : '', claude_model: 'claude-sonnet-5-5',
    saved: [] as any[], tested: [] as any[], testResult: { status: 200, body: { ok: true } as any },
    saveError: null as string | null,
  };
  const view = () => ({ has_key: s.has_key, key_hint: s.key_hint, claude_model: s.claude_model });
  page.route('**/api/settings', r => {
    if (r.request().method() === 'GET') return json(r, view());
    const b = r.request().postDataJSON();
    s.saved.push(b);
    if (s.saveError) return json(r, { detail: s.saveError }, 400);
    if (b.anthropic_api_key !== undefined) { s.has_key = !!b.anthropic_api_key; s.key_hint = s.has_key ? 'sk-ant-…wxyz' : ''; }
    if (b.claude_model) s.claude_model = b.claude_model;
    return json(r, view());
  });
  page.route('**/api/settings/test', r => { s.tested.push(r.request().postDataJSON()); return json(r, s.testResult.body, s.testResult.status); });
  return s;
}

/** A fake voice-enrolment backend. */
function fakeVoice(page: Page) {
  const s = { enrolled: false, sampling: null as null | { seconds: number; level: number; device: string }, calls: [] as string[], stopError: null as string | null };
  const view = () => ({ enrolled: s.enrolled, created: s.enrolled ? 1_762_400_000 : null, threshold: null, sampling: s.sampling });
  page.route('**/api/voice', r => {
    if (r.request().method() === 'DELETE') { s.calls.push('delete'); s.enrolled = false; }
    return json(r, view());
  });
  page.route('**/api/voice/start', r => { s.calls.push('start'); s.sampling = { seconds: 5, level: 0.05, device: 'MacBook Pro Microphone' }; return json(r, view()); });
  page.route('**/api/voice/stop', r => {
    s.calls.push('stop');
    if (s.stopError) return json(r, { detail: s.stopError }, 400);
    s.sampling = null; s.enrolled = true; return json(r, view());
  });
  page.route('**/api/voice/cancel', r => { s.calls.push('cancel'); s.sampling = null; return json(r, view()); });
  return s;
}

const open = async (page: Page) => { await page.locator('#settingsBtn').click(); await expect(page.locator('#settings')).toBeVisible(); };

test.describe('Settings: Anthropic API key', () => {
  test('with no key saved: explains where it is stored and hides Remove', async ({ page }) => {
    fakeSettings(page); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await expect(page.locator('#keyHint')).toContainText('No key saved');
    await expect(page.locator('#keyRemove')).toBeHidden();
    await expect(page.locator('#claudeModel')).toHaveValue('claude-sonnet-5-5');
    await expect(page.locator('#apiKey')).toHaveAttribute('type', 'password');
  });

  test('with a saved key: shows only the masked hint and offers Remove', async ({ page }) => {
    fakeSettings(page, { has_key: true }); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await expect(page.locator('#keyHint')).toContainText('Saved key: sk-ant-…abcd');
    await expect(page.locator('#apiKey')).toHaveValue('');
    await expect(page.locator('#keyRemove')).toBeVisible();
  });

  test('saves a key and model and confirms', async ({ page }) => {
    const s = fakeSettings(page); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#apiKey').fill('sk-ant-test-key-1234567890-wxyz');
    await page.locator('#claudeModel').fill('claude-opus-5-5');
    await page.locator('#settingsSave').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Saved.');
    await expect(page.locator('#settingsMsg')).toHaveClass(/ok/);
    expect(s.saved[0]).toEqual({ claude_model: 'claude-opus-5-5', anthropic_api_key: 'sk-ant-test-key-1234567890-wxyz' });
    await expect(page.locator('#keyHint')).toContainText('sk-ant-…wxyz');
    await expect(page.locator('#apiKey')).toHaveValue('');      // the key is never kept in the field
    await expect(page.locator('#keyRemove')).toBeVisible();
  });

  test('leaving the key field empty does not send a key', async ({ page }) => {
    const s = fakeSettings(page, { has_key: true }); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#settingsSave').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Saved.');
    expect(s.saved[0]).not.toHaveProperty('anthropic_api_key');
  });

  test('shows the server validation error', async ({ page }) => {
    const s = fakeSettings(page); fakeVoice(page);
    s.saveError = 'That does not look like an Anthropic API key (it should start with sk-ant-).';
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#apiKey').fill('nope');
    await page.locator('#settingsSave').click();
    await expect(page.locator('#settingsMsg')).toContainText('does not look like an Anthropic API key');
    await expect(page.locator('#settingsMsg')).toHaveClass(/bad/);
  });

  test('Test key reports success and failure', async ({ page }) => {
    const s = fakeSettings(page); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#apiKey').fill('sk-ant-typed');
    await page.locator('#keyTest').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Key works.');
    expect(s.tested[0]).toEqual({ anthropic_api_key: 'sk-ant-typed' });
    s.testResult = { status: 400, body: { detail: 'Anthropic rejected this key.' } };
    await page.locator('#keyTest').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Anthropic rejected this key.');
    await expect(page.locator('#settingsMsg')).toHaveClass(/bad/);
  });

  test('Remove key asks for confirmation first', async ({ page }) => {
    const s = fakeSettings(page, { has_key: true }); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    page.once('dialog', d => void d.dismiss());
    await page.locator('#keyRemove').click();
    expect(s.saved).toHaveLength(0);
    page.once('dialog', d => { expect(d.message()).toBe('Remove the saved Anthropic API key?'); void d.accept(); });
    await page.locator('#keyRemove').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Key removed.');
    expect(s.saved[0]).toEqual({ anthropic_api_key: '' });
    await expect(page.locator('#keyRemove')).toBeHidden();
    await expect(page.locator('#keyHint')).toContainText('No key saved');
  });

  test('saving a key makes Claude selectable without the warning', async ({ page }) => {
    // Needs a server whose /api/config reports claude_ready: simulate that after Save.
    fakeSettings(page); fakeVoice(page);
    await mockEnvironment(page);
    let ready = false;
    await page.route('**/api/config', async r => {
      const res = await r.fetch();
      const body = await res.json();
      return json(r, { ...body, claude_ready: ready });
    });
    await openApp(page);
    await expect(page.locator('#provider option[value="claude"]')).toHaveText('Claude API (no API key set)');
    await open(page);
    ready = true;
    await page.locator('#apiKey').fill('sk-ant-test-key-1234567890-wxyz');
    await page.locator('#settingsSave').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Saved.');
    await expect(page.locator('#provider option[value="claude"]')).toHaveText('Claude API');
  });

  test('closes with the Close button and with Escape', async ({ page }) => {
    fakeSettings(page); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#settingsClose').click();
    await expect(page.locator('#settings')).toBeHidden();
    await open(page);
    await page.keyboard.press('Escape');
    await expect(page.locator('#settings')).toBeHidden();
  });
});

test.describe('Settings: My voice', () => {
  test('not enrolled: invites the user to record, no Remove button', async ({ page }) => {
    fakeSettings(page); fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await expect(page.locator('#voiceHint')).toContainText('Optional. Record about 30 seconds');
    await expect(page.locator('#voiceStart')).toHaveText('Record my voice');
    await expect(page.locator('#voiceDelete')).toBeHidden();
    await expect(page.locator('#voicePrompt')).toBeHidden();
  });

  test('enrolment: read the prompt, Save unlocks after the minimum time, then it is enrolled', async ({ page }) => {
    fakeSettings(page); const v = fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#voiceStart').click();
    await expect(page.locator('#voicePrompt')).toBeVisible();
    await expect(page.locator('#voicePrompt')).toContainText('Please read this aloud');
    await expect(page.locator('#voiceMeter')).toBeVisible();
    await expect(page.locator('#voiceStart')).toBeHidden();
    await expect(page.locator('#voiceStop')).toBeDisabled();
    await expect(page.locator('#voiceHint')).toContainText('Recording from MacBook Pro Microphone… 5 s. Keep reading, at least 20 s more.');
    v.sampling = { seconds: 31, level: 0.1, device: 'MacBook Pro Microphone' };
    await expect(page.locator('#voiceStop')).toBeEnabled();
    await expect(page.locator('#voiceHint')).toContainText('Press Save when you are done.');
    await page.locator('#voiceStop').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Voice saved.');
    await expect(page.locator('#voiceHint')).toContainText('Your voice is saved on this Mac');
    await expect(page.locator('#voiceStart')).toHaveText('Record again');
    await expect(page.locator('#voiceDelete')).toBeVisible();
    expect(v.calls).toEqual(['start', 'stop']);
  });

  test('Cancel abandons the sample', async ({ page }) => {
    fakeSettings(page); const v = fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#voiceStart').click();
    await page.locator('#voiceCancel').click();
    await expect(page.locator('#voicePrompt')).toBeHidden();
    await expect(page.locator('#voiceStart')).toBeVisible();
    expect(v.calls).toEqual(['start', 'cancel']);
  });

  test('closing the dialog while recording a sample releases the microphone', async ({ page }) => {
    fakeSettings(page); const v = fakeVoice(page);
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#voiceStart').click();
    await expect(page.locator('#voiceStop')).toBeVisible();
    await page.locator('#settingsClose').click();
    await expect.poll(() => v.calls).toContain('cancel');
  });

  test('a sample that is too short shows the error and stays usable', async ({ page }) => {
    fakeSettings(page); const v = fakeVoice(page);
    v.stopError = 'Not enough speech in that sample.';
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await page.locator('#voiceStart').click();
    v.sampling = { seconds: 30, level: 0.1, device: 'MacBook Pro Microphone' };
    await expect(page.locator('#voiceStop')).toBeEnabled();
    await page.locator('#voiceStop').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Not enough speech in that sample.');
    await expect(page.locator('#settingsMsg')).toHaveClass(/bad/);
  });

  test('Remove deletes the saved voice after confirmation', async ({ page }) => {
    fakeSettings(page); const v = fakeVoice(page);
    v.enrolled = true;
    await mockEnvironment(page);
    await openApp(page);
    await open(page);
    await expect(page.locator('#voiceDelete')).toBeVisible();
    page.once('dialog', d => { expect(d.message()).toBe('Remove your saved voice from this Mac?'); void d.accept(); });
    await page.locator('#voiceDelete').click();
    await expect(page.locator('#settingsMsg')).toHaveText('Voice removed.');
    await expect(page.locator('#voiceDelete')).toBeHidden();
    expect(v.calls).toEqual(['delete']);
  });
});
