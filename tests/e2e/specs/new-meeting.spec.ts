import { test, expect } from '@playwright/test';
import { DEVICES, METRICS, json, mockEnvironment, openApp } from '../helpers';

test.describe('New meeting page', () => {
  test('shows the shell and the start controls', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    await expect(page).toHaveTitle('Meeting Notes');
    await expect(page.getByRole('heading', { name: 'New meeting', level: 1 })).toBeVisible();
    await expect(page.locator('#recBtn')).toHaveText('● Start recording');
    await expect(page.locator('#uploadBtn')).toBeVisible();
    await expect(page.locator('#settingsBtn')).toBeVisible();
    await expect(page.locator('#v-rec')).toBeHidden();
    await expect(page.locator('#v-meeting')).toBeHidden();
  });

  test('shows the empty state when there are no meetings', async ({ page }) => {
    await mockEnvironment(page);
    await page.route('**/api/meetings', r => json(r, []));
    await openApp(page);
    await expect(page.locator('#list')).toContainText('No meetings yet.');
    await expect(page.locator('#count')).toHaveText('0');
  });

  test('picks BlackHole as system audio and the built-in mic by default', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    await expect(page.locator('#mic option')).toHaveCount(DEVICES.length);
    await expect(page.locator('#mic')).toHaveValue('0');
    await expect(page.locator('#sys')).toHaveValue('1');
    await expect(page.locator('#hint')).toHaveText('');
  });

  test('warns when BlackHole is not installed', async ({ page }) => {
    await mockEnvironment(page, { devices: [DEVICES[0], DEVICES[2]] });
    await openApp(page);
    await expect(page.locator('#sys')).toHaveValue('');
    await expect(page.locator('#hint')).toContainText('BlackHole not found');
    await expect(page.locator('#hint')).toHaveClass(/warn/);
  });

  test('explains when the audio devices cannot be listed', async ({ page }) => {
    await mockEnvironment(page);
    await page.route('**/api/devices', r => json(r, { detail: 'PortAudio error' }, 500));
    await openApp(page, '', { devices: false });
    await expect(page.locator('#hint')).toContainText('Cannot list audio devices: PortAudio error');
  });

  test('remembers a chosen device by name across reloads', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    await page.locator('#mic').selectOption({ label: 'USB Headset' });
    await page.locator('#sys').selectOption('');
    await page.reload();
    await expect(page.locator('#mic option').first()).toBeAttached();
    await expect(page.locator('#mic')).toHaveValue('2');
    await expect(page.locator('#sys')).toHaveValue('');
  });

  test('lists the speech-to-text engines with the English-only hint for Parakeet', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    await expect(page.locator('#stt')).toHaveValue(/^parakeet:/);
    await expect(page.locator('#stt option')).toHaveCount(2);
    await expect(page.locator('#sttHint')).toContainText('English only');
    await page.locator('#stt').selectOption({ label: 'Whisper large-v3 turbo (multilingual)' });
    await expect(page.locator('#sttHint')).toHaveText('');
    await page.reload();
    await expect(page.locator('#stt')).toHaveValue(/^whisper:/);
  });

  test('summary model: Ollama models, local-only hint, Claude needs a key', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    const provider = page.locator('#provider');
    await expect(provider).toHaveValue('ollama:qwen2.5:14b');
    await expect(provider.locator('option')).toHaveText([
      'Claude API (no API key set)',
      'Ollama · qwen2.5:14b (9 GB)',
      'Ollama · llama3.1:8b (4.7 GB)',
    ]);
    await expect(page.locator('#providerHint')).toHaveText('Runs on this Mac. Nothing leaves your machine.');
    await provider.selectOption('claude');
    await expect(page.locator('#providerHint')).toContainText('Claude needs an API key');
    await provider.selectOption('ollama:llama3.1:8b');
    await page.reload();
    await expect(provider).toHaveValue('ollama:llama3.1:8b');
  });

  test('summary model: tells the user when Ollama is not running', async ({ page }) => {
    await mockEnvironment(page, { ollama: { running: false, models: [] } });
    await openApp(page);
    await expect(page.locator('#provider')).toHaveValue('ollama');
    await expect(page.locator('#provider option').last()).toHaveText('Ollama (not running, or no models installed)');
    await expect(page.locator('#providerHint')).toContainText('Ollama is not running');
    await expect(page.locator('#providerHint')).toHaveClass(/warn/);
  });

  test('shows a clear error when the server cannot be reached', async ({ page }) => {
    await mockEnvironment(page);
    await page.route('**/api/config', r => r.abort());
    await page.goto('/');
    await expect(page.locator('#v-new')).toContainText('Cannot reach the app server');
  });

  test('refuses to start without a microphone', async ({ page }) => {
    await mockEnvironment(page, { devices: [] });
    let started = false;
    await page.route('**/api/start', r => { started = true; return json(r, { id: 'x' }); });
    await openApp(page, '', { devices: false });
    await expect(page.locator('#hint')).toContainText('BlackHole not found');   // devices have finished loading
    let message = '';
    page.once('dialog', d => { message = d.message(); void d.accept(); });   // alert() fires synchronously inside the click
    await page.locator('#recBtn').click();
    expect(message).toBe('Choose a microphone first.');
    expect(started).toBe(false);
    await expect(page.locator('#recBtn')).toBeEnabled();
  });
});

test.describe('Performance panel and health chips', () => {
  test('is open on a wide screen, shows live numbers and can be closed', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    const toggle = page.locator('#perfToggle');
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await expect(page.locator('#health')).toContainText('CPU 42%');
    await expect(page.locator('#health')).toContainText('RAM 10.5/16 GB');
    await expect(page.locator('#health')).toContainText('GPU 18%');
    await expect(page.locator('#perf')).toContainText('This Mac');
    await expect(page.locator('#perf')).toContainText('plugged in');
    await expect(page.locator('#perf')).toContainText('No summary yet.');
    await page.locator('#perfClose').click();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await expect(page.locator('#perfPanel')).toBeHidden();
    await page.reload();
    await expect(page.locator('#perfToggle')).toHaveAttribute('aria-expanded', 'false');
  });

  test('flags heavy swapping and a missing Ollama model', async ({ page }) => {
    await mockEnvironment(page, { metrics: { ...METRICS, system: { ...METRICS.system, swap_used_gb: 4.2,
      swap_mb_s: 35, mem_pressure: 'warn', mem_status: 'bad', cpu: 95 } } });
    await openApp(page);
    await expect(page.locator('#health .chip.bad', { hasText: 'Memory swapping 35 MB/s' })).toBeVisible();
    await expect(page.locator('#health .chip.bad', { hasText: 'CPU' })).toBeVisible();
    await expect(page.locator('#perf')).toContainText('Short of RAM');
  });

  test('warns on memory pressure without swap traffic', async ({ page }) => {
    await mockEnvironment(page, { metrics: { ...METRICS, system: { ...METRICS.system,
      mem_pressure: 'warn', mem_status: 'warn' } } });
    await openApp(page);
    await expect(page.locator('#health .chip.warn', { hasText: 'Memory warn pressure' })).toBeVisible();
    await expect(page.locator('#perf')).toContainText('RAM is tight');
  });

  test('does not warn about swap left over from earlier when RAM is fine', async ({ page }) => {
    await mockEnvironment(page, { metrics: { ...METRICS, system: { ...METRICS.system, swap_used_gb: 4.5 } } });
    await openApp(page);
    await expect(page.locator('#perf')).toContainText('Swap used');
    await expect(page.locator('#perf')).toContainText('4.5 GB');
    await expect(page.locator('#health')).not.toContainText('Memory');
    await expect(page.locator('#perf .pnote.bad')).toHaveCount(0);
    await expect(page.locator('#perf b.bad')).toHaveCount(0);
  });

  test('is closed by default on a narrow screen and opens as an overlay', async ({ page }) => {
    await page.setViewportSize({ width: 1000, height: 800 });
    await mockEnvironment(page);
    await openApp(page);
    await expect(page.locator('#perfToggle')).toHaveAttribute('aria-expanded', 'false');
    await page.locator('#perfToggle').click();
    await expect(page.locator('#perfPanel')).toBeVisible();
    await expect(page.locator('#perfPanel')).toHaveCSS('position', 'fixed');
  });

  test('hides the health chips on a phone-sized screen', async ({ page }) => {
    await page.setViewportSize({ width: 600, height: 900 });
    await mockEnvironment(page);
    await openApp(page);
    await expect(page.locator('#health')).toBeHidden();
    await expect(page.locator('#recBtn')).toBeVisible();
  });
});
