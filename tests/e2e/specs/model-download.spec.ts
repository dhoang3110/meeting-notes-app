import { test, expect } from '@playwright/test';
import { json, mockEnvironment, openApp } from '../helpers';

const MISSING = { repo: 'mlx-community/parakeet-tdt-0.6b-v2', ready: false, downloading: false, bytes: 0, total: 2.472e9, error: null };

test.describe('Speech model download', () => {
  test('hidden when the model is already on this Mac', async ({ page }) => {
    await mockEnvironment(page);
    await openApp(page);
    await expect(page.locator('#modelCard')).toBeHidden();
    await expect(page.locator('#recBtn')).toBeEnabled();
  });

  test('first run: explains the download, blocks recording, shows progress, then unlocks', async ({ page }) => {
    let state: Record<string, unknown> = MISSING;
    let downloads = 0;
    await mockEnvironment(page);
    await page.route('**/api/model/status*', r => json(r, state));
    await page.route('**/api/model/download', r => {
      downloads++;
      state = { ...MISSING, downloading: true, bytes: 0.62e9 };
      return json(r, state);
    });
    await openApp(page);

    await expect(page.locator('#modelCard')).toBeVisible();
    await expect(page.locator('#modelText')).toContainText('2.47 GB');
    await expect(page.locator('#modelText')).toContainText('never leaves');
    await expect(page.locator('#recBtn')).toBeDisabled();
    await expect(page.locator('#uploadBtn')).toBeDisabled();

    await page.locator('#modelBtn').click();
    await expect(page.locator('#modelBar')).toBeVisible();
    await expect(page.locator('#modelHint')).toContainText('620 MB of 2.47 GB (25%)');
    await expect(page.locator('#modelBtn')).toBeHidden();
    expect(downloads).toBe(1);

    state = { ...MISSING, ready: true, bytes: 2.472e9 };          // the next poll sees it finished
    await expect(page.locator('#modelCard')).toBeHidden({ timeout: 5000 });
    await expect(page.locator('#recBtn')).toBeEnabled();
    await expect(page.locator('#uploadBtn')).toBeEnabled();
  });

  test('a failed download shows the reason and offers a retry', async ({ page }) => {
    await mockEnvironment(page, { model: { ...MISSING, bytes: 0.3e9, error: 'Could not reach the download server. Check your internet connection and try again.' } });
    await openApp(page);
    await expect(page.locator('#modelHint')).toContainText('internet connection');
    await expect(page.locator('#modelBtn')).toHaveText('Retry download');
    await expect(page.locator('#recBtn')).toBeDisabled();
  });
});
