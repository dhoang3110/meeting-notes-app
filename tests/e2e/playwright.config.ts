import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';

const PORT = Number(process.env.E2E_PORT || 8799);
const ROOT = path.resolve(__dirname, '../..');

export default defineConfig({
  testDir: './specs',
  // Tests share one server and one notes folder, so each test seeds uniquely named meetings.
  fullyParallel: true,
  workers: process.env.CI ? 2 : 4,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    viewport: { width: 1280, height: 800 },
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } } }],
  webServer: {
    command: `${ROOT}/.venv/bin/python ${__dirname}/server.py`,
    url: `http://127.0.0.1:${PORT}/api/config`,
    cwd: ROOT,
    env: { E2E_PORT: String(PORT) },
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
