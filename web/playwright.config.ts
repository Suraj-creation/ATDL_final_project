import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const python = process.env.ATDL_PYTHON ?? 'python';
const baseURL = 'http://127.0.0.1:8765';

export default defineConfig({
  testDir: './tests',
  timeout: 65000,
  expect: { timeout: 16000 },
  workers: 1,
  use: {
    baseURL,
    channel: process.env.ATDL_BROWSER ?? 'chrome',
    viewport: { width: 1440, height: 900 },
    permissions: ['camera'],
    launchOptions: { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] },
  },
  webServer: {
    cwd: fileURLToPath(new URL('..', import.meta.url)),
    command: `"${python}" -m uvicorn src.atdl.web.app:app --port 8765 --ws-max-size 12582912`,
    url: baseURL + '/api/health',
    timeout: 65000,
    reuseExistingServer: !process.env.CI && process.env.ATDL_DEVICE !== 'cuda',
    env: { ATDL_DEVICE: process.env.ATDL_DEVICE ?? 'cpu' },
  },
});
