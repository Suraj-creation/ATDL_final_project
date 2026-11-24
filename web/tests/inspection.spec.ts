import { expect, test } from '@playwright/test';

test('recorded video, exact-frame trace, thresholds, model comparison, and export', async ({ page, request }) => {
  const health = await (await request.get('/api/health')).json();
  expect(health.ready).toBe(true);
  if (process.env.ATDL_DEVICE === 'cuda') expect(health.device).toMatch(/^cuda/);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Steel Inspection' })).toBeVisible();
  await expect(page.getByText('Browser round trip')).toBeVisible();
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect(page.getByText(/[1-9][0-9]* analyzed ·/)).toBeVisible();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.getByRole('slider', { name: 'Seek clip' }).fill('2.25');
  await expect(page.getByTestId('frame-time')).toHaveText('00:02.25');
  await page.getByRole('button', { name: 'Preprocess', exact: true }).click();
  await expect(page.getByRole('img', { name: 'Actual normalized and padded tensor rendered as an image' })).toBeVisible();
  await page.getByRole('button', { name: 'Model', exact: true }).click();
  await page.getByRole('button', { name: 'Capture activations' }).click();
  await expect(page.getByRole('img', { name: /Mean absolute activation/ }).first()).toBeVisible();
  let predictions = 0;
  page.on('request', request => { if (request.url().includes('/api/predict')) predictions++; });
  await page.getByText('Model and thresholds', { exact: true }).click();
  await page.getByRole('slider', { name: 'Confidence', exact: true }).fill('1');
  await expect(page.getByTestId('box-count')).toHaveText('0 visible');
  expect(predictions).toBe(0);
  await page.getByRole('button', { name: 'Reset', exact: true }).click();
  await page.getByText('Per-class thresholds', { exact: true }).click();
  const crazing = page.getByRole('spinbutton', { name: 'Crazing confidence threshold' });
  await crazing.fill('0.2');
  await expect(crazing).toBeVisible();
  await page.getByRole('button', { name: 'Show annotations' }).click();
  await expect(page.getByLabel('Annotation matching at IoU 0.50')).toBeVisible();
  await page.getByRole('button', { name: 'Compare models', exact: true }).click();
  await expect(page.locator('.split-label')).toHaveCount(2);
  await expect(page.locator('.split-label').last()).toContainText('Standard YOLO26n');
  await page.getByRole('button', { name: 'Output', exact: true }).click();
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export', exact: true }).click();
  expect((await downloaded).suggestedFilename()).toBe('atdl-inspection.json');
  const sourceImage = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export frame', exact: true }).click();
  expect((await sourceImage).suggestedFilename()).toBe('atdl-frame.jpg');
  await page.getByRole('button', { name: 'Sample clip' }).click();
  await expect(page.getByTestId('frame-time')).toHaveText('00:00.00');
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeEnabled();
});

test('training and results use project evidence and open a paired comparison', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Browser round trip')).toBeVisible();
  await page.getByRole('link', { name: 'Experiments', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Annotation conversion' })).toBeVisible();
  const original = await page.locator('pre').first().textContent();
  await page.getByRole('slider', { name: 'Horizontal box position' }).fill('0.12');
  await expect(page.locator('pre').first()).not.toHaveText(original ?? '');
  await page.getByRole('button', { name: 'Imbalance', exact: true }).click();
  await expect(page.getByText('3,223 image presentations per epoch')).toBeVisible();
  await page.getByLabel('Training recipe').selectOption('full_repeat');
  await expect(page.getByText('1,543 image presentations per epoch')).toBeVisible();
  await page.getByRole('button', { name: 'Rotate · 1.08× brightness' }).click();
  await expect(page.getByText('(1 − y, x, h, w)')).toBeVisible();
  await page.getByRole('button', { name: 'Training', exact: true }).click();
  await page.getByLabel('Training seed').selectOption('43');
  await expect(page.getByText('full_combined_seed43', { exact: true })).toBeVisible();
  await page.getByRole('slider', { name: 'Training epoch' }).fill('6');
  await expect(page.getByText('6 / 20', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Box loss', exact: true }).click();
  await expect(page.getByRole('img', { name: 'Box loss by training epoch. Epoch 6 selected.' })).toBeVisible();
  await page.getByRole('button', { name: 'Resolution', exact: true }).click();
  await page.getByRole('button', { name: 'Compare on a frame' }).click();
  await expect(page.locator('.split-label').last()).toContainText('320');
  await page.getByRole('link', { name: 'Results', exact: true }).click();
  await expect(page.getByText('0.7842')).toBeVisible();
  await page.getByRole('button', { name: 'YOLOv8n', exact: true }).click();
  await expect(page.getByText('YOLOv8n · validation tail AP50')).toBeVisible();
  await page.getByRole('button', { name: 'Patches', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Patches', exact: true })).toBeVisible();
  const inspect = page.getByRole('button', { name: 'Inspect case' });
  if (await inspect.count()) {
    await inspect.click();
    await expect(page.getByRole('heading', { name: 'Steel Inspection' })).toBeVisible();
    await expect(page.getByLabel('Annotation matching at IoU 0.50')).toBeVisible();
  } else {
    await page.getByRole('button', { name: 'Show validation montage' }).click();
    await expect(page.getByRole('img', { name: 'Saved validation error montage' })).toBeVisible();
  }
});

test('all-frame mode, custom still upload, and browser webcam capture', async ({ page, request }) => {
  test.setTimeout(150_000);
  await page.goto('/');
  await expect(page.getByText('Browser round trip')).toBeVisible();
  await page.getByRole('button', { name: 'Analyze every frame' }).click();
  await expect(page.getByText('120 frames analyzed')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole('button', { name: 'Export summary' })).toBeEnabled();
  const video = await request.get('/api/sample/clip');
  await page.getByLabel('Open steel media').setInputFiles({
    name: 'uploaded-coupon.webm', mimeType: 'video/webm', buffer: await video.body(),
  });
  await expect(page.getByText('uploaded-coupon.webm', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('Browser round trip')).toBeVisible();
  await page.getByRole('slider', { name: 'Seek clip' }).fill('1.50');
  await expect(page.getByTestId('frame-time')).toHaveText('00:01.50');
  const image = await request.get('/api/sample/image?position=20');
  await page.getByLabel('Open steel media').setInputFiles({
    name: 'uploaded-coupon.png', mimeType: 'image/png', buffer: await image.body(),
  });
  await expect(page.getByText('uploaded-coupon.png', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('Browser round trip')).toBeVisible();
  await page.getByRole('button', { name: 'Webcam', exact: true }).click();
  await expect(page.getByText('Local webcam', { exact: true }).first()).toBeVisible();
  await expect(page.getByText(/[1-9][0-9]* analyzed ·/)).toBeVisible();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible();
});

test('phone layout keeps source controls and working views in bounds', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Open media' })).toBeVisible();
  await expect(page.getByText('Browser round trip')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.getByRole('link', { name: 'Experiments', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Dataset split' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.getByRole('link', { name: 'Results', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Validation ranking' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
