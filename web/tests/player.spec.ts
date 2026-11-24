import { expect, test } from '@playwright/test';

test('inspection exposes playback layers and same-frame comparison from the player', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Browser round trip')).toBeVisible();
  const viewer = page.getByRole('region', { name: 'Frame viewer' });
  await expect(viewer.getByRole('group', { name: 'Frame layers' })).toBeVisible();
  await viewer.getByRole('button', { name: 'Original', exact: true }).click();
  await expect(viewer.locator('.box.detection')).toHaveCount(0);
  await viewer.getByRole('button', { name: 'Detections', exact: true }).click();
  await viewer.focus();
  await page.keyboard.press('Space');
  await expect(viewer.getByRole('button', { name: 'Pause', exact: true })).toBeVisible();
  await expect(viewer.locator('[data-analysis-mark]').nth(1)).toBeAttached();
  await page.keyboard.press('Space');
  await expect(viewer.getByRole('button', { name: 'Play', exact: true })).toBeVisible();
  const jump = viewer.getByRole('button', { name: 'Jump to detection' });
  const jumpTime = (await jump.getAttribute('title'))?.replace('Jump to ', '');
  await jump.click();
  await expect(viewer.getByTestId('frame-time')).toHaveText(jumpTime ?? '00:00.00');
  await expect(viewer.getByText('Frame ready', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Compare models', exact: true }).click();
  await expect(page.locator('.split-label')).toHaveCount(2);
  await page.getByRole('button', { name: 'Exit comparison', exact: true }).click();
  await expect(page.locator('.split-label')).toHaveCount(0);
});
