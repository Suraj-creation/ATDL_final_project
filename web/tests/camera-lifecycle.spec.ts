
import { expect, test } from '@playwright/test';

test('leaving Inspect cancels a pending webcam permission', async ({ page }) => {
  await page.addInitScript(() => {
    const source = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    const state = { release: () => {}, requested: false, completed: false, tracks: [] as MediaStreamTrack[] };
    Object.assign(window, { atdlCameraTest: state });
    navigator.mediaDevices.getUserMedia = async options => {
      state.requested = true;
      await new Promise<void>(resolve => { state.release = resolve; });
      try {
        const media = await source(options);
        state.tracks = media.getTracks();
        return media;
      } finally {
        state.completed = true;
      }
    };
  });
  await page.goto('/');
  await expect(page.getByText('Browser round trip')).toBeVisible();
  await page.getByRole('button', { name: 'Webcam', exact: true }).click();
  await page.getByRole('link', { name: 'Experiments', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Experiments', exact: true })).toBeVisible();
  await page.evaluate(() => (window as any).atdlCameraTest.release());
  await expect.poll(() => page.evaluate(() => {
    const state = (window as any).atdlCameraTest;
    return state.completed && state.tracks.length > 0 &&
      state.tracks.every((track: MediaStreamTrack) => track.readyState === 'ended');
  })).toBe(true);
  await page.getByRole('link', { name: 'Inspect', exact: true }).click();
  await expect(page.getByText('Steel coupon · synthetic', { exact: true }).first()).toBeVisible();
});
