import { test, expect } from '@playwright/test';

test.describe('Frontend Navigation', () => {
  test('should load classifier page at root', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await expect(page.locator('h3')).toContainText('Cat Classifier', { timeout: 10000 });
  });

  test('should navigate to generator page', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.click('a:has-text("Cat Generator")');
    await expect(page.locator('h3')).toContainText('Cat Image Generator', { timeout: 10000 });
  });

  test('should navigate to benchmark page', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.click('a:has-text("Benchmark")');
    await expect(page.locator('h3')).toContainText('Performance Benchmark', { timeout: 10000 });
  });
});
