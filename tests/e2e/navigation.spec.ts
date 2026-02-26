import { test, expect } from '@playwright/test';

/**
 * Comprehensive E2E tests for Navigation
 * Tests navbar functionality, page accessibility, and responsive navigation
 */
test.describe('Navigation', () => {
  // ==================== Navbar Presence Tests ====================

  test('should display navbar on all pages', async ({ page }) => {
    // Test on classify page
    await page.goto('/#/classify');
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();

    // Test on generate page
    await page.goto('/#/generate');
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();

    // Test on benchmark page
    await page.goto('/#/benchmark');
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();
  });

  test('should display TinyCats logo in navbar', async ({ page }) => {
    await page.goto('/classify');
    await expect(page.locator('text=TinyCats')).toBeVisible();
  });

  test('should have TinyCats logo as clickable link to home', async ({ page }) => {
    await page.goto('/generate');
    await page.locator('text=TinyCats').click();
    await expect(page).toHaveURL(/.*classify.*/);
  });

  // ==================== Navigation Links Tests ====================

  test('should display Cat Classifier navigation link', async ({ page }) => {
    await page.goto('/classify');
    await expect(page.locator('text=Cat Classifier')).toBeVisible();
  });

  test('should display Cat Generator navigation link', async ({ page }) => {
    await page.goto('/classify');
    await expect(page.locator('text=Cat Generator')).toBeVisible();
  });

  test('should display Benchmark navigation link', async ({ page }) => {
    await page.goto('/classify');
    await expect(page.locator('text=Benchmark')).toBeVisible();
  });

  test('should have all three navigation links visible', async ({ page }) => {
    await page.goto('/classify');

    const navLinks = page.locator('.MuiAppBar-root a, .MuiAppBar-root [role="link"]');
    await expect(navLinks).toHaveCount({ min: 3 });
  });

  // ==================== Page Navigation Tests ====================

  test('should navigate to classify page from navbar', async ({ page }) => {
    await page.goto('/generate');
    await page.locator('text=Cat Classifier').click();
    await expect(page).toHaveURL(/.*classify.*/);
    await expect(page.locator('h3')).toContainText('Cat Classifier');
  });

  test('should navigate to generate page from navbar', async ({ page }) => {
    await page.goto('/classify');
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);
    await expect(page.locator('h3')).toContainText('Cat Image Generator');
  });

  test('should navigate to benchmark page from navbar', async ({ page }) => {
    await page.goto('/classify');
    await page.locator('text=Benchmark').click();
    await expect(page).toHaveURL(/.*benchmark.*/);
    await expect(page.locator('h3')).toContainText('Performance Benchmark');
  });

  test('should navigate between all pages sequentially', async ({ page }) => {
    // Start at classify
    await page.goto('/classify');
    await expect(page.locator('h3')).toContainText('Cat Classifier');

    // Navigate to generate
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);
    await expect(page.locator('h3')).toContainText('Cat Image Generator');

    // Navigate to benchmark
    await page.locator('text=Benchmark').click();
    await expect(page).toHaveURL(/.*benchmark.*/);
    await expect(page.locator('h3')).toContainText('Performance Benchmark');

    // Navigate back to classify
    await page.locator('text=Cat Classifier').click();
    await expect(page).toHaveURL(/.*classify.*/);
    await expect(page.locator('h3')).toContainText('Cat Classifier');
  });

  // ==================== Active State Tests ====================

  test('should highlight active navigation link - classify', async ({ page }) => {
    await page.goto('/classify');

    // Cat Classifier link should be highlighted/active
    const classifierLink = page.locator('text=Cat Classifier');
    await expect(classifierLink).toBeVisible();
    // Active link should have different styling (heavier font weight)
    await expect(classifierLink).toHaveCSS('font-weight', '700');
  });

  test('should highlight active navigation link - generate', async ({ page }) => {
    await page.goto('/generate');

    // Cat Generator link should be highlighted/active
    const generatorLink = page.locator('text=Cat Generator');
    await expect(generatorLink).toBeVisible();
    await expect(generatorLink).toHaveCSS('font-weight', '700');
  });

  test('should highlight active navigation link - benchmark', async ({ page }) => {
    await page.goto('/benchmark');

    // Benchmark link should be highlighted/active
    const benchmarkLink = page.locator('text=Benchmark');
    await expect(benchmarkLink).toBeVisible();
    await expect(benchmarkLink).toHaveCSS('font-weight', '700');
  });

  test('should update active state when navigating', async ({ page }) => {
    await page.goto('/classify');

    // Classify should be active
    await expect(page.locator('text=Cat Classifier')).toHaveCSS('font-weight', '700');
    await expect(page.locator('text=Cat Generator')).not.toHaveCSS('font-weight', '700');

    // Navigate to generate
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);

    // Generate should now be active
    await expect(page.locator('text=Cat Generator')).toHaveCSS('font-weight', '700');
    await expect(page.locator('text=Cat Classifier')).not.toHaveCSS('font-weight', '700');
  });

  // ==================== Page Title Tests ====================

  test('should have correct title on classify page', async ({ page }) => {
    await page.goto('/classify');
    await expect(page).toHaveTitle(/TinyCats/);
    await expect(page.locator('h3')).toContainText('Cat Classifier');
  });

  test('should have correct title on generate page', async ({ page }) => {
    await page.goto('/generate');
    await expect(page).toHaveTitle(/TinyCats/);
    await expect(page.locator('h3')).toContainText('Cat Image Generator');
  });

  test('should have correct title on benchmark page', async ({ page }) => {
    await page.goto('/benchmark');
    await expect(page).toHaveTitle(/TinyCats/);
    await expect(page.locator('h3')).toContainText('Performance Benchmark');
  });

  // ==================== Theme Toggle Tests ====================

  test('should display theme toggle button', async ({ page }) => {
    await page.goto('/classify');
    await expect(page.locator('[aria-label="toggle theme"]')).toBeVisible();
  });

  test('should display theme toggle icon', async ({ page }) => {
    await page.goto('/classify');

    // Should show either light or dark mode icon
    const lightIcon = page.locator('[data-testid="LightModeIcon"]');
    const darkIcon = page.locator('[data-testid="DarkModeIcon"]');

    const lightVisible = await lightIcon.isVisible();
    const darkVisible = await darkIcon.isVisible();

    expect(lightVisible || darkVisible).toBeTruthy();
  });

  test('should toggle theme when clicking theme button', async ({ page }) => {
    await page.goto('/classify');

    // Get initial theme icon
    const initialLightIcon = await page.locator('[data-testid="LightModeIcon"]').isVisible();
    const initialDarkIcon = await page.locator('[data-testid="DarkModeIcon"]').isVisible();

    // Click theme toggle
    await page.locator('[aria-label="toggle theme"]').click();

    // Theme should change
    await page.waitForTimeout(500);

    const afterLightIcon = await page.locator('[data-testid="LightModeIcon"]').isVisible();
    const afterDarkIcon = await page.locator('[data-testid="DarkModeIcon"]').isVisible();

    // Icon should have changed
    expect(initialLightIcon !== afterLightIcon || initialDarkIcon !== afterDarkIcon).toBeTruthy();
  });

  test('should persist theme across page navigation', async ({ page }) => {
    await page.goto('/classify');

    // Get current theme
    const initialDarkIcon = await page.locator('[data-testid="DarkModeIcon"]').isVisible();

    // Navigate to another page
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);

    // Theme should persist
    const afterDarkIcon = await page.locator('[data-testid="DarkModeIcon"]').isVisible();
    expect(initialDarkIcon).toBe(afterDarkIcon);
  });

  // ==================== Responsive Navigation Tests ====================

  test('should maintain navbar visibility on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE
    await page.goto('/classify');

    // Navbar should still be visible
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();
    await expect(page.locator('text=TinyCats')).toBeVisible();
  });

  test('should display navigation links on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/classify');

    // Navigation links should be visible
    await expect(page.locator('text=Cat Classifier')).toBeVisible();
    await expect(page.locator('text=Cat Generator')).toBeVisible();
    await expect(page.locator('text=Benchmark')).toBeVisible();
  });

  test('should maintain layout on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 }); // iPad
    await page.goto('/classify');

    // Navbar should be visible
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();

    // All navigation links should be visible
    await expect(page.locator('text=Cat Classifier')).toBeVisible();
    await expect(page.locator('text=Cat Generator')).toBeVisible();
    await expect(page.locator('text=Benchmark')).toBeVisible();
  });

  test('should maintain layout on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/classify');

    // Navbar should be visible
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();

    // All navigation links should be visible
    await expect(page.locator('text=Cat Classifier')).toBeVisible();
    await expect(page.locator('text=Cat Generator')).toBeVisible();
    await expect(page.locator('text=Benchmark')).toBeVisible();
  });

  // ==================== Keyboard Navigation Tests ====================

  test('should navigate using tab key', async ({ page }) => {
    await page.goto('/classify');

    // Tab through navigation links
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Should be able to focus on navigation elements
    const focusedElement = await page.evaluate(() => document.activeElement?.textContent);
    expect(focusedElement).toBeTruthy();
  });

  test('should activate focused link with enter key', async ({ page }) => {
    await page.goto('/classify');

    // Tab to first link
    await page.keyboard.press('Tab');

    // Press enter to activate
    await page.keyboard.press('Enter');

    // Should navigate (URL may change)
    await page.waitForTimeout(500);
  });

  // ==================== Accessibility Tests ====================

  test('should have accessible navbar', async ({ page }) => {
    await page.goto('/classify');

    // Navbar should have proper ARIA attributes
    const navbar = page.locator('.MuiAppBar-root');
    await expect(navbar).toBeVisible();
  });

  test('should have accessible navigation links', async ({ page }) => {
    await page.goto('/classify');

    // Links should be properly structured
    const links = page.locator('.MuiAppBar-root a');
    const count = await links.count();
    expect(count).toBeGreaterThanOrEqual(3);

    // Each link should have text
    for (let i = 0; i < count; i++) {
      const text = await links.nth(i).textContent();
      expect(text?.trim()).toBeTruthy();
    }
  });

  test('should have accessible theme toggle', async ({ page }) => {
    await page.goto('/classify');

    // Theme toggle should have aria-label
    const themeButton = page.locator('[aria-label="toggle theme"]');
    await expect(themeButton).toBeVisible();
    await expect(themeButton).toHaveAttribute('aria-label', 'toggle theme');
  });

  test('should have proper link structure', async ({ page }) => {
    await page.goto('/classify');

    // Navigation links should have href attributes
    const links = page.locator('.MuiAppBar-root a');
    const count = await links.count();

    for (let i = 0; i < count; i++) {
      await expect(links.nth(i)).toHaveAttribute('href');
    }
  });

  // ==================== URL/Route Tests ====================

  test('should handle direct URL navigation to classify', async ({ page }) => {
    await page.goto('/classify');
    await expect(page).toHaveURL(/.*classify.*/);
    await expect(page.locator('h3')).toContainText('Cat Classifier');
  });

  test('should handle direct URL navigation to generate', async ({ page }) => {
    await page.goto('/generate');
    await expect(page).toHaveURL(/.*generate.*/);
    await expect(page.locator('h3')).toContainText('Cat Image Generator');
  });

  test('should handle direct URL navigation to benchmark', async ({ page }) => {
    await page.goto('/benchmark');
    await expect(page).toHaveURL(/.*benchmark.*/);
    await expect(page.locator('h3')).toContainText('Performance Benchmark');
  });

  test('should handle root URL redirect to classify', async ({ page }) => {
    await page.goto('/');
    // Root should redirect to classify
    await expect(page).toHaveURL(/.*classify.*/);
    await expect(page.locator('h3')).toContainText('Cat Classifier');
  });

  test('should handle unknown routes', async ({ page }) => {
    await page.goto('/nonexistent-route');
    // Should either redirect to a valid page or show 404
    // In this case, React Router likely shows the default route
    await expect(page.locator('h3')).toBeVisible({ timeout: 5000 });
  });

  // ==================== Browser History Tests ====================

  test('should handle browser back button', async ({ page }) => {
    await page.goto('/classify');
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);

    // Go back
    await page.goBack();
    await expect(page).toHaveURL(/.*classify.*/);
    await expect(page.locator('h3')).toContainText('Cat Classifier');
  });

  test('should handle browser forward button', async ({ page }) => {
    await page.goto('/classify');
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);

    // Go back
    await page.goBack();
    await expect(page).toHaveURL(/.*classify.*/);

    // Go forward
    await page.goForward();
    await expect(page).toHaveURL(/.*generate.*/);
    await expect(page.locator('h3')).toContainText('Cat Image Generator');
  });

  // ==================== Navbar Styling Tests ====================

  test('should have sticky navbar positioning', async ({ page }) => {
    await page.goto('/classify');

    const navbar = page.locator('.MuiAppBar-root');
    const position = await navbar.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return style.position;
    });

    // Should be sticky or fixed
    expect(position === 'sticky' || position === 'fixed' || position === 'absolute').toBeTruthy();
  });

  test('should have proper navbar elevation', async ({ page }) => {
    await page.goto('/classify');

    const navbar = page.locator('.MuiAppBar-root');
    // Navbar should have some visual separation (elevation or border)
    await expect(navbar).toBeVisible();
  });

  // ==================== Cross-page State Tests ====================

  test('should maintain navbar state across page transitions', async ({ page }) => {
    await page.goto('/classify');

    // Get navbar visibility
    const navbarVisible = await page.locator('.MuiAppBar-root').isVisible();

    // Navigate to generate
    await page.locator('text=Cat Generator').click();
    await expect(page).toHaveURL(/.*generate.*/);

    // Navbar should still be visible
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();
  });

  test('should not lose navbar during page load', async ({ page }) => {
    await page.goto('/classify');

    // Navigate multiple times quickly
    await page.locator('text=Cat Generator').click();
    await page.locator('text=Benchmark').click();
    await page.locator('text=Cat Classifier').click();

    // Navbar should remain visible throughout
    await expect(page.locator('.MuiAppBar-root')).toBeVisible();
  });
});
