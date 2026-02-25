# Skill: agent-browser

Browser automation using Playwright for E2E testing.

## When Use

- Verify frontend functionality works in real browser
- Test model loading and inference in browser
- Verify generation pipeline end-to-end
- Debug browser-specific issues

## Setup

```bash
# Install Playwright
npm install -D @playwright/test
npx playwright install chromium

# Install browser dependencies (if needed)
npx playwright install-deps
```

## Running Tests

```bash
# Run all E2E tests
npx playwright test

# Run specific test file
npx playwright test tests/e2e/generation.spec.ts

# Run with UI
npx playwright test --ui

# Run in headed mode
npx playwright test --headed

# Run specific test
npx playwright test tests/e2e/generation.spec.ts -g "should load"
```

## Configuration

Playwright is configured in `playwright.config.ts`:

```typescript
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120000,
  retries: 2,
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  webServer: {
    command: 'npm run dev',
    port: 5173,
    reuseExistingServer: !process.env.CI,
  },
});
```

## Test Structure

Tests should be in `tests/e2e/`:

```typescript
import { test, expect } from '@playwright/test';

test.describe('Cat Generator', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/generate');
  });

  test('should load the generation page', async ({ page }) => {
    await expect(page.locator('h3')).toContainText('Cat Image Generator');
  });

  test('should generate an image', async ({ page }) => {
    await expect(page.locator('button:has-text("Generate")')).toBeEnabled({ timeout: 60000 });
    await page.click('button:has-text("Generate")');
    await expect(page.locator('img[alt*="Generated"]')).toBeVisible({ timeout: 120000 });
  });
});
```

## Verification Checklist

- [ ] Model loads without errors
- [ ] All 13 breeds selectable
- [ ] Generate button works
- [ ] Progress updates during generation
- [ ] Generated image displays correctly
- [ ] Download button works
- [ ] No console errors

## CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: E2E Tests
  run: |
    cd frontend
    npm install
    npx playwright install --with-deps
    npx playwright test
```

## Debugging

```bash
# Show browser console logs
npx playwright test --reporter=list

# Generate trace on failure
npx playwright test --trace on

# Open trace viewer
npx playwright show-trace trace.zip
```

## Common Issues

### Model fails to load
- Check model file exists in `public/models/`
- Verify CORS headers
- Check browser console for errors

### Generation timeout
- Increase timeout in test
- Check Web Worker initialization
