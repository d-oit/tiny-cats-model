# ADR-037: E2E Testing Strategy 2026

**Date:** 2026-02-27
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** GOAP.md Phase 16, ADR-019 (Evaluation Results), ADR-035 (Full Model Training)

## Context

### Current State

The tiny-cats-model project has basic E2E test infrastructure:
- **Playwright configured:** `playwright.config.ts` exists
- **Navigation tests:** 4 test specs exist (navigation only)
- **Test files:**
  - `tests/e2e/navigation.spec.ts` - ✅ Complete (457 lines)
  - `tests/e2e/classification.spec.ts` - ⚠️ Navigation only (646 lines)
  - `tests/e2e/generation.spec.ts` - ⚠️ Navigation only (1116 lines)
  - `tests/e2e/benchmark.spec.ts` - ⚠️ Navigation only (922 lines)
  - `tests/e2e/setup.ts` - Test fixtures (407 lines)
  - `tests/e2e/global-setup.ts` - Global setup (69 lines)

### Problem Statement

Current E2E tests only verify navigation works:
- ✅ Can navigate to pages
- ✅ Pages load without errors
- ❌ **Cannot verify inference works** (classification, generation)
- ❌ **Cannot verify model loading** (ONNX Runtime Web)
- ❌ **Cannot verify user interactions** (upload, generate, benchmark)

### Requirements

**2026 Best Practices for E2E Testing:**
1. **Full user journey testing** - Not just navigation
2. **Inference verification** - Verify model predictions
3. **Performance testing** - Verify latency metrics
4. **Edge case handling** - Error states, invalid inputs
5. **CI integration** - Run on every PR
6. **Flake management** - Retry logic, stable selectors

## Decision

We will implement **comprehensive E2E test coverage** for all user-facing functionality.

### Test Coverage Matrix

| Page | Current | Target | Priority |
|------|---------|--------|----------|
| **Home** | ✅ Navigation | N/A | - |
| **Classification** | ⚠️ Navigation only | ✅ Full inference (60+ tests) | HIGH |
| **Generation** | ⚠️ Navigation only | ✅ Full generation (80+ tests) | HIGH |
| **Benchmark** | ⚠️ Navigation only | ✅ Metrics verification (75+ tests) | MEDIUM |

### Test Architecture

#### 1. Test Structure

```typescript
tests/e2e/
├── classification.spec.ts      # Classification page tests
│   ├── Navigation (existing)
│   ├── Image Upload Tests
│   ├── Prediction Tests
│   ├── Error Handling Tests
│   └── Performance Tests
├── generation.spec.ts          # Generation page tests
│   ├── Navigation (existing)
│   ├── Breed Selection Tests
│   ├── Generation Tests
│   ├── CFG Slider Tests
│   └── Download Tests
├── benchmark.spec.ts           # Benchmark page tests
│   ├── Navigation (existing)
│   ├── Metrics Display Tests
│   ├── Performance Tests
│   └── Comparison Tests
├── navigation.spec.ts          # Site-wide navigation (existing)
├── setup.ts                    # Test fixtures and helpers
└── global-setup.ts             # Global test setup
```

#### 2. Classification Page Tests

**Image Upload Tests (15 tests):**
```typescript
test('uploads valid cat image and shows prediction', async ({ page }) => {
  // Upload test image
  // Verify breed prediction displays
  // Verify confidence percentage shows
});

test('rejects file larger than 10MB', async ({ page }) => {
  // Upload large file
  // Verify error message
});

test('handles non-image files gracefully', async ({ page }) => {
  // Upload .txt file
  // Verify error handling
});

test('supports multiple sequential uploads', async ({ page }) => {
  // Upload image 1, verify
  // Upload image 2, verify
  // Ensure no state leakage
});
```

**Prediction Tests (20 tests):**
```typescript
test('correctly predicts all 12 cat breeds', async ({ page }) => {
  // Test each breed: Abyssinian, Bengal, Birman, etc.
  // Verify prediction matches expected breed
});

test('identifies non-cat images as "other"', async ({ page }) => {
  // Upload dog image
  // Verify prediction is "other" or "not_cat"
});

test('displays confidence score in valid range', async ({ page }) => {
  // Upload image
  // Verify confidence is 0-100%
  // Verify reasonable confidence (>50% for clear images)
});

test('handles low confidence predictions', async ({ page }) => {
  // Upload ambiguous image
  // Verify UI shows low confidence indicator
});
```

**Error Handling Tests (10 tests):**
```typescript
test('shows error if model fails to load', async ({ page }) => {
  // Block model request
  // Verify error message displays
});

test('recovers from inference error', async ({ page }) => {
  // Trigger error
  // Verify retry works
});
```

**Performance Tests (15 tests):**
```typescript
test('completes inference in under 2 seconds', async ({ page }) => {
  // Upload image
  // Measure time to prediction
  // Verify < 2000ms
});

test('shows loading state during inference', async ({ page }) => {
  // Upload image
  // Verify loading spinner appears
  // Verify loading state clears after prediction
});
```

#### 3. Generation Page Tests

**Breed Selection Tests (15 tests):**
```typescript
test('allows selection of all 13 breeds', async ({ page }) => {
  // Select each breed: 12 cat breeds + other
  // Verify breed name displays
  // Verify selection persists
});

test('shows breed preview or description', async ({ page }) => {
  // Select breed
  // Verify breed info displays
});
```

**Generation Tests (30 tests):**
```typescript
test('generates image for selected breed', async ({ page }) => {
  // Select breed
  // Click generate
  // Verify image appears
  // Verify image is valid (not blank)
});

test('generates different images with same settings', async ({ page }) => {
  // Generate image 1
  // Generate image 2
  // Verify images are different (random noise)
});

test('handles all 13 breeds successfully', async ({ page }) => {
  // For each breed, generate image
  // Verify all generate without error
});

test('generates multiple images sequentially', async ({ page }) => {
  // Generate 5 images in sequence
  // Verify all complete successfully
});
```

**CFG Slider Tests (15 tests):**
```typescript
test('CFG slider affects output diversity', async ({ page }) => {
  // Generate with CFG=1.0
  // Generate with CFG=3.0
  // Verify outputs differ
});

test('CFG value displays correctly', async ({ page }) => {
  // Adjust slider
  // Verify value updates
});

test('handles extreme CFG values', async ({ page }) => {
  // Set CFG=0.1 (very low)
  // Set CFG=10.0 (very high)
  // Verify generation still works
});
```

**Download Tests (10 tests):**
```typescript
test('allows downloading generated image', async ({ page }) => {
  // Generate image
  // Click download
  // Verify download triggers
});

test('downloaded image has correct format', async ({ page }) => {
  // Generate image
  // Download
  // Verify PNG format
});
```

**Performance Tests (10 tests):**
```typescript
test('completes generation in under 5 seconds', async ({ page }) => {
  // Generate image
  // Measure time
  // Verify < 5000ms
});

test('shows progress during ODE integration', async ({ page }) => {
  // Click generate
  // Verify progress bar shows
  // Verify step counter updates
});
```

#### 4. Benchmark Page Tests

**Metrics Display Tests (25 tests):**
```typescript
test('displays latency metrics (p50, p95, p99)', async ({ page }) => {
  // Run benchmark
  // Verify p50 displays
  // Verify p95 displays
  // Verify p99 displays
});

test('displays FPS calculation', async ({ page }) => {
  // Run benchmark
  // Verify FPS shows
  // Verify FPS is reasonable (>10 FPS)
});

test('shows memory usage metrics', async ({ page }) => {
  // Run benchmark
  // Verify memory displays
});

test('displays model information', async ({ page }) => {
  // Verify model name shows
  // Verify model size shows
});
```

**Performance Tests (25 tests):**
```typescript
test('completes benchmark run successfully', async ({ page }) => {
  // Start benchmark
  // Wait for completion
  // Verify results display
});

test('handles multiple benchmark runs', async ({ page }) => {
  // Run benchmark 3 times
  // Verify results update each time
});

test('supports different batch sizes', async ({ page }) => {
  // Test batch size 1, 4, 8, 16
  // Verify all run successfully
});
```

**Comparison Tests (15 tests):**
```typescript
test('compares classifier vs generator performance', async ({ page }) => {
  // Run classifier benchmark
  // Run generator benchmark
  // Verify comparison displays
});

test('shows WebGPU vs WASM fallback', async ({ page }) => {
  // Verify backend detection
  // Show which backend is used
});
```

#### 5. Test Fixtures and Utilities

**Enhanced setup.ts:**
```typescript
import { test as base, expect } from '@playwright/test';

// Test fixtures
export const test = base.extend<{
  catImage: string;
  dogImage: string;
  largeFile: string;
}>({
  catImage: async ({}, use) => {
    // Provide test cat image path
    await use('tests/assets/cat.jpg');
  },
  dogImage: async ({}, use) => {
    // Provide test dog image path
    await use('tests/assets/dog.jpg');
  },
  largeFile: async ({}, use) => {
    // Generate large file for testing
    await use('tests/assets/large.png');
  },
});

export { expect };
```

**Helper functions:**
```typescript
// tests/e2e/helpers.ts

export async function uploadImage(page: Page, imagePath: string) {
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(imagePath);
}

export async function waitForPrediction(page: Page) {
  await expect(page.locator('.prediction-result')).toBeVisible({ timeout: 5000 });
}

export async function generateCatImage(page: Page, breed: string) {
  await page.selectOption('select[name="breed"]', breed);
  await page.click('button:has-text("Generate")');
  await expect(page.locator('.generated-image')).toBeVisible({ timeout: 10000 });
}
```

### CI Integration

#### GitHub Actions Workflow

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      
      - name: Install dependencies
        run: |
          npm ci
          cd frontend && npm ci
      
      - name: Install Playwright browsers
        run: npx playwright install --with-deps
      
      - name: Build frontend
        run: cd frontend && npm run build
      
      - name: Serve frontend
        run: |
          npx serve frontend/dist -l 5173 &
          sleep 5
      
      - name: Run E2E tests
        run: npx playwright test
      
      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/
```

#### Test Retry Strategy

```typescript
// playwright.config.ts
export default defineConfig({
  retries: process.env.CI ? 2 : 0, // Retry in CI only
  workers: process.env.CI ? 1 : undefined, // Single worker in CI
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
});
```

### Test Data Management

#### Test Assets

```
tests/assets/
├── cat.jpg                  # Valid cat image (Abyssinian)
├── cat1.jpg                 # Another cat image
├── cat2.png                 # PNG format cat
├── dog.jpg                  # Non-cat image (for "other" test)
├── dog1.jpg                 # Another non-cat
├── large.png                # >10MB file (for size validation)
├── tiny.png                 # Very small file
├── corrupt.jpg              # Corrupt image file
├── invalid.jpg              # Invalid image data
├── invalid.txt              # Non-image file
├── test.webp                # WebP format test
└── generate-test-images.js  # Script to generate test images
```

#### Image Generation Script

```javascript
// tests/assets/generate-test-images.js
const { createCanvas } = require('canvas');

// Generate test images of various sizes and formats
const sizes = [
  { name: 'tiny', width: 10, height: 10 },
  { name: 'small', width: 100, height: 100 },
  { name: 'normal', width: 512, height: 512 },
  { name: 'large', width: 2048, height: 2048 },
];

sizes.forEach(({ name, width, height }) => {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  // Draw colored rectangle
  ctx.fillStyle = '#FF6B6B';
  ctx.fillRect(0, 0, width, height);
  
  // Save
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(`${name}.png`, buffer);
});
```

## Implementation

### Phase 1: Test Infrastructure (Completed)
- [x] Playwright configuration
- [x] Basic navigation tests
- [x] Test fixtures and setup
- [x] Global setup for test server

### Phase 2: Classification Tests (Pending)
- [ ] Add image upload tests (15 tests)
- [ ] Add prediction tests (20 tests)
- [ ] Add error handling tests (10 tests)
- [ ] Add performance tests (15 tests)
- [ ] **Total: 60+ tests**

### Phase 3: Generation Tests (Pending)
- [ ] Add breed selection tests (15 tests)
- [ ] Add generation tests (30 tests)
- [ ] Add CFG slider tests (15 tests)
- [ ] Add download tests (10 tests)
- [ ] Add performance tests (10 tests)
- [ ] **Total: 80+ tests**

### Phase 4: Benchmark Tests (Pending)
- [ ] Add metrics display tests (25 tests)
- [ ] Add performance tests (25 tests)
- [ ] Add comparison tests (15 tests)
- [ ] **Total: 75+ tests**

### Phase 5: CI Integration (Pending)
- [ ] Add E2E workflow to GitHub Actions
- [ ] Configure test retry strategy
- [ ] Set up test report artifacts
- [ ] Add test coverage reporting

## Consequences

### Positive
- ✅ **Full coverage** - All user journeys tested
- ✅ **Confidence** - Catch regressions before deployment
- ✅ **Documentation** - Tests serve as usage examples
- ✅ **CI integration** - Automated testing on every PR
- ✅ **Performance monitoring** - Catch performance regressions

### Negative
- ⚠️ **CI time** - Adds 10-15 minutes to CI pipeline
- ⚠️ **Maintenance** - Tests need updates when UI changes
- ⚠️ **Flakiness** - E2E tests can be flaky (mitigated with retries)
- ⚠️ **Complexity** - More complex than unit tests

### Neutral
- ℹ️ **Test assets** - Need to maintain test image library
- ℹ️ **Browser dependencies** - Requires Playwright browsers
- ℹ️ **Server requirement** - Need running frontend for tests

## Alternatives Considered

### Alternative 1: Unit Tests Only
**Proposal:** Skip E2E tests, rely on unit tests.

**Rejected because:**
- Unit tests don't catch integration issues
- Can't verify ONNX Runtime Web behavior
- User experience not tested
- Industry standard includes E2E tests

### Alternative 2: Manual Testing Only
**Proposal:** Test manually before each release.

**Rejected because:**
- Time-consuming and error-prone
- Not scalable
- Can't automate in CI
- Easy to miss regressions

### Alternative 3: Cypress Instead of Playwright
**Proposal:** Use Cypress for E2E testing.

**Rejected because:**
- Playwright has better TypeScript support
- Playwright supports multiple browsers out of box
- Playwright has better performance
- Project already uses Playwright

### Alternative 4: Snapshot Testing
**Proposal:** Use visual regression testing with snapshots.

**Partially adopted:**
- Will use for UI consistency
- Not sufficient alone (need functional tests)
- Can be added as supplement

## Testing Best Practices

### 1. Test Isolation
```typescript
// Each test should be independent
test('test 1', async ({ page }) => {
  // Setup
  // Test
  // Cleanup (automatic)
});

test('test 2', async ({ page }) => {
  // Fresh state, no leakage from test 1
});
```

### 2. Stable Selectors
```typescript
// ✅ Good: Stable data-testid
await page.click('[data-testid="generate-button"]');

// ❌ Bad: Fragile XPath
await page.click('/html/body/div[2]/button[3]');

// ⚠️ Okay: Semantic selectors (can change)
await page.click('.generate-btn');
```

### 3. Proper Waiting
```typescript
// ✅ Good: Wait for specific state
await expect(page.locator('.result')).toBeVisible();

// ❌ Bad: Fixed delay
await page.waitForTimeout(5000);

// ⚠️ Okay: Wait for load state
await page.waitForLoadState('networkidle');
```

### 4. Descriptive Test Names
```typescript
// ✅ Good: Clear what and why
test('uploads cat image and shows breed prediction with confidence');

// ❌ Bad: Vague
test('test upload');
```

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Classification tests | 60+ tests | `npx playwright test --list` |
| Generation tests | 80+ tests | `npx playwright test --list` |
| Benchmark tests | 75+ tests | `npx playwright test --list` |
| CI integration | Workflow runs on PR | GitHub Actions tab |
| Test stability | <5% flake rate | CI run history |
| Coverage | All user journeys | Manual review |

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1 | ✅ Complete | Infrastructure setup |
| Phase 2 | 4-6 hours | Classification tests |
| Phase 3 | 6-8 hours | Generation tests |
| Phase 4 | 4-6 hours | Benchmark tests |
| Phase 5 | 2-3 hours | CI integration |
| **Total** | **16-23 hours** | **~2-3 days** |

## References

- Playwright Docs: https://playwright.dev/
- Playwright Best Practices: https://playwright.dev/docs/best-practices
- Testing Library: https://testing-library.com/
- ADR-019: Sample Evaluation Results
- ADR-035: Full Model Training Plan

## Appendix: Example Test File

```typescript
// tests/e2e/classification.spec.ts
import { test, expect } from './setup';

test.describe('Classification Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/classify');
  });

  test('uploads valid cat image and shows prediction', async ({ page }) => {
    // Upload test cat image
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles('tests/assets/cat.jpg');
    
    // Wait for prediction
    await expect(page.locator('[data-testid="prediction-result"]'))
      .toBeVisible({ timeout: 5000 });
    
    // Verify breed displays
    const breedElement = page.locator('[data-testid="breed-name"]');
    await expect(breedElement).toBeVisible();
    
    // Verify confidence displays
    const confidenceElement = page.locator('[data-testid="confidence"]');
    await expect(confidenceElement).toBeVisible();
    
    // Verify confidence is in valid range
    const confidenceText = await confidenceElement.textContent();
    const confidence = parseFloat(confidenceText?.replace('%', '') || '0');
    expect(confidence).toBeGreaterThanOrEqual(0);
    expect(confidence).toBeLessThanOrEqual(100);
  });

  test('rejects file larger than 10MB', async ({ page }) => {
    // Upload large file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles('tests/assets/large.png');
    
    // Verify error message
    await expect(page.locator('[data-testid="error-message"]'))
      .toBeVisible();
    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('File size exceeds');
  });

  test('correctly predicts all 12 cat breeds', async ({ page }) => {
    const breeds = [
      'Abyssinian', 'Bengal', 'Birman', 'Bombay',
      'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon',
      'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx'
    ];
    
    for (const breed of breeds) {
      // Upload breed-specific test image
      await page.goto(`/classify?breed=${breed}`);
      
      // Verify prediction
      await expect(page.locator('[data-testid="breed-name"]'))
        .toContainText(breed);
    }
  });
});
```
