/**
 * Global Test Setup for TinyCats E2E Tests
 *
 * This file provides:
 * - Global test fixtures for reusable setup
 * - Common test utilities
 * - Test image generation helpers
 * - API mocking utilities
 * - Error handling helpers
 *
 * Usage: Import fixtures in test files
 * Example: import { testFixtures, testImages } from './setup';
 */

import { test as base, expect, type Page, type BrowserContext } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

// ============================================================================
// Test Image Paths
// ============================================================================

export const TEST_ASSETS_DIR = path.join(__dirname, '..', 'assets');

export const testImages = {
  // Valid images
  cat: path.join(TEST_ASSETS_DIR, 'cat1.jpg'),
  catPng: path.join(TEST_ASSETS_DIR, 'cat2.png'),
  dog: path.join(TEST_ASSETS_DIR, 'dog1.jpg'),
  webp: path.join(TEST_ASSETS_DIR, 'test.webp'),

  // Size variants
  small: path.join(TEST_ASSETS_DIR, 'small.png'),      // 10x10
  large: path.join(TEST_ASSETS_DIR, 'large.png'),      // 4000x4000

  // Error cases
  corrupt: path.join(TEST_ASSETS_DIR, 'invalid.jpg'),  // Corrupt image
  notAnImage: path.join(TEST_ASSETS_DIR, 'invalid.txt'), // Text file
};

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * Extended test fixtures for TinyCats E2E tests
 */
export const testFixtures = base.extend<{
  // Pre-configured pages
  classifyPage: Page;
  generatePage: Page;
  benchmarkPage: Page;

  // Test utilities
  uploadImage: (page: Page, imagePath: string) => Promise<void>;
  waitForModelLoad: (page: Page, modelType?: 'classifier' | 'generator') => Promise<void>;
  takeSnapshot: (page: Page, name: string) => Promise<void>;

  // Mock utilities
  mockNetworkFailure: (page: Page, pattern?: RegExp) => Promise<void>;
  mockApiResponse: (page: Page, url: RegExp, response: object) => Promise<void>;

  // Test data
  testImagePath: string;
}>({
  // Pre-navigated classify page
  classifyPage: async ({ page }, use) => {
    await page.goto('/#/classify');
    await use(page);
  },

  // Pre-navigated generate page
  generatePage: async ({ page }, use) => {
    await page.goto('/#/generate');
    await use(page);
  },

  // Pre-navigated benchmark page
  benchmarkPage: async ({ page }, use) => {
    await page.goto('/#/benchmark');
    await use(page);
  },

  // Utility to upload an image
  uploadImage: async ({}, use) => {
    const uploadImage = async (page: Page, imagePath: string) => {
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles(imagePath);
    };
    await use(uploadImage);
  },

  // Utility to wait for model load
  waitForModelLoad: async ({}, use) => {
    const waitForModelLoad = async (
      page: Page,
      modelType: 'classifier' | 'generator' = 'classifier'
    ) => {
      const loadingText = modelType === 'classifier'
        ? 'Loading model'
        : 'Loading generator model';

      // Wait for loading message to appear
      await expect(page.locator('.MuiAlert-root')).toContainText(loadingText, { timeout: 5000 });

      // Wait for loading message to disappear (model loaded)
      await page.waitForSelector('.MuiAlert-root:not(:has-text("Loading"))', {
        timeout: 60000,
        state: 'visible',
      }).catch(() => {
        // If no alert, model might already be loaded
      });

      // Additional wait for model readiness
      await page.waitForTimeout(2000);
    };
    await use(waitForModelLoad);
  },

  // Utility to take a snapshot for debugging
  takeSnapshot: async ({}, use) => {
    const takeSnapshot = async (page: Page, name: string) => {
      const outputDir = path.join(__dirname, '..', '..', 'test-results', 'snapshots');

      // Ensure output directory exists
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      const snapshotPath = path.join(outputDir, `${name}-${Date.now()}.png`);
      await page.screenshot({ path: snapshotPath, fullPage: true });
      console.log(`Snapshot saved: ${snapshotPath}`);
    };
    await use(takeSnapshot);
  },

  // Utility to mock network failure
  mockNetworkFailure: async ({ page }, use) => {
    const mockNetworkFailure = async (page: Page, pattern?: RegExp) => {
      await page.route(pattern || /.*/, async (route) => {
        // Simulate network failure
        await route.abort('failed');
      });
    };
    await use(mockNetworkFailure);
  },

  // Utility to mock API response
  mockApiResponse: async ({ page }, use) => {
    const mockApiResponse = async (page: Page, url: RegExp, response: object) => {
      await page.route(url, async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(response),
        });
      });
    };
    await use(mockApiResponse);
  },

  // Default test image path
  testImagePath: [testImages.cat, { option: true }],
});

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generate a test image programmatically
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param color RGB color array [r, g, b]
 * @returns Buffer containing PNG data
 */
export function generateTestImage(
  width: number = 100,
  height: number = 100,
  color: [number, number, number] = [255, 0, 0]
): Buffer {
  // Create a minimal valid PNG
  // This is a simplified PNG generator for testing purposes

  const pngSignature = Buffer.from([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
  ]);

  // IHDR chunk
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData.writeUInt8(8, 8);  // Bit depth
  ihdrData.writeUInt8(2, 9);  // Color type (RGB)
  ihdrData.writeUInt8(0, 10); // Compression
  ihdrData.writeUInt8(0, 11); // Filter
  ihdrData.writeUInt8(0, 12); // Interlace

  const ihdrCrc = crc32(Buffer.concat([Buffer.from('IHDR'), ihdrData]));
  const ihdrChunk = Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, 0x0D]), // Length
    Buffer.from('IHDR'),
    ihdrData,
    ihdrCrc,
  ]);

  // IDAT chunk (compressed image data)
  // For simplicity, use a pre-computed minimal IDAT
  const idatData = Buffer.from([
    0x08, 0xD7, 0x63, 0xF8, 0x0F, 0x00, 0x00, 0x01,
    0x01, 0x00, 0x18, 0xDD, 0x8D, 0x01, 0x00, 0x00,
  ]);
  const idatCrc = crc32(Buffer.concat([Buffer.from('IDAT'), idatData]));
  const idatChunk = Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, idatData.length]),
    Buffer.from('IDAT'),
    idatData,
    idatCrc,
  ]);

  // IEND chunk
  const iendCrc = crc32(Buffer.from('IEND'));
  const iendChunk = Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, 0x00]),
    Buffer.from('IEND'),
    iendCrc,
  ]);

  return Buffer.concat([pngSignature, ihdrChunk, idatChunk, iendChunk]);
}

/**
 * CRC32 calculation for PNG chunks
 */
function crc32(data: Buffer): Buffer {
  let crc = 0xFFFFFFFF;
  const table = getCrc32Table();

  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
  }

  crc = crc ^ 0xFFFFFFFF;

  const result = Buffer.alloc(4);
  result.writeUInt32BE(crc >>> 0, 0);
  return result;
}

let crc32Table: number[] | null = null;

function getCrc32Table(): number[] {
  if (crc32Table) return crc32Table;

  crc32Table = [];
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crc32Table[i] = c;
  }

  return crc32Table;
}

/**
 * Create a corrupt image file
 * @returns Buffer containing invalid image data
 */
export function createCorruptImage(): Buffer {
  // Return random bytes that don't form a valid image
  return Buffer.from([
    0xFF, 0xD8, 0xFF, 0x00, // Start of JPEG but invalid
    0x00, 0x00, 0x00, 0x00,
    0xDE, 0xAD, 0xBE, 0xEF, // Invalid data
    0x00, 0x00, 0x00, 0x00,
  ]);
}

/**
 * Wait for element with retry
 * @param page Playwright page
 * @param selector CSS selector
 * @param options Wait options
 */
export async function waitForElementWithRetry(
  page: Page,
  selector: string,
  options: { timeout?: number; retries?: number } = {}
): Promise<void> {
  const { timeout = 10000, retries = 3 } = options;
  const retryDelay = timeout / retries;

  for (let i = 0; i < retries; i++) {
    try {
      await page.waitForSelector(selector, { timeout: retryDelay });
      return;
    } catch (e) {
      if (i === retries - 1) throw e;
      await page.waitForTimeout(500);
    }
  }
}

/**
 * Simulate network conditions
 * @param page Playwright page
 * @param options Network options
 */
export async function simulateNetworkCondition(
  page: Page,
  options: {
    offline?: boolean;
    slow?: boolean;
    failOnPattern?: RegExp;
  } = {}
): Promise<void> {
  const { offline = false, slow = false, failOnPattern } = options;

  if (offline) {
    await page.setOffline(true);
  }

  if (slow) {
    await page.route(/.*/, async (route) => {
      await page.waitForTimeout(1000); // 1 second delay
      await route.continue();
    });
  }

  if (failOnPattern) {
    await page.route(failOnPattern, async (route) => {
      await route.abort('failed');
    });
  }
}

// ============================================================================
// Export configured test
// ============================================================================

/**
 * Export the test object with fixtures
 * Use this instead of the default test import
 */
export const test = testFixtures;
export { expect };

// ============================================================================
// Global Setup/Teardown (for use with playwright.config.ts)
// ============================================================================

/**
 * Global setup function
 * Called once before all tests
 */
export async function globalSetup() {
  console.log('Global test setup starting...');

  // Ensure test assets directory exists
  if (!fs.existsSync(TEST_ASSETS_DIR)) {
    fs.mkdirSync(TEST_ASSETS_DIR, { recursive: true });
    console.log(`Created test assets directory: ${TEST_ASSETS_DIR}`);
  }

  // Generate any missing test images
  ensureTestAssets();

  console.log('Global test setup complete.');
  return () => {
    console.log('Global test teardown complete.');
  };
}

/**
 * Ensure all test assets exist
 */
function ensureTestAssets() {
  const requiredAssets = [
    'cat1.jpg',
    'cat2.png',
    'dog1.jpg',
    'small.png',
    'large.png',
    'invalid.jpg',
    'invalid.txt',
    'test.webp',
  ];

  for (const asset of requiredAssets) {
    const assetPath = path.join(TEST_ASSETS_DIR, asset);
    if (!fs.existsSync(assetPath)) {
      console.warn(`Missing test asset: ${assetPath}`);
    }
  }
}

/**
 * Global teardown function
 * Called once after all tests
 */
export async function globalTeardown() {
  console.log('Global test teardown starting...');
  // Clean up any temporary files if needed
  console.log('Global test teardown complete.');
}
