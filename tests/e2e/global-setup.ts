/**
 * Global Setup for Playwright E2E Tests
 *
 * This file is used by Playwright's globalSetup configuration.
 * It runs once before all tests and can be used to:
 * - Set up test fixtures
 * - Generate test data
 * - Start services
 * - Configure environment
 */

import * as fs from 'fs';
import * as path from 'path';

const TEST_ASSETS_DIR = path.join(__dirname, '..', 'assets');

/**
 * Global setup function
 * Called once before all tests
 */
export default async function globalSetup() {
  console.log('Global test setup starting...');

  // Ensure test assets directory exists
  if (!fs.existsSync(TEST_ASSETS_DIR)) {
    fs.mkdirSync(TEST_ASSETS_DIR, { recursive: true });
    console.log(`Created test assets directory: ${TEST_ASSETS_DIR}`);
  }

  // Generate any missing test images
  ensureTestAssets();

  console.log('Global test setup complete.');

  // Return teardown function
  return async () => {
    console.log('Global test teardown starting...');
    // Clean up any temporary files if needed
    console.log('Global test teardown complete.');
  };
}

/**
 * Ensure all test assets exist
 */
function ensureTestAssets() {
  const requiredAssets = [
    'cat.jpg',
    'cat1.jpg',
    'cat2.png',
    'dog.jpg',
    'dog1.jpg',
    'tiny.png',
    'small.png',
    'large.png',
    'corrupt.jpg',
    'invalid.jpg',
    'notanimage.txt',
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
