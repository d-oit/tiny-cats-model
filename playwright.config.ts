import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for TinyCats E2E Tests
 *
 * This configuration is optimized for:
 * - GitHub Actions CI environment
 * - Headless browser testing
 * - Screenshot and video capture on failures
 * - Proper timeout handling for model loading
 *
 * Note: The app uses HashRouter, so navigation uses hash-based URLs
 */
export default defineConfig({
  // Directory containing test files
  testDir: './tests/e2e',

  // Timeout for each test (30 seconds as per requirements)
  timeout: 30000,

  // Timeout for expectations
  expect: {
    timeout: 10000,
  },

  // Number of retries in CI
  retries: 2,

  // Run tests in parallel (disabled in CI for stability)
  fullyParallel: !process.env.CI,

  // Fail the build on CI if test files are marked as .only
  forbidOnly: !!process.env.CI,

  // Number of concurrent workers (1 for CI stability)
  workers: process.env.CI ? 1 : undefined,

  // Test reporter
  reporter: process.env.CI ? 'github' : 'list',

  // Shared configuration for all tests
  use: {
    // Base URL for the application
    // Vite base is /tiny-cats-model/, app uses HashRouter
    baseURL: 'http://localhost:5173/tiny-cats-model',

    // Collect trace when retrying failed tests
    trace: 'on-first-retry',

    // Capture screenshot on failure
    screenshot: 'only-on-failure',

    // Capture video on failure
    video: 'retain-on-failure',

    // Browser context options
    viewport: { width: 1280, height: 720 },

    // Actionability options
    actionTimeout: 10000,

    // Navigation timeout
    navigationTimeout: 30000,
  },

  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // Emulate reduced motion for consistent animations
        colorScheme: 'dark',
      },
    },
    // Uncomment to test on other browsers
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },
    // Mobile testing
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },
  ],

  // Directory for test artifacts (screenshots, videos, traces)
  outputDir: './test-results/',

  // Global setup file - runs once before all tests
  globalSetup: require.resolve('./tests/e2e/global-setup'),

  // Web server configuration
  webServer: {
    // Command to start the development server
    command: 'cd frontend && npm run dev',

    // Port the server runs on
    port: 5173,

    // Reuse existing server when not in CI
    reuseExistingServer: !process.env.CI,

    // Timeout for server startup
    timeout: 120000,

    // Environment variables for the server
    env: {
      NODE_ENV: 'test',
    },
  },

  // Global setup file (optional)
  // globalSetup: require.resolve('./tests/global-setup'),

  // Global teardown file (optional)
  // globalTeardown: require.resolve('./tests/global-teardown'),
});
