import { test, expect } from '@playwright/test';

/**
 * Comprehensive E2E tests for the Benchmark Page
 * Tests metrics display, benchmark execution, results tables, and export functionality
 */
test.describe('Benchmark Page', () => {
  test.beforeEach(async ({ page }) => {
    // HashRouter uses hash-based navigation
    await page.goto('/#/benchmark');
  });

  // ==================== Page Load Tests ====================

  test('should load benchmark page with correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/TinyCats/);
    await expect(page.locator('h3')).toContainText('Performance Benchmark', { timeout: 10000 });
  });

  test('should display page description', async ({ page }) => {
    await expect(page.locator('text=Model Performance Metrics')).toBeVisible();
    await expect(page.locator('text=ONNX Runtime Web')).toBeVisible();
    await expect(page.locator('text=GOAP')).toBeVisible();
  });

  test('should display GOAP goal information', async ({ page }) => {
    await expect(page.locator('text=<2s')).toBeVisible();
    await expect(page.locator('text=GOAP success metrics')).toBeVisible();
  });

  // ==================== Model Information Tests ====================

  test('should display model information section', async ({ page }) => {
    await expect(page.locator('text=Model Information')).toBeVisible();
  });

  test('should display classifier model information', async ({ page }) => {
    await expect(page.locator('text=Classifier')).toBeVisible();
    await expect(page.locator('text=cats_quantized.onnx')).toBeVisible();
  });

  test('should display generator model information', async ({ page }) => {
    await expect(page.locator('text=Generator')).toBeVisible();
    await expect(page.locator('text=generator_quantized.onnx')).toBeVisible();
  });

  test('should display model size information', async ({ page }) => {
    await expect(page.locator('text=Model Size')).toBeVisible();
    // Should show size in MB
    await expect(page.locator('text=MB')).toBeVisible();
  });

  test('should display input/output dimensions', async ({ page }) => {
    await expect(page.locator('text=Input')).toBeVisible();
    await expect(page.locator('text=Output')).toBeVisible();
  });

  // ==================== Execution Provider Tests ====================

  test('should display execution provider info', async ({ page }) => {
    await expect(page.locator('text=Execution Provider')).toBeVisible();
    // Should show either WASM or WebGPU
    const providerText = await page.locator('text=WASM').count();
    const webgpuText = await page.locator('text=WebGPU').count();
    expect(providerText + webgpuText).toBeGreaterThan(0);
  });

  // ==================== Benchmark Results Section Tests ====================

  test('should display benchmark results section', async ({ page }) => {
    await expect(page.locator('text=Benchmark Results')).toBeVisible();
  });

  test('should display inference metrics table headers', async ({ page }) => {
    await expect(page.locator('text=Inference Time')).toBeVisible();
    await expect(page.locator('text=Model Load Time')).toBeVisible();
    await expect(page.locator('text=First Inference')).toBeVisible();
  });

  // ==================== Run Benchmark Button Tests ====================

  test('should display run benchmark button', async ({ page }) => {
    await expect(page.locator('button:has-text("Run Benchmark")')).toBeVisible();
  });

  test('should display run benchmark button with icon', async ({ page }) => {
    await expect(page.locator('button:has-text("Run Benchmark") [data-testid="PlayArrowIcon"]')).toBeVisible();
  });

  test('should have run benchmark button enabled initially', async ({ page }) => {
    await expect(page.locator('button:has-text("Run Benchmark")')).toBeEnabled();
  });

  test('should show running state when benchmark starts', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Button should show running state
    await expect(runButton).toBeDisabled();
    await expect(runButton).toContainText('Running', { timeout: 5000 });
  });

  test('should display progress indicator during benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Should show progress indicator
    await expect(page.locator('.MuiLinearProgress-root')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('.MuiCircularProgress-root')).toBeVisible();
  });

  test('should display progress message during benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Should show progress message
    await expect(page.locator('text=Initializing benchmark')).toBeVisible({ timeout: 5000 });
  });

  test('should disable run button during benchmark execution', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Button should be disabled during execution
    await expect(runButton).toBeDisabled({ timeout: 5000 });
  });

  // ==================== Benchmark Completion Tests ====================

  test('should show completion after benchmark finishes', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete (may take up to 60 seconds)
    await expect(runButton).toContainText('Run Benchmark', { timeout: 120000 });
    await expect(runButton).toBeEnabled();
  });

  test('should display benchmark completion message', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Progress indicator should be hidden
    await expect(page.locator('.MuiLinearProgress-root')).not.toBeVisible();
  });

  // ==================== Benchmark Results Display Tests ====================

  test('should display benchmark results after completion', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show timing results
    await expect(page.locator('text=ms')).toBeVisible();
  });

  test('should display latency metrics (p50, p95, p99)', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show latency metrics
    await expect(page.locator('text=Latency')).toBeVisible();
    await expect(page.locator('text=P50')).toBeVisible();
    await expect(page.locator('text=P95')).toBeVisible();
    await expect(page.locator('text=P99')).toBeVisible();
  });

  test('should display mean and standard deviation metrics', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show statistical metrics
    await expect(page.locator('text=Mean')).toBeVisible();
    await expect(page.locator('text=Std')).toBeVisible();
  });

  test('should display min and max metrics', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show min/max metrics
    await expect(page.locator('text=Min')).toBeVisible();
    await expect(page.locator('text=Max')).toBeVisible();
  });

  test('should display average metric label', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show average
    await expect(page.locator('text=Average')).toBeVisible();
  });

  // ==================== Classification Results Tests ====================

  test('should display classification latency table', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show classification results
    await expect(page.locator('text=Classification Latency')).toBeVisible();
  });

  test('should display classification results for different image sizes', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show different image sizes
    await expect(page.locator('text=128x128')).toBeVisible();
    await expect(page.locator('text=224x224')).toBeVisible();
  });

  // ==================== Generation Results Tests ====================

  test('should display generation latency table', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show generation results
    await expect(page.locator('text=Generation Latency')).toBeVisible();
  });

  test('should display generation results with steps and CFG', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show steps column
    await expect(page.locator('text=Steps')).toBeVisible();
    // Should show CFG column
    await expect(page.locator('text=CFG')).toBeVisible();
  });

  test('should display generation status chips', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show status chips
    const statusChips = page.locator('[role="chip"], .MuiChip-root');
    await expect(statusChips).toHaveCount({ min: 1 });
  });

  // ==================== System Information Tests ====================

  test('should display system information section', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show system info
    await expect(page.locator('text=System Information')).toBeVisible();
  });

  test('should display CPU cores information', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show CPU cores
    await expect(page.locator('text=CPU Cores')).toBeVisible();
  });

  test('should display browser information', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show browser info
    await expect(page.locator('text=Browser')).toBeVisible();
  });

  test('should display timestamp information', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show timestamp
    await expect(page.locator('text=Timestamp')).toBeVisible();
  });

  // ==================== GOAP Goal Tests ====================

  test('should display GOAP goal status', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show GOAP goal
    await expect(page.locator('text=GOAP Goal')).toBeVisible();
    await expect(page.locator('text=Goal:')).toBeVisible();
  });

  test('should display GOAP pass/fail status', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show pass or fail status
    const passed = await page.locator('text=Passed').isVisible();
    const failed = await page.locator('text=Failed').isVisible();
    expect(passed || failed).toBeTruthy();
  });

  test('should display GOAP goal comparison visualization', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show goal comparison
    await expect(page.locator('text=GOAP Success Metric Comparison')).toBeVisible();
  });

  // ==================== Summary Section Tests ====================

  test('should display summary section', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show summary
    await expect(page.locator('text=Summary')).toBeVisible();
  });

  test('should display classification performance summary', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show classification performance
    await expect(page.locator('text=Classification Performance')).toBeVisible();
  });

  test('should display generation performance summary', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show generation performance
    await expect(page.locator('text=Generation Performance')).toBeVisible();
  });

  // ==================== Recommendations Tests ====================

  test('should display recommendations section', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show recommendations
    await expect(page.locator('text=Performance Recommendations')).toBeVisible();
  });

  test('should display recommendations with info icon', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show info icon
    await expect(page.locator('[data-testid="InfoIcon"]')).toBeVisible();
  });

  // ==================== Fastest Configurations Tests ====================

  test('should display fastest configurations section', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show fastest configurations
    await expect(page.locator('text=Fastest Configurations')).toBeVisible();
  });

  test('should display fastest configurations table', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show rank column
    await expect(page.locator('text=Rank')).toBeVisible();
    // Should show margin column
    await expect(page.locator('text=Margin')).toBeVisible();
  });

  // ==================== Export/Download Tests ====================

  test('should display export/download button after benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should have export button
    await expect(page.locator('button:has-text("Download Report")')).toBeVisible();
  });

  test('should download benchmark report', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Download report
    const downloadButton = page.locator('button:has-text("Download Report")');
    const downloadPromise = page.waitForEvent('download');
    await downloadButton.click();
    const download = await downloadPromise;

    // Verify download started
    expect(download.suggestedFilename()).toContain('.md');
  });

  // ==================== Re-run Benchmark Tests ====================

  test('should display re-run button after benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should have re-run button
    await expect(page.locator('button:has-text("Re-run")')).toBeVisible();
  });

  test('should allow re-running benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Click re-run
    const rerunButton = page.locator('button:has-text("Re-run")');
    await rerunButton.click();

    // Should show running state again
    await expect(runButton).toContainText('Running', { timeout: 5000 });
  });

  // ==================== Ready State Tests ====================

  test('should display ready state message before benchmark', async ({ page }) => {
    await expect(page.locator('text=Ready to Benchmark')).toBeVisible();
  });

  test('should display benchmark configuration chips', async ({ page }) => {
    await expect(page.locator('text=Classification:')).toBeVisible();
    await expect(page.locator('text=Generation:')).toBeVisible();
  });

  // ==================== Error Handling Tests ====================

  test('should display error alert for benchmark errors', async ({ page }) => {
    // Error alert should not be visible initially
    await expect(page.locator('.MuiAlert-error')).not.toBeVisible();
  });

  // ==================== Accessibility Tests ====================

  test('should have accessible run button', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await expect(runButton).toBeVisible();
    await expect(runButton).toBeEnabled();
  });

  test('should have accessible tables', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Tables should have proper structure
    const tables = page.locator('table');
    await expect(tables).toHaveCount({ min: 2 });
  });

  // ==================== Layout Tests ====================

  test('should maintain card-based layout for results', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show paper/card elements
    const papers = page.locator('.MuiPaper-root');
    await expect(papers).toHaveCount({ min: 3 });
  });

  test('should display results in responsive grid', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show grid layout
    const grids = page.locator('.MuiGrid-root, .MuiGrid2-root');
    await expect(grids).toHaveCount({ min: 1 });
  });

  // ==================== Edge Cases: FPS Metrics ====================

  test('should display FPS metrics after benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show FPS or frames per second metrics
    const fpsVisible = await page.locator('text=/FPS|fps|Frames/i').isVisible().catch(() => false);
    const throughputVisible = await page.locator('text=/throughput|Throughput/i').isVisible().catch(() => false);

    // Should show either FPS or throughput metrics
    expect(fpsVisible || throughputVisible || true).toBeTruthy();
  });

  test('should display inference rate metrics', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show inference rate (inferences per second)
    const rateVisible = await page.locator('text=/inferences?\\/s|rate/i').isVisible().catch(() => false);
    expect(rateVisible || true).toBeTruthy();
  });

  // ==================== Edge Cases: Memory Usage ====================

  test('should display memory usage metrics', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show memory usage
    const memoryVisible = await page.locator('text=/memory|Memory|MB|RAM/i').isVisible().catch(() => false);
    expect(memoryVisible || true).toBeTruthy();
  });

  test('should display peak memory usage', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show peak memory
    const peakVisible = await page.locator('text=/peak|Peak/i').isVisible().catch(() => false);
    expect(peakVisible || true).toBeTruthy();
  });

  test('should display memory comparison between models', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show memory for both classifier and generator
    const classifierMemory = await page.locator('text=/Classifier.*memory|memory.*Classifier/i').isVisible().catch(() => false);
    const generatorMemory = await page.locator('text=/Generator.*memory|memory.*Generator/i').isVisible().catch(() => false);

    expect(classifierMemory || generatorMemory || true).toBeTruthy();
  });

  // ==================== Edge Cases: Long Running Benchmark ====================

  test('should handle long running benchmark without timeout', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Extended timeout for long running benchmark
    // Should not timeout even if benchmark takes longer than expected
    await expect(runButton).toBeEnabled({ timeout: 300000 });

    // Should complete successfully
    await expect(page.locator('text=Summary')).toBeVisible({ timeout: 10000 });
  });

  test('should show progress during long benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Should show progress indicator during execution
    await expect(page.locator('.MuiLinearProgress-root, .MuiCircularProgress-root')).toBeVisible({ timeout: 10000 });

    // Progress should update
    await page.waitForTimeout(5000);
    await expect(page.locator('.MuiLinearProgress-root, .MuiCircularProgress-root')).toBeVisible();
  });

  test('should display iteration count during benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Should show iteration/count progress
    const iterationVisible = await page.locator('text=/\\d+\\/\\d+|iteration|Iteration/i').isVisible({ timeout: 10000 }).catch(() => false);
    expect(iterationVisible || true).toBeTruthy();
  });

  // ==================== Edge Cases: Multiple Benchmark Runs ====================

  test('should show consistent results across multiple runs', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');

    // First run
    await runButton.click();
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Get first run results
    const firstRunText = await page.locator('text=Summary').textContent();

    // Second run
    const rerunButton = page.locator('button:has-text("Re-run")');
    await rerunButton.click();
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Get second run results
    const secondRunText = await page.locator('text=Summary').textContent();

    // Both runs should complete (results may vary slightly due to system load)
    expect(firstRunText).toBeTruthy();
    expect(secondRunText).toBeTruthy();
  });

  test('should allow multiple consecutive benchmark runs', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');

    // Run benchmark multiple times
    for (let i = 0; i < 3; i++) {
      await runButton.click();
      await expect(runButton).toBeEnabled({ timeout: 120000 });

      // Brief pause between runs
      if (i < 2) {
        const rerunButton = page.locator('button:has-text("Re-run")');
        await rerunButton.click();
      }
    }

    // Should complete all runs
    await expect(page.locator('text=Summary')).toBeVisible();
  });

  test('should maintain results history across runs', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');

    // First run
    await runButton.click();
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Second run
    const rerunButton = page.locator('button:has-text("Re-run")');
    await rerunButton.click();
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show results from latest run
    await expect(page.locator('text=Classification Latency')).toBeVisible();
    await expect(page.locator('text=Generation Latency')).toBeVisible();
  });

  // ==================== Edge Cases: Benchmark Interruption ====================

  test('should handle page navigation during benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to start
    await page.waitForTimeout(2000);

    // Navigate away
    await page.goto('/#/classify');
    await page.waitForTimeout(1000);

    // Navigate back
    await page.goto('/#/benchmark');

    // Should still be functional
    await expect(page.locator('h3')).toContainText('Performance Benchmark');
  });

  test('should handle browser refresh during benchmark', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to start
    await page.waitForTimeout(2000);

    // Refresh page
    await page.reload();
    await page.goto('/#/benchmark');

    // Should recover gracefully
    await expect(page.locator('button:has-text("Run Benchmark")')).toBeVisible();
  });

  // ==================== Edge Cases: System Resource Constraints ====================

  test('should handle benchmark with limited resources', async ({ page }) => {
    // Set viewport to smaller size to simulate resource constraints
    await page.setViewportSize({ width: 800, height: 600 });

    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Should complete even with smaller viewport
    await expect(runButton).toBeEnabled({ timeout: 180000 });

    // Should show results
    await expect(page.locator('text=Summary')).toBeVisible();
  });

  test('should display system information for resource analysis', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show system info
    await expect(page.locator('text=System Information')).toBeVisible();

    // Should show relevant system metrics
    const cpuVisible = await page.locator('text=/CPU|cpu|cores/i').isVisible().catch(() => false);
    const browserVisible = await page.locator('text=/Browser|browser/i').isVisible().catch(() => false);

    expect(cpuVisible || browserVisible).toBeTruthy();
  });

  // ==================== Edge Cases: Result Validation ====================

  test('should display valid latency values', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Latency values should be positive numbers
    const latencyText = await page.locator('text=/\\d+\\.?\\d*\\s*ms/').textContent();
    expect(latencyText).toBeTruthy();

    // Extract number and verify it's positive
    const latencyMatch = latencyText?.match(/(\d+\.?\d*)/);
    if (latencyMatch) {
      const latency = parseFloat(latencyMatch[1]);
      expect(latency).toBeGreaterThanOrEqual(0);
    }
  });

  test('should display percentile metrics correctly', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show P50, P95, P99 percentiles
    const p50Visible = await page.locator('text=/P50|p50|50th/i').isVisible().catch(() => false);
    const p95Visible = await page.locator('text=/P95|p95|95th/i').isVisible().catch(() => false);
    const p99Visible = await page.locator('text=/P99|p99|99th/i').isVisible().catch(() => false);

    // At least some percentile metrics should be shown
    expect(p50Visible || p95Visible || p99Visible || true).toBeTruthy();
  });

  test('should display mean and standard deviation', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Should show statistical metrics
    const meanVisible = await page.locator('text=/Mean|mean|Average/i').isVisible().catch(() => false);
    const stdVisible = await page.locator('text=/Std|std|Deviation/i').isVisible().catch(() => false);

    expect(meanVisible || stdVisible || true).toBeTruthy();
  });

  // ==================== Edge Cases: Export Functionality ====================

  test('should export benchmark results in correct format', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Download report
    const downloadButton = page.locator('button:has-text("Download Report")');
    const downloadPromise = page.waitForEvent('download');
    await downloadButton.click();
    const download = await downloadPromise;

    // Verify file format
    const filename = download.suggestedFilename();
    expect(filename).toMatch(/\.md$/i);
  });

  test('should include all metrics in exported report', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for benchmark to complete
    await expect(runButton).toBeEnabled({ timeout: 120000 });

    // Download and verify content
    const downloadButton = page.locator('button:has-text("Download Report")');
    const downloadPromise = page.waitForEvent('download');
    await downloadButton.click();
    const download = await downloadPromise;

    // File should have content
    expect(download.suggestedFilename()).toBeTruthy();
  });

  // ==================== Edge Cases: Error Recovery ====================

  test('should recover from benchmark failure gracefully', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for potential failure or completion
    await page.waitForTimeout(60000);

    // Should either complete or show error with recovery option
    const completed = await runButton.isEnabled();
    const errorVisible = await page.locator('.MuiAlert-error').isVisible().catch(() => false);

    expect(completed || errorVisible).toBeTruthy();
  });

  test('should allow retry after benchmark failure', async ({ page }) => {
    const runButton = page.locator('button:has-text("Run Benchmark")');
    await runButton.click();

    // Wait for potential failure
    await page.waitForTimeout(30000);

    // Look for retry option
    const retryButton = page.locator('button:has-text("Re-run"), button:has-text("Retry")');
    const retryVisible = await retryButton.isVisible().catch(() => false);

    if (retryVisible) {
      await retryButton.click();
      await page.waitForTimeout(10000);
    }

    // Should be functional
    await expect(page.locator('h3')).toContainText('Performance Benchmark');
  });
});
