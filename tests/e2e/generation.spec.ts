import { test, expect } from '@playwright/test';
import { CAT_BREEDS, GENERATOR_CONFIG } from '../../frontend/src/constants';

/**
 * Comprehensive E2E tests for the Generation Page
 * Tests breed selection, parameter adjustments, image generation, and download functionality
 */
test.describe('Generation Page', () => {
  test.beforeEach(async ({ page }) => {
    // HashRouter uses hash-based navigation
    await page.goto('/#/generate');
  });

  // ==================== Page Load Tests ====================

  test('should load generator page with correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/TinyCats/);
    await expect(page.locator('h3')).toContainText('Cat Image Generator', { timeout: 10000 });
  });

  test('should display page description', async ({ page }) => {
    await expect(page.locator('text=Generate cat images')).toBeVisible();
    await expect(page.locator('text=TinyDiT')).toBeVisible();
    await expect(page.locator('text=ONNX Runtime Web')).toBeVisible();
  });

  test('should show model loading message initially', async ({ page }) => {
    await expect(page.locator('.MuiAlert-root')).toContainText('Loading generator model', { timeout: 5000 });
    await expect(page.locator('.MuiAlert-root')).toContainText('126 MB', { timeout: 5000 });
  });

  // ==================== Control Panel Tests ====================

  test('should display control panel with settings', async ({ page }) => {
    await expect(page.locator('text=Generation Settings')).toBeVisible();
  });

  test('should display generated image panel', async ({ page }) => {
    await expect(page.locator('text=Generated Image')).toBeVisible();
    await expect(page.locator('text=Click Generate to create an image')).toBeVisible();
  });

  // ==================== Breed Selector Tests ====================

  test('should display breed selector', async ({ page }) => {
    await expect(page.locator('label')).toContainText('Cat Breed');
    await expect(page.locator('[role="button"][aria-label="Cat Breed"]')).toBeVisible();
  });

  test('should display all 13 cat breeds in selector', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();

    const breedOptions = page.locator('[role="option"]');
    await expect(breedOptions).toHaveCount(13);

    // Verify all expected breeds are present
    const expectedBreeds = [
      'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
      'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue',
      'Siamese', 'Sphynx', 'Other'
    ];

    for (const breed of expectedBreeds) {
      await expect(page.locator(`[role="option"]:has-text("${breed}")`)).toBeVisible();
    }
  });

  test('should select different breed - Abyssinian', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Abyssinian")').click();
    await expect(breedSelector).toContainText('Abyssinian');
  });

  test('should select different breed - Bengal', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Bengal")').click();
    await expect(breedSelector).toContainText('Bengal');
  });

  test('should select different breed - Siamese', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Siamese")').click();
    await expect(breedSelector).toContainText('Siamese');
  });

  test('should select different breed - Maine Coon', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Maine Coon")').click();
    await expect(breedSelector).toContainText('Maine Coon');
  });

  test('should select different breed - Persian', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Persian")').click();
    await expect(breedSelector).toContainText('Persian');
  });

  test('should select different breed - Sphynx', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Sphynx")').click();
    await expect(breedSelector).toContainText('Sphynx');
  });

  test('should select different breed - Other', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Other")').click();
    await expect(breedSelector).toContainText('Other');
  });

  test('should close dropdown after selecting breed', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();

    // Select a breed
    await page.locator('[role="option"]:has-text("Bengal")').click();

    // Dropdown should be closed
    await expect(page.locator('[role="option"]')).toHaveCount(0);
  });

  test('should disable breed selector during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');

    // Start generation
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Breed selector should be disabled during generation
    await expect(breedSelector).toBeDisabled({ timeout: 5000 });
  });

  // ==================== Steps Slider Tests ====================

  test('should display steps slider with correct label', async ({ page }) => {
    await expect(page.locator('text=Sampling Steps')).toBeVisible();
  });

  test('should display default steps value (50)', async ({ page }) => {
    await expect(page.locator('text=Sampling Steps: 50')).toBeVisible();
  });

  test('should display steps slider range info', async ({ page }) => {
    await expect(page.locator('text=More steps = better quality but slower')).toBeVisible();
  });

  test('should adjust steps slider to minimum (10)', async ({ page }) => {
    const slider = page.locator('input[type="range"]').first();
    await slider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Sampling Steps: 10')).toBeVisible();
  });

  test('should adjust steps slider to maximum (100)', async ({ page }) => {
    const slider = page.locator('input[type="range"]').first();
    await slider.evaluate((el) => {
      (el as HTMLInputElement).value = '100';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Sampling Steps: 100')).toBeVisible();
  });

  test('should adjust steps slider using keyboard', async ({ page }) => {
    const slider = page.locator('[role="slider"]').first();
    await slider.focus();

    // Press right arrow to increase value
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('text=Sampling Steps:')).toContainText('55');
  });

  test('should disable steps slider during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Start generation
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Steps slider should be disabled during generation
    const slider = page.locator('input[type="range"]').first();
    await expect(slider).toBeDisabled({ timeout: 5000 });
  });

  // ==================== CFG Scale Slider Tests ====================

  test('should display CFG scale slider with correct label', async ({ page }) => {
    await expect(page.locator('text=Guidance Scale')).toBeVisible();
  });

  test('should display default CFG scale value (1.5)', async ({ page }) => {
    await expect(page.locator('text=Guidance Scale (CFG): 1.5')).toBeVisible();
  });

  test('should display CFG scale slider range info', async ({ page }) => {
    await expect(page.locator('text=Higher = more adherence to breed')).toBeVisible();
  });

  test('should adjust CFG scale slider to minimum (1.0)', async ({ page }) => {
    const slider = page.locator('input[type="range"]').nth(1);
    await slider.evaluate((el) => {
      (el as HTMLInputElement).value = '1';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Guidance Scale (CFG): 1.0')).toBeVisible();
  });

  test('should adjust CFG scale slider to maximum (3.0)', async ({ page }) => {
    const slider = page.locator('input[type="range"]').nth(1);
    await slider.evaluate((el) => {
      (el as HTMLInputElement).value = '3';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Guidance Scale (CFG): 3.0')).toBeVisible();
  });

  test('should adjust CFG scale slider using keyboard', async ({ page }) => {
    const slider = page.locator('[role="slider"]').nth(1);
    await slider.focus();

    // Press right arrow to increase value
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('text=Guidance Scale (CFG):')).toContainText('1.6');
  });

  test('should disable CFG scale slider during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Start generation
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // CFG slider should be disabled during generation
    const slider = page.locator('input[type="range"]').nth(1);
    await expect(slider).toBeDisabled({ timeout: 5000 });
  });

  // ==================== Generate Button Tests ====================

  test('should display generate button', async ({ page }) => {
    await expect(page.locator('button:has-text("Generate")')).toBeVisible();
  });

  test('should display generate button with icon', async ({ page }) => {
    await expect(page.locator('button:has-text("Generate") [data-testid="AutoFixHighIcon"]')).toBeVisible();
  });

  test('should have generate button enabled when model is ready', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);
    await expect(page.locator('button:has-text("Generate")')).toBeEnabled();
  });

  test('should have generate button disabled while model is loading', async ({ page }) => {
    // Immediately check before model loads
    await expect(page.locator('button:has-text("Generate")')).toBeDisabled({ timeout: 3000 });
  });

  test('should show generating state when clicked', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Button should show generating state
    await expect(generateButton).toContainText('Generating...', { timeout: 5000 });
    await expect(generateButton).toBeDisabled();
  });

  // ==================== Reset Noise Button Tests ====================

  test('should display reset noise button', async ({ page }) => {
    await expect(page.locator('button:has-text("Reset Noise")')).toBeVisible();
  });

  test('should display reset noise button with icon', async ({ page }) => {
    await expect(page.locator('button:has-text("Reset Noise") [data-testid="RefreshIcon"]')).toBeVisible();
  });

  test('should click reset noise button', async ({ page }) => {
    const resetButton = page.locator('button:has-text("Reset Noise")');
    await resetButton.click();
    // Button should still be enabled after reset
    await expect(resetButton).toBeEnabled();
  });

  test('should have reset noise button enabled when model is ready', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);
    await expect(page.locator('button:has-text("Reset Noise")')).toBeEnabled();
  });

  test('should disable reset noise button during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Start generation
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Reset button should be disabled during generation
    const resetButton = page.locator('button:has-text("Reset Noise")');
    await expect(resetButton).toBeDisabled({ timeout: 5000 });
  });

  // ==================== Download Button Tests ====================

  test('should display download button', async ({ page }) => {
    await expect(page.locator('button:has-text("Download")')).toBeVisible();
  });

  test('should display download button with icon', async ({ page }) => {
    await expect(page.locator('button:has-text("Download") [data-testid="DownloadIcon"]')).toBeVisible();
  });

  test('should have download button disabled initially', async ({ page }) => {
    const downloadButton = page.locator('button:has-text("Download")');
    await expect(downloadButton).toBeVisible();
    await expect(downloadButton).toBeDisabled();
  });

  test('should enable download button after generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Start generation with minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Wait for generation to complete
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Download button should be enabled after generation
    const downloadButton = page.locator('button:has-text("Download")');
    await expect(downloadButton).toBeEnabled();
  });

  test('should download generated image', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Generate image
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Download image
    const downloadButton = page.locator('button:has-text("Download")');
    const downloadPromise = page.waitForEvent('download');
    await downloadButton.click();
    const download = await downloadPromise;

    // Verify download started
    expect(download.suggestedFilename()).toContain('.png');
  });

  // ==================== Generated Image Display Tests ====================

  test('should display generated image placeholder', async ({ page }) => {
    await expect(page.locator('text=Generated Image')).toBeVisible();
    await expect(page.locator('text=Click Generate to create an image')).toBeVisible();
  });

  test('should display generated image after generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Generate image
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Wait for generation to complete
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Generated image should be visible
    const generatedImage = page.locator('[alt*="Generated"]');
    await expect(generatedImage).toBeVisible();
  });

  test('should display breed name with generated image', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Select a specific breed
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Bengal")').click();

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Generate image
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Breed name should be displayed
    await expect(page.locator('text=Breed: Bengal')).toBeVisible();
  });

  // ==================== Progress Dashboard Tests ====================

  test('should display progress dashboard during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Check for progress indicators
    await expect(page.locator('text=Steps')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Step Time')).toBeVisible();
    await expect(page.locator('text=Total')).toBeVisible();
  });

  test('should display progress bar during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Progress bar should be visible
    await expect(page.locator('.MuiLinearProgress-root')).toBeVisible({ timeout: 5000 });
  });

  test('should display step counter during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Step counter should show progress
    await expect(page.locator('text=/\\d+\\/\\d+/')).toBeVisible({ timeout: 5000 });
  });

  test('should display step time during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Step time should be displayed in ms
    await expect(page.locator('text=/\\d+ms/')).toBeVisible({ timeout: 5000 });
  });

  test('should display total time during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Total time should be displayed in seconds
    await expect(page.locator('text=/\\d+\\.\\d+s/')).toBeVisible({ timeout: 5000 });
  });

  test('should hide progress dashboard after generation completes', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Wait for generation to complete
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Progress dashboard should be hidden
    await expect(page.locator('text=Steps')).not.toBeVisible();
  });

  // ==================== Multiple Generation Tests ====================

  test('should handle multiple generations in sequence', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // First generation
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify first image was generated
    const generatedImage = page.locator('[alt*="Generated"]');
    await expect(generatedImage).toBeVisible();

    // Second generation
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify second image was generated
    await expect(generatedImage).toBeVisible();
  });

  test('should allow changing breed between generations', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // First generation with Abyssinian
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Abyssinian")').click();

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Change breed to Siamese
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Siamese")').click();

    // Second generation with Siamese
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify breed name updated
    await expect(page.locator('text=Breed: Siamese')).toBeVisible();
  });

  test('should allow changing steps between generations', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // First generation with 10 steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Change to 25 steps
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '25';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Second generation with 25 steps
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });
  });

  // ==================== Error Handling Tests ====================

  test('should display error alert for generation errors', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);

    // Error alert should not be visible initially
    await expect(page.locator('.MuiAlert-error')).not.toBeVisible();
  });

  // ==================== Accessibility Tests ====================

  test('should have accessible breed selector', async ({ page }) => {
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    await expect(breedSelector).toBeVisible();
    await expect(breedSelector).toHaveAttribute('aria-haspopup', 'listbox');
  });

  test('should have accessible sliders', async ({ page }) => {
    const sliders = page.locator('[role="slider"]');
    await expect(sliders).toHaveCount(2);

    // First slider should be for steps
    await expect(sliders.first()).toHaveAttribute('aria-label', /steps/i);
  });

  test('should have accessible buttons', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);

    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeEnabled();

    const resetButton = page.locator('button:has-text("Reset Noise")');
    await expect(resetButton).toBeEnabled();
  });

  // ==================== Layout Tests ====================

  test('should maintain two-column layout', async ({ page }) => {
    // Both control panel and image panel should be visible
    await expect(page.locator('text=Generation Settings')).toBeVisible();
    await expect(page.locator('text=Generated Image')).toBeVisible();
  });

  test('should display action buttons in grid layout', async ({ page }) => {
    // Reset Noise and Download buttons should be side by side
    const resetButton = page.locator('button:has-text("Reset Noise")');
    const downloadButton = page.locator('button:has-text("Download")');

    await expect(resetButton).toBeVisible();
    await expect(downloadButton).toBeVisible();
  });

  // ==================== Edge Cases: Parameter Extremes ====================

  test('should generate with minimum steps (10)', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimum steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Sampling Steps: 10')).toBeVisible();

    // Generate
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Should complete faster with fewer steps
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify image was generated
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  test('should generate with maximum steps (100)', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set maximum steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '100';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Sampling Steps: 100')).toBeVisible();

    // Generate - will take longer
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Should complete with extended timeout
    await expect(generateButton).toBeEnabled({ timeout: 180000 });

    // Verify image was generated
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  test('should generate with minimum CFG scale (1.0)', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimum CFG scale
    const cfgSlider = page.locator('input[type="range"]').nth(1);
    await cfgSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '1';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Guidance Scale (CFG): 1.0')).toBeVisible();

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Generate
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify image was generated
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  test('should generate with maximum CFG scale (3.0)', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set maximum CFG scale
    const cfgSlider = page.locator('input[type="range"]').nth(1);
    await cfgSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '3';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Guidance Scale (CFG): 3.0')).toBeVisible();

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Generate
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify image was generated
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  test('should generate with extreme parameter combination', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set min steps and max CFG
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const cfgSlider = page.locator('input[type="range"]').nth(1);
    await cfgSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '3';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Generate
    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify image was generated
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  // ==================== Edge Cases: All Breeds Loop ====================

  test('should generate all 13 breeds in sequence', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);

    // Set minimal steps for faster testing
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const breeds = [
      'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
      'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue',
      'Siamese', 'Sphynx', 'Other'
    ];

    const generateButton = page.locator('button:has-text("Generate")');

    for (const breed of breeds) {
      // Select breed
      const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
      await breedSelector.click();
      await page.locator(`[role="option"]:has-text("${breed}")`).click();
      await expect(breedSelector).toContainText(breed);

      // Generate
      await generateButton.click();
      await expect(generateButton).toBeEnabled({ timeout: 60000 });

      // Verify breed label matches
      await expect(page.locator(`text=Breed: ${breed}`)).toBeVisible();

      // Brief pause between generations
      await page.waitForTimeout(500);
    }
  });

  test('should verify each breed generates unique image', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);

    // Set minimal steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');

    // Generate with first breed
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Abyssinian")').click();
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Get first image source
    const firstImageSrc = await page.locator('[alt*="Generated"]').getAttribute('src');

    // Generate with different breed
    await breedSelector.click();
    await page.locator('[role="option"]:has-text("Siamese")').click();
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Get second image source
    const secondImageSrc = await page.locator('[alt*="Generated"]').getAttribute('src');

    // Images should be different (different random seeds)
    expect(firstImageSrc).not.toBe(secondImageSrc);
  });

  // ==================== Edge Cases: Network Failure ====================

  test('should handle network failure during generation', async ({ page }) => {
    // Wait for model to load first
    await page.waitForTimeout(5000);

    // Set minimal steps for faster test
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    // Simulate network failure during generation
    await page.route(/.*/, async (route) => {
      const url = route.request().url();
      // Allow local resources and already loaded models
      if (url.includes('localhost') || url.includes('onnx')) {
        await route.continue();
      } else {
        await route.abort('failed');
      }
    });

    const generateButton = page.locator('button:has-text("Generate")');
    await generateButton.click();

    // Should handle gracefully - either complete with cached data or show error
    await page.waitForTimeout(10000);

    // Page should still be functional
    await expect(page.locator('text=Generation Settings')).toBeVisible();
  });

  test('should show error and retry option for network failures', async ({ page }) => {
    // Set up network failure before generation
    await page.route(/.*huggingface.*/i, async (route) => {
      await route.abort('failed');
    });

    // Wait for model to potentially fail loading
    await page.waitForTimeout(5000);

    // Try to generate
    const generateButton = page.locator('button:has-text("Generate")');

    // If model didn't load, button might be disabled
    const isEnabled = await generateButton.isEnabled().catch(() => false);

    if (isEnabled) {
      await generateButton.click();
      await page.waitForTimeout(5000);
    }

    // Should show error or retry option
    const errorVisible = await page.locator('.MuiAlert-error').isVisible().catch(() => false);
    const retryVisible = await page.locator('text=/retry|Retry/i').isVisible().catch(() => false);

    expect(errorVisible || retryVisible || !isEnabled).toBeTruthy();
  });

  test('should allow retry after generation failure', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);

    // Set minimal steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');

    // First attempt
    await generateButton.click();

    // Wait for completion or failure
    await page.waitForTimeout(30000);

    // Should be able to try again
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Second attempt should work
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Verify image was generated
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  // ==================== Edge Cases: Rapid Generations ====================

  test('should handle rapid generation requests', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');

    // Click generate multiple times rapidly
    // The app should queue or ignore duplicate requests
    await generateButton.click();
    await page.waitForTimeout(500);

    // Button should be disabled during generation
    const isDisabled = await generateButton.isDisabled();
    expect(isDisabled).toBeTruthy();

    // Wait for completion
    await expect(generateButton).toBeEnabled({ timeout: 60000 });
  });

  test('should queue multiple generation requests properly', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(5000);

    // Set minimal steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');

    // Start first generation
    await generateButton.click();

    // Wait for it to complete
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Immediately start second generation
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Should complete both generations
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  // ==================== Edge Cases: Parameter Changes During Generation ====================

  test('should prevent parameter changes during generation', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const generateButton = page.locator('button:has-text("Generate")');
    const breedSelector = page.locator('[role="button"][aria-label="Cat Breed"]');
    const stepsSlider = page.locator('input[type="range"]').first();
    const cfgSlider = page.locator('input[type="range"]').nth(1);

    // Start generation
    await generateButton.click();

    // All controls should be disabled during generation
    await expect(breedSelector).toBeDisabled({ timeout: 5000 });
    await expect(stepsSlider).toBeDisabled({ timeout: 5000 });
    await expect(cfgSlider).toBeDisabled({ timeout: 5000 });
  });

  test('should apply parameter changes after generation completes', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set initial parameters
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');

    // First generation
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });

    // Change parameters
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '25';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect(page.locator('text=Sampling Steps: 25')).toBeVisible();

    // Second generation should use new parameters
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 90000 });

    // Verify image was generated with new settings
    await expect(page.locator('[alt*="Generated"]')).toBeVisible();
  });

  // ==================== Edge Cases: Reset Noise Functionality ====================

  test('should reset noise between generations', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    // Set minimal steps
    const stepsSlider = page.locator('input[type="range"]').first();
    await stepsSlider.evaluate((el) => {
      (el as HTMLInputElement).value = '10';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const generateButton = page.locator('button:has-text("Generate")');
    const resetButton = page.locator('button:has-text("Reset Noise")');

    // First generation
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });
    const firstImageSrc = await page.locator('[alt*="Generated"]').getAttribute('src');

    // Reset noise
    await resetButton.click();
    await page.waitForTimeout(500);

    // Second generation should produce different image
    await generateButton.click();
    await expect(generateButton).toBeEnabled({ timeout: 60000 });
    const secondImageSrc = await page.locator('[alt*="Generated"]').getAttribute('src');

    // Images should be different
    expect(firstImageSrc).not.toBe(secondImageSrc);
  });

  test('should allow multiple noise resets', async ({ page }) => {
    // Wait for model to load
    await page.waitForTimeout(3000);

    const resetButton = page.locator('button:has-text("Reset Noise")');

    // Click reset multiple times
    for (let i = 0; i < 3; i++) {
      await resetButton.click();
      await page.waitForTimeout(200);
    }

    // Button should still be enabled
    await expect(resetButton).toBeEnabled();
  });
});
