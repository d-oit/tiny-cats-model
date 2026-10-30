import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { generateTestImage, testImages } from "./setup";

const VALID_PNG = generateTestImage(1, 1);
const INVALID_FILE = Buffer.from("this is not an image");

async function openClassifier(page: Page) {
  page.on("console", (message) => {
    if (message.type() === "error") console.error("[browser console]", message.text());
  });
  page.on("pageerror", (error) => console.error("[browser pageerror]", error));
  await page.goto("/tiny-cats-model/#/classify");
  await expect(page.getByText("Verified model", { exact: false })).toBeVisible({
    timeout: 90_000,
  });
}

test.describe.configure({ timeout: 120_000 });

test.describe("browser runtime contract", () => {
  test("@runtime browser tensor matches the shared Python preprocessing values", async ({ page }) => {
    test.skip(
      process.env.PLAYWRIGHT_PREVIEW === "true",
      "the tensor capture hook is development-only"
    );
    await page.addInitScript(() => {
      window.__tinyCatsPreprocessedTensor = null;
      window.__tinyCatsDecodedDimensions = null;
      window.__TINY_CATS_CAPTURE_PREPROCESSING__ = (tensor, dimensions) => {
        window.__tinyCatsPreprocessedTensor = Array.from(tensor);
        window.__tinyCatsDecodedDimensions = dimensions;
      };
    });
    await openClassifier(page);
    await page.getByTestId("classify-file-input").setInputFiles({
      name: "solid.png",
      mimeType: "image/png",
      buffer: generateTestImage(224, 224, [127, 89, 213]),
    });
    await page.waitForFunction(
      () =>
        window.__tinyCatsPreprocessedTensor?.length === 3 * 224 * 224 &&
        window.__tinyCatsDecodedDimensions?.width === 224 &&
        window.__tinyCatsDecodedDimensions?.height === 224
    );
    expect(await page.evaluate(() => window.__tinyCatsDecodedDimensions)).toEqual({
      width: 224,
      height: 224,
    });
    const tensor = await page.evaluate(() => window.__tinyCatsPreprocessedTensor);
    const area = 224 * 224;
    const expected = [
      (127 / 255 - 0.485) / 0.229,
      (89 / 255 - 0.456) / 0.224,
      (213 / 255 - 0.406) / 0.225,
    ];
    for (let channel = 0; channel < 3; channel++) {
      expect(tensor?.[channel * area]).toBeCloseTo(expected[channel], 6);
      expect(tensor?.[channel * area + area - 1]).toBeCloseTo(expected[channel], 6);
    }

    const transparentPng = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 224;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas context unavailable");
      context.fillStyle = "rgba(255, 0, 0, 0.5)";
      context.fillRect(0, 0, 224, 224);
      return canvas.toDataURL("image/png");
    });
    await page.evaluate(() => {
      window.__tinyCatsPreprocessedTensor = null;
    });
    await page.getByTestId("classify-file-input").setInputFiles({
      name: "half-alpha.png",
      mimeType: "image/png",
      buffer: Buffer.from(transparentPng.split(",")[1], "base64"),
    });
    await page.waitForFunction(
      () => window.__tinyCatsPreprocessedTensor?.length === 3 * 224 * 224
    );
    const alphaTensor = await page.evaluate(
      () => window.__tinyCatsPreprocessedTensor
    );
    expect(alphaTensor?.[0]).toBeCloseTo((128 / 255 - 0.485) / 0.229, 5);
    expect(alphaTensor?.[area]).toBeCloseTo(-0.456 / 0.224, 5);
    expect(alphaTensor?.[2 * area]).toBeCloseTo(-0.406 / 0.225, 5);
    await page.evaluate(() => {
      window.__tinyCatsDecodedDimensions = null;
    });
    await page.getByTestId("classify-file-input").setInputFiles(testImages.exifOrientation);
    await page.waitForFunction(
      () =>
        window.__tinyCatsDecodedDimensions?.width === 3 &&
        window.__tinyCatsDecodedDimensions?.height === 2
    );
  });

  test("@runtime classifier uses verified WASM inference and canonical labels", async ({ page }) => {
    await openClassifier(page);
    await expect(page.getByText(/Verified model · local · WASM/)).toBeVisible();

    await page.getByTestId("classify-file-input").setInputFiles({
      name: "pixel.png",
      mimeType: "image/png",
      buffer: VALID_PNG,
    });

    const result = page.getByTestId("classification-result");
    await expect(result).toBeVisible({ timeout: 30_000 });
    await expect(result.getByText("Cat", { exact: true })).toBeVisible();
    await expect(result.getByText("Not a Cat", { exact: true })).toBeVisible();
    await expect(result.getByText(/\d+\.\d%/).first()).toBeVisible();
  });

  test("@runtime upload can be activated with the keyboard", async ({ page }) => {
    await openClassifier(page);
    const chooserPromise = page.waitForEvent("filechooser");
    await page.getByTestId("classify-dropzone").press("Enter");
    const chooser = await chooserPromise;
    await chooser.setFiles({
      name: "keyboard.png",
      mimeType: "image/png",
      buffer: VALID_PNG,
    });
    await expect(page.getByTestId("classification-result")).toBeVisible({
      timeout: 30_000,
    });
  });

  test("@runtime rejects corrupt image bytes and recovers from model integrity failure on retry", async ({ page }) => {
    let modelRequestCount = 0;
    await page.context().route("**/models/cats_quantized.onnx", async (route) => {
      modelRequestCount += 1;
      if (modelRequestCount === 1) {
        await route.fulfill({ status: 200, body: "corrupt model bytes" });
        return;
      }
      await route.continue();
    });

    await page.goto("/tiny-cats-model/#/classify");
    await expect(page.getByRole("alert").filter({ hasText: "mismatch" })).toBeVisible({
      timeout: 60_000,
    });
    await page.getByRole("button", { name: "Retry" }).click();
    await expect(page.getByText(/Verified model · local · WASM/)).toBeVisible({
      timeout: 90_000,
    });

    await page.getByTestId("classify-file-input").setInputFiles({
      name: "fake.png",
      mimeType: "image/png",
      buffer: INVALID_FILE,
    });
    await expect(page.getByRole("alert")).toContainText("not a supported image");
    await expect(page.getByTestId("classification-result")).toHaveCount(0);
  });

  test("@runtime accepts JPEG, PNG and WebP but rejects unsupported and oversized images", async ({ page }) => {
    await openClassifier(page);
    const supported = [
      { file: testImages.cat, name: "fixture.jpg", mimeType: "image/jpeg" },
      { file: testImages.catPng, name: "fixture.png", mimeType: "image/png" },
      { file: testImages.webp, name: "fixture.webp", mimeType: "image/webp" },
    ];
    const input = page.getByTestId("classify-file-input");
    const result = page.getByTestId("classification-result");
    for (const [index, image] of supported.entries()) {
      if (index > 0) {
        await page.getByRole("button", { name: "Try Another Image" }).click();
      }
      await input.setInputFiles(image.file);
      await expect(result).toBeVisible({ timeout: 30_000 });
      await expect(input).toBeEnabled({ timeout: 30_000 });
      await expect(page.getByRole("alert")).toHaveCount(0);
    }

    await page.getByRole("button", { name: "Try Another Image" }).click();
    await input.setInputFiles({
      name: "unsupported.png",
      mimeType: "application/pdf",
      buffer: generateTestImage(1, 1),
    });
    await expect(page.getByRole("alert")).toContainText("Unsupported file type");

    await input.setInputFiles({
      name: "too-wide.png",
      mimeType: "image/png",
      buffer: generateTestImage(8193, 1),
    });
    await expect(page.getByRole("alert")).toContainText("per-axis limit");
  });

  test("@runtime falls back to WASM after a WebGPU session creation failure", async ({ page }) => {
    await page.addInitScript(() => {
      window.__TINY_CATS_TEST_ASSETS__ = {
        cats: { forceWebGpuFailure: true },
      };
    });
    await openClassifier(page);
    await expect(page.getByText(/Verified model · local · WASM/)).toBeVisible();
  });

  test("@runtime rejects a stale bundled fallback digest", async ({ page }) => {
    await page.addInitScript(() => {
      window.__TINY_CATS_TEST_ASSETS__ = {
        cats: { sha256: "0".repeat(64) },
      };
    });
    await page.goto("/tiny-cats-model/#/classify");
    await expect(page.getByRole("alert").filter({ hasText: "digest mismatch" })).toBeVisible({
      timeout: 60_000,
    });
    await expect(page.getByTestId("classify-file-input")).toBeDisabled();
  });

  test("@runtime uses WebGPU only when the browser exposes an adapter", async ({ page }) => {
    const available = await page.evaluate(async () => {
      const gpu = (
        navigator as Navigator & {
          gpu?: { requestAdapter: () => Promise<unknown | null> };
        }
      ).gpu;
      return gpu ? (await gpu.requestAdapter()) !== null : false;
    });
    test.skip(!available, "WebGPU adapter is unavailable on this runner");
    await openClassifier(page);
    await expect(page.getByText(/Verified model · local · (WEBGPU|WASM)/)).toBeVisible();
  });

  test("@runtime terminates the inference worker on route change", async ({ page }) => {
    await page.addInitScript(() => {
      const NativeWorker = window.Worker;
      window.__tinyCatsTerminatedWorkers = 0;
      window.Worker = class extends NativeWorker {
        terminate() {
          window.__tinyCatsTerminatedWorkers += 1;
          super.terminate();
        }
      };
    });
    await openClassifier(page);
    await page.goto("/tiny-cats-model/#/generate");
    await expect
      .poll(() => page.evaluate(() => window.__tinyCatsTerminatedWorkers))
      .toBeGreaterThanOrEqual(1);
  });

  test("@runtime generator reports progress and repeats exactly for a fixed seed", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/tiny-cats-model/#/generate");
    await expect(page.getByText(/Verified model · local/)).toBeVisible({ timeout: 120_000 });

    const sliders = page.getByRole("slider");
    await sliders.nth(0).press("Home"); // minimum 10 sampling steps
    await sliders.nth(1).press("Home"); // CFG 0 avoids the unconditional pass
    await page.getByTestId("noise-seed").fill("5150");

    const generate = page.getByTestId("generate-button");
    await generate.click();
    await expect(page.getByTestId("generation-progress")).toBeVisible({ timeout: 30_000 });
    const image = page.getByTestId("generated-image");
    await expect(image).toBeVisible({ timeout: 120_000 });
    const firstImage = await image.getAttribute("src");

    await generate.click();
    await expect(image).toBeVisible({ timeout: 120_000 });
    await expect(image).toHaveAttribute("src", firstImage ?? "");
  });

  for (const route of ["classify", "generate", "benchmark"]) {
    test(`@runtime ${route} has no WCAG 2.1 A/AA violations`, async ({ page }) => {
      await page.context().route("**/models/*.onnx", (request) => request.abort());
      await page.goto(`/tiny-cats-model/#/${route}`);
      const violations = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();
      expect(violations.violations, JSON.stringify(violations.violations, null, 2)).toEqual([]);
    });
  }
});

declare global {
  interface Window {
    __tinyCatsTerminatedWorkers: number;
    __tinyCatsPreprocessedTensor: number[] | null;
    __tinyCatsDecodedDimensions: { width: number; height: number } | null;
    __TINY_CATS_CAPTURE_PREPROCESSING__?: (
      tensor: Float32Array,
      dimensions: { width: number; height: number }
    ) => void;
    __TINY_CATS_TEST_ASSETS__?: Partial<
      Record<string, { sha256?: string; forceWebGpuFailure?: boolean }>
    >;
}
}
