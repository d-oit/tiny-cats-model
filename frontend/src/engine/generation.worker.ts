import * as ort from "onnxruntime-web";
import * as Comlink from "comlink";

import { GENERATOR_CONFIG, type GeneratorConfig } from "../constants";

ort.env.wasm.wasmPaths = "/tiny-cats-model/";

async function getExecutionProvider(): Promise<["wasm"] | ["webgpu"]> {
  if (typeof navigator !== "undefined") {
    try {
      const gpu = (navigator as any).gpu;
      if (gpu) {
        return ["webgpu"];
      }
    } catch {
      // WebGPU not available
    }
  }
  return ["wasm"];
}

async function loadOrt(): Promise<typeof ort> {
  const ep = await getExecutionProvider();
  if (ep[0] === "webgpu") {
    const ortWebGpu = await import("onnxruntime-web/webgpu");
    return ortWebGpu as unknown as typeof ort;
  }
  return ort;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
if (typeof navigator !== "undefined" && (self as any).crossOriginIsolated) {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 2;
} else {
  ort.env.wasm.numThreads = 1;
}

ort.env.wasm.proxy = false;

let ortInstance: typeof ort | null = null;

export interface GenerationProgress {
  step: number;
  totalSteps: number;
  stepTime: number;
  totalTime: number;
  imageData?: ImageData;
}

export interface GenerationResult {
  imageData: ImageData;
  totalTime: number;
  avgStepTime: number;
}

export interface GenerationOptions {
  breedIndex: number;
  steps: number;
  cfgScale: number;
  noiseSeed?: number;
}

/**
 * Load model with HF Hub primary and local fallback (ADR-034)
 */
async function fetchModelWithFallback(primaryPath: string, fallbackPath: string): Promise<ArrayBuffer> {
  // Try primary (HF Hub) first
  try {
    console.log(`Loading model from HF Hub: ${primaryPath}`);
    const response = await fetch(primaryPath, { mode: "cors", credentials: "omit" });
    if (!response.ok) {
      throw new Error(`HF Hub returned ${response.status}`);
    }
    const data = await response.arrayBuffer();
    console.log("Model loaded successfully from HF Hub");
    return data;
  } catch (error) {
    console.warn(`HF Hub load failed: ${error}`);
    console.log(`Falling back to local model: ${fallbackPath}`);
    
    // Fall back to local model
    const response = await fetch(fallbackPath, { mode: "cors", credentials: "omit" });
    if (!response.ok) {
      throw new Error(`Local model also failed: ${response.status}`);
    }
    const data = await response.arrayBuffer();
    console.log("Model loaded successfully from local fallback");
    return data;
  }
}

class GenerationEngine {
  session: ort.InferenceSession | null = null;
  config: GeneratorConfig = GENERATOR_CONFIG;
  executionProvider: "wasm" | "webgpu" = "wasm";

  async loadModel(): Promise<string> {
    console.log("Loading generator model...");
    try {
      ortInstance = await loadOrt();
      this.executionProvider = (await getExecutionProvider())[0];
      console.log("Using execution provider:", this.executionProvider);

      // Use HF Hub with local fallback (ADR-034)
      const modelData = await fetchModelWithFallback(
        this.config.modelPath,
        this.config.localFallback
      );
      const modelDataUint8 = new Uint8Array(modelData);

      console.log("Creating inference session...");

      this.session = await ortInstance.InferenceSession.create(modelDataUint8, {
        executionProviders: [this.executionProvider],
        graphOptimizationLevel: "all"
      });

      console.log("Generator model loaded");
    } catch (e) {
      console.log("Error while loading the generator model");
      console.error(e);
      throw e;
    }
    return "Model Loaded";
  }

  private generateGaussianNoise(size: number, seed?: number): Float32Array {
    const data = new Float32Array(size);

    // Simple seeded random number generator (mulberry32)
    let random = seed !== undefined ? this.mulberry32(seed) : Math.random;

    for (let i = 0; i < size; i += 2) {
      let u = 0, v = 0;
      while (u === 0) u = random();
      while (v === 0) v = random();
      const mag = Math.sqrt(-2.0 * Math.log(u));
      const z0 = mag * Math.cos(2.0 * Math.PI * v);
      const z1 = mag * Math.sin(2.0 * Math.PI * v);
      data[i] = z0;
      if (i + 1 < size) {
        data[i + 1] = z1;
      }
    }
    return data;
  }

  private mulberry32(seed: number): () => number {
    return function() {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  private async predictVelocity(
    noise: Float32Array,
    timestep: number,
    breedIndex: number,
    cfgScale: number
  ): Promise<Float32Array> {
    if (!this.session) throw new Error("Model not loaded");
    if (!ortInstance) throw new Error("ORT not loaded");

    const [height, width] = this.config.imgDims;
    const inputSize = 3 * height * width;

    // Create noise tensor
    const noiseTensor = new ortInstance.Tensor("float32", noise, [1, 3, height, width]);

    // Create timestep tensor
    const timestepTensor = new ortInstance.Tensor("float32", new Float32Array([timestep]), [1]);

    // Create breed tensor
    const breedTensor = new ortInstance.Tensor("int64", new BigInt64Array([BigInt(breedIndex)]), [1]);

    // Run inference for conditional prediction
    const result = await this.session.run({
      noise: noiseTensor,
      timestep: timestepTensor,
      breed: breedTensor
    });

    const conditionalVelocity = (result.velocity as ort.Tensor).data as Float32Array;

    // If CFG scale > 1, also run unconditional prediction
    if (cfgScale > 1.0) {
      const uncondBreedTensor = new ortInstance.Tensor("int64", new BigInt64Array([BigInt(-1)]), [1]);

      const uncondResult = await this.session.run({
        noise: noiseTensor,
        timestep: timestepTensor,
        breed: uncondBreedTensor
      });

      const unconditionalVelocity = (uncondResult.velocity as ort.Tensor).data as Float32Array;

      // Apply classifier-free guidance: v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
      const guidedVelocity = new Float32Array(inputSize);
      for (let i = 0; i < inputSize; i++) {
        guidedVelocity[i] = unconditionalVelocity[i] + cfgScale * (conditionalVelocity[i] - unconditionalVelocity[i]);
      }

      return guidedVelocity;
    }

    return conditionalVelocity;
  }

  private tensorToImageData(tensor: Float32Array, width: number, height: number): ImageData {
    const imageData = new ImageData(width, height);
    const data = imageData.data;
    const totalPixels = width * height;

    for (let i = 0; i < totalPixels; i++) {
      // Convert from [-1, 1] to [0, 255]
      const r = Math.round(((tensor[i] + 1) / 2) * 255);
      const g = Math.round(((tensor[i + totalPixels] + 1) / 2) * 255);
      const b = Math.round(((tensor[i + totalPixels * 2] + 1) / 2) * 255);

      data[i * 4] = Math.max(0, Math.min(255, r));
      data[i * 4 + 1] = Math.max(0, Math.min(255, g));
      data[i * 4 + 2] = Math.max(0, Math.min(255, b));
      data[i * 4 + 3] = 255; // Alpha
    }

    return imageData;
  }

  async generate(
    options: GenerationOptions,
    onProgress?: (progress: GenerationProgress) => void
  ): Promise<GenerationResult> {
    if (!this.session) throw new Error("Model not loaded");

    const [height, width] = this.config.imgDims;
    const inputSize = 3 * height * width;
    const { breedIndex, steps, cfgScale, noiseSeed } = options;

    // Initialize with Gaussian noise
    let x = this.generateGaussianNoise(inputSize, noiseSeed);

    const startTime = performance.now();
    let lastStepTime = startTime;

    // Euler ODE integration for flow matching
    // Flow matching: dx/dt = v(x, t), integrate from t=0 to t=1
    const dt = 1.0 / steps;

    for (let step = 0; step < steps; step++) {
      const t = step / steps;

      // Predict velocity at current timestep
      const velocity = await this.predictVelocity(x, t, breedIndex, cfgScale);

      // Euler step: x(t+dt) = x(t) + dt * v(x, t)
      for (let i = 0; i < inputSize; i++) {
        x[i] += dt * velocity[i];
      }

      const currentTime = performance.now();
      const stepTime = currentTime - lastStepTime;
      const totalTime = currentTime - startTime;
      lastStepTime = currentTime;

      // Report progress every few steps
      if (onProgress && (step % 5 === 0 || step === steps - 1)) {
        const imageData = this.tensorToImageData(x, width, height);
        onProgress({
          step: step + 1,
          totalSteps: steps,
          stepTime,
          totalTime,
          imageData
        });
      }
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;

    return {
      imageData: this.tensorToImageData(x, width, height),
      totalTime,
      avgStepTime: totalTime / steps
    };
  }
}

export default GenerationEngine;

Comlink.expose(new GenerationEngine());
