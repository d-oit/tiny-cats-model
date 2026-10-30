import * as ort from "onnxruntime-web";
import * as Comlink from "comlink";

import { GENERATOR_CONFIG, GENERATOR_UNCONDITIONAL_INDEX } from "../constants";
import {
  fetchVerifiedModel,
  mergeAsset,
  type ModelAssetOverride,
} from "../modelAssets";
import {
  createSessionWithFallback,
  resolveInputName,
  resolveOutputName,
  type ExecutionProvider,
} from "./ortRuntime";

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

export interface GeneratorLoadInfo {
  provider: ExecutionProvider;
  source: "hub" | "local";
  sha256: string;
  size: number;
}

export interface GenerationEngineApi {
  loadModel(override?: ModelAssetOverride): Promise<GeneratorLoadInfo>;
  generate(
    options: GenerationOptions,
    onProgress?: (progress: GenerationProgress) => void
  ): Promise<GenerationResult>;
  dispose(): Promise<void>;
}

const MIN_STEPS = 1;
const MAX_STEPS = 200;

class GenerationEngine implements GenerationEngineApi {
  session: ort.InferenceSession | null = null;
  runtime: typeof ort | null = null;
  config = GENERATOR_CONFIG;
  executionProvider: ExecutionProvider = "wasm";
  inputNames: { noise: string; timestep: string; breed: string } = {
    noise: "noise",
    timestep: "timestep",
    breed: "breed",
  };
  outputName = "velocity";

  async loadModel(override?: ModelAssetOverride): Promise<GeneratorLoadInfo> {
    await this.dispose();
    const asset = mergeAsset(this.config.asset, override);
    const verified = await fetchVerifiedModel(asset);
    const created = await createSessionWithFallback(
      new Uint8Array(verified.data),
      override?.forceWebGpuFailure === true
    );

    this.runtime = created.ort;
    this.session = created.session;
    this.executionProvider = created.provider;
    this.inputNames = {
      noise: resolveInputName(created.session, "noise"),
      timestep: resolveInputName(created.session, "timestep"),
      breed: resolveInputName(created.session, "breed"),
    };
    this.outputName = resolveOutputName(created.session, "velocity");

    return {
      provider: created.provider,
      source: verified.source,
      sha256: verified.sha256,
      size: verified.size,
    };
  }

  private generateGaussianNoise(size: number, seed?: number): Float32Array {
    const data = new Float32Array(size);
    const random = seed !== undefined ? this.mulberry32(seed) : Math.random;
    for (let i = 0; i < size; i += 2) {
      let u = 0;
      let v = 0;
      while (u === 0) u = random();
      while (v === 0) v = random();
      const magnitude = Math.sqrt(-2.0 * Math.log(u));
      data[i] = magnitude * Math.cos(2.0 * Math.PI * v);
      if (i + 1 < size) {
        data[i + 1] = magnitude * Math.sin(2.0 * Math.PI * v);
      }
    }
    return data;
  }

  private mulberry32(seed: number): () => number {
    let state = seed >>> 0;
    return function () {
      state += 0x6d2b79f5;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  private async predictVelocity(
    noise: Float32Array,
    timestep: number,
    breedIndex: number,
    cfgScale: number
  ): Promise<Float32Array> {
    if (!this.session || !this.runtime) throw new Error("Model not loaded");
    const [height, width] = this.config.imgDims;
    const inputSize = 3 * height * width;

    const noiseTensor = new this.runtime.Tensor("float32", noise, [
      1,
      3,
      height,
      width,
    ]);
    const timestepTensor = new this.runtime.Tensor(
      "float32",
      new Float32Array([timestep]),
      [1]
    );
    const breedTensor = new this.runtime.Tensor(
      "int64",
      new BigInt64Array([BigInt(breedIndex)]),
      [1]
    );

    const result = await this.session.run({
      [this.inputNames.noise]: noiseTensor,
      [this.inputNames.timestep]: timestepTensor,
      [this.inputNames.breed]: breedTensor,
    });
    const output = result[this.outputName] as ort.Tensor | undefined;
    if (!output) {
      throw new Error(`Generator result did not contain '${this.outputName}'`);
    }
    const conditionalVelocity = output.data as Float32Array;
    if (conditionalVelocity.length !== inputSize) {
      throw new Error(
        `Generator velocity has ${conditionalVelocity.length} values; expected ${inputSize}`
      );
    }

    if (cfgScale <= 1.0) return conditionalVelocity;

    const uncondBreedTensor = new this.runtime.Tensor(
      "int64",
      new BigInt64Array([BigInt(GENERATOR_UNCONDITIONAL_INDEX)]),
      [1]
    );
    const uncondResult = await this.session.run({
      [this.inputNames.noise]: noiseTensor,
      [this.inputNames.timestep]: timestepTensor,
      [this.inputNames.breed]: uncondBreedTensor,
    });
    const unconditionalOutput = uncondResult[this.outputName] as ort.Tensor | undefined;
    if (!unconditionalOutput) {
      throw new Error(`Unconditional result did not contain '${this.outputName}'`);
    }
    const unconditionalVelocity = unconditionalOutput.data as Float32Array;
    if (unconditionalVelocity.length !== inputSize) {
      throw new Error("Unconditional generator velocity has the wrong length");
    }

    const guided = new Float32Array(inputSize);
    for (let i = 0; i < inputSize; i++) {
      guided[i] =
        unconditionalVelocity[i] +
        cfgScale * (conditionalVelocity[i] - unconditionalVelocity[i]);
    }
    return guided;
  }

  private tensorToImageData(
    tensor: Float32Array,
    width: number,
    height: number
  ): ImageData {
    const imageData = new ImageData(width, height);
    const data = imageData.data;
    const totalPixels = width * height;
    for (let i = 0; i < totalPixels; i++) {
      const r = Math.round(((tensor[i] + 1) / 2) * 255);
      const g = Math.round(((tensor[i + totalPixels] + 1) / 2) * 255);
      const b = Math.round(((tensor[i + totalPixels * 2] + 1) / 2) * 255);
      data[i * 4] = Math.max(0, Math.min(255, r));
      data[i * 4 + 1] = Math.max(0, Math.min(255, g));
      data[i * 4 + 2] = Math.max(0, Math.min(255, b));
      data[i * 4 + 3] = 255;
    }
    return imageData;
  }

  private validateOptions(options: GenerationOptions): void {
    const { breedIndex, steps, cfgScale, noiseSeed } = options;
    if (!Number.isInteger(breedIndex) || breedIndex < 0 || breedIndex >= this.config.numBreeds) {
      throw new Error(
        `Breed index must be an integer in [0, ${this.config.numBreeds - 1}], got ${breedIndex}`
      );
    }
    if (!Number.isInteger(steps) || steps < MIN_STEPS || steps > MAX_STEPS) {
      throw new Error(`Steps must be an integer in [${MIN_STEPS}, ${MAX_STEPS}]`);
    }
    if (!Number.isFinite(cfgScale) || cfgScale < 0 || cfgScale > 10) {
      throw new Error("CFG scale must be a finite number in [0, 10]");
    }
    if (noiseSeed !== undefined && (!Number.isFinite(noiseSeed) || noiseSeed < 0)) {
      throw new Error("Noise seed must be a non-negative finite number");
    }
  }

  async generate(
    options: GenerationOptions,
    onProgress?: (progress: GenerationProgress) => void
  ): Promise<GenerationResult> {
    if (!this.session) throw new Error("Model not loaded");
    this.validateOptions(options);

    const [height, width] = this.config.imgDims;
    const inputSize = 3 * height * width;
    const { breedIndex, steps, cfgScale, noiseSeed } = options;
    const x = this.generateGaussianNoise(inputSize, noiseSeed);

    const startTime = performance.now();
    let lastStepTime = startTime;
    const dt = 1.0 / steps;

    for (let step = 0; step < steps; step++) {
      const t = step / steps;
      const velocity = await this.predictVelocity(x, t, breedIndex, cfgScale);
      for (let i = 0; i < inputSize; i++) {
        x[i] += dt * velocity[i];
      }

      const currentTime = performance.now();
      const stepTime = currentTime - lastStepTime;
      const totalTime = currentTime - startTime;
      lastStepTime = currentTime;

      if (onProgress && (step % 5 === 0 || step === steps - 1)) {
        onProgress({
          step: step + 1,
          totalSteps: steps,
          stepTime,
          totalTime,
          imageData: this.tensorToImageData(x, width, height),
        });
      }
    }

    const totalTime = performance.now() - startTime;
    return {
      imageData: this.tensorToImageData(x, width, height),
      totalTime,
      avgStepTime: totalTime / steps,
    };
  }

  async dispose(): Promise<void> {
    if (this.session) {
      try {
        await this.session.release();
      } catch {
        // Ignore teardown errors; worker termination is authoritative.
      }
    }
    this.session = null;
    this.runtime = null;
  }
}

Comlink.expose(new GenerationEngine());