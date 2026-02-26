import * as ort from "onnxruntime-web";
import * as Comlink from "comlink";

import { MODEL_CONFIGS, type ModelConfig, type ModelType } from "../constants";

ort.env.wasm.wasmPaths = "/tiny-cats-model/";

async function getExecutionProvider(): Promise<["wasm"] | ["webgpu"]> {
  if (typeof navigator !== "undefined") {
    try {
      // WebGPU type check: navigator.gpu may not be typed in all environments
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

export interface ClassificationResult {
  classIndex: number;
  className: string;
  confidence: number;
}

export interface ClassificationTelemetry {
  image: Float32Array;
  result: ClassificationResult;
  probabilities: number[];
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

class InferenceEngine {
  session: ort.InferenceSession | null = null;
  modelType: string = "";
  configs: ModelConfig | null = null;
  executionProvider: "wasm" | "webgpu" = "wasm";

  async loadModel(path: string, modelType: ModelType) {
    console.log("Loading model...");
    try {
      this.modelType = modelType;
      this.configs = MODEL_CONFIGS[modelType];

      ortInstance = await loadOrt();
      this.executionProvider = (await getExecutionProvider())[0];
      console.log("Using execution provider:", this.executionProvider);

      // Use HF Hub with local fallback (ADR-034)
      const modelData = await fetchModelWithFallback(
        this.configs.modelPath,
        this.configs.localFallback
      );
      const modelDataUint8 = new Uint8Array(modelData);

      console.log("Creating inference session...");

      this.session = await ortInstance.InferenceSession.create(modelDataUint8, {
        executionProviders: [this.executionProvider],
        graphOptimizationLevel: "all"
      });

      console.log("Model loaded");
    } catch (e) {
      console.log("Error while loading the model");
      console.error(e);
      throw e;
    }
    return "Model Loaded";
  }

  async classify(imageTensor: Float32Array): Promise<ClassificationTelemetry> {
    if (!this.session) throw new Error("Model not loaded");
    if (!this.configs) throw new Error("Model config not loaded");
    if (!ortInstance) throw new Error("ORT not loaded");

    const [height, width] = this.configs.imgDims;
    const inputDims = [1, 3, height, width];

    const inputTensor = new ortInstance.Tensor("float32", imageTensor, inputDims);

    const result = await this.session.run({
      input: inputTensor
    });

    const output = result.output as ort.Tensor;
    const logits = output.data as Float32Array;

    const expLogits = logits.map(Math.exp);
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probabilities = expLogits.map(x => x / sumExp);

    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[maxIndex];

    const classResult: ClassificationResult = {
      classIndex: maxIndex,
      className: this.configs.classNames[maxIndex],
      confidence: confidence
    };

    return {
      image: imageTensor,
      result: classResult,
      probabilities: Array.from(probabilities)
    };
  }
}

Comlink.expose(new InferenceEngine());
