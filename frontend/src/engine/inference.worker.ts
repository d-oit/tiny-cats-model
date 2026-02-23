import * as ort from "onnxruntime-web";
import * as Comlink from "comlink";

import { MODEL_CONFIGS, type ModelConfig, type ModelType } from "../constants";

ort.env.wasm.wasmPaths = "/tiny-cats-model/";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
if (typeof navigator !== "undefined" && (self as any).crossOriginIsolated) {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 2;
} else {
  ort.env.wasm.numThreads = 1;
}

ort.env.wasm.proxy = false;

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

class InferenceEngine {
  session: ort.InferenceSession | null = null;
  modelType: string = "";
  configs: ModelConfig | null = null;

  async loadModel(path: string, modelType: ModelType) {
    console.log("Loading model...");
    try {
      this.modelType = modelType;
      this.configs = MODEL_CONFIGS[modelType];

      console.log("Fetching model from:", path);
      const response = await fetch(path, { mode: "cors", credentials: "omit" });
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status}`);
      }
      const modelData = await response.arrayBuffer();
      const modelDataUint8 = new Uint8Array(modelData);

      console.log("Creating inference session...");

      this.session = await ort.InferenceSession.create(modelDataUint8, {
        executionProviders: ["wasm"],
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

    const [height, width] = this.configs.imgDims;
    const inputDims = [1, 3, height, width];

    const inputTensor = new ort.Tensor("float32", imageTensor, inputDims);

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
