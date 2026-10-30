import * as ort from "onnxruntime-web";
import * as Comlink from "comlink";

import { MODEL_CONFIGS, type ModelType } from "../constants";
import {
  fetchVerifiedModel,
  mergeAsset,
  type ModelAssetOverride,
} from "../modelAssets";
import {
  createSessionWithFallback,
  resolveInputName,
  resolveOutputName,
  stableSoftmax,
  type ExecutionProvider,
} from "./ortRuntime";

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

export interface ModelLoadInfo {
  provider: ExecutionProvider;
  source: "hub" | "local";
  sha256: string;
  size: number;
}

export interface InferenceEngineApi {
  loadModel(modelType: ModelType, override?: ModelAssetOverride): Promise<ModelLoadInfo>;
  classify(imageTensor: Float32Array): Promise<ClassificationTelemetry>;
  dispose(): Promise<void>;
}

class InferenceEngine implements InferenceEngineApi {
  session: ort.InferenceSession | null = null;
  runtime: typeof ort | null = null;
  configs = MODEL_CONFIGS.cats;
  executionProvider: ExecutionProvider = "wasm";
  inputName = "input";
  outputName = "output";

  async loadModel(
    modelType: ModelType,
    override?: ModelAssetOverride
  ): Promise<ModelLoadInfo> {
    if (modelType !== "cats") {
      throw new Error(`Unsupported classifier model type: ${String(modelType)}`);
    }
    await this.dispose();

    const config = MODEL_CONFIGS[modelType];
    this.configs = config;
    const asset = mergeAsset(config.asset, override);
    const verified = await fetchVerifiedModel(asset);
    const created = await createSessionWithFallback(
      new Uint8Array(verified.data),
      override?.forceWebGpuFailure === true
    );

    this.runtime = created.ort;
    this.session = created.session;
    this.executionProvider = created.provider;
    this.inputName = resolveInputName(created.session, asset.contract.input.name);
    this.outputName = resolveOutputName(created.session, asset.contract.output.name);

    return {
      provider: created.provider,
      source: verified.source,
      sha256: verified.sha256,
      size: verified.size,
    };
  }

  async classify(imageTensor: Float32Array): Promise<ClassificationTelemetry> {
    if (!this.session || !this.runtime) throw new Error("Model not loaded");
    const [height, width] = this.configs.imgDims;
    const expected = 3 * height * width;
    if (imageTensor.length !== expected) {
      throw new Error(
        `Classifier tensor has ${imageTensor.length} values; expected ${expected}`
      );
    }

    const inputTensor = new this.runtime.Tensor("float32", imageTensor, [
      1,
      3,
      height,
      width,
    ]);
    const result = await this.session.run({ [this.inputName]: inputTensor });
    const output = result[this.outputName] as ort.Tensor | undefined;
    if (!output) {
      throw new Error(
        `Classifier result did not contain output '${this.outputName}'`
      );
    }
    const logits = output.data as Float32Array;
    const labels = this.configs.classNames;
    if (logits.length !== labels.length) {
      throw new Error(
        `Classifier emitted ${logits.length} logits but metadata declares ` +
          `${labels.length} labels (${labels.join(", ")})`
      );
    }

    const probabilities = stableSoftmax(logits);
    let maxIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[maxIndex]) maxIndex = i;
    }

    return {
      image: imageTensor,
      result: {
        classIndex: maxIndex,
        className: labels[maxIndex],
        confidence: probabilities[maxIndex],
      },
      probabilities: Array.from(probabilities),
    };
  }

  async dispose(): Promise<void> {
    if (this.session) {
      try {
        await this.session.release();
      } catch {
        // Session teardown failures must not block worker termination.
      }
    }
    this.session = null;
    this.runtime = null;
  }
}

Comlink.expose(new InferenceEngine());