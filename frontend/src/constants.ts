import modelContract from "./generated/modelContract.json";
import { getModelAsset, type ModelAsset } from "./modelAssets";

export const PAGES = [
  { name: "Cat Classifier", path: "/classify" },
  { name: "Cat Generator", path: "/generate" },
  { name: "Benchmark", path: "/benchmark" },
];

export type ModelType = "cats";

/** Binary classifier labels in output order (issue #197). */
export const CLASSIFIER_LABELS: string[] = modelContract.classifier.output.labels;

const classifierPreprocessing = modelContract.classifier.preprocessing;

export interface ModelConfig {
  modelType: "classifier" | "generator";
  asset: ModelAsset;
  imgDims: number[];
  numClasses: number;
  classNames: string[];
  mean: number[];
  std: number[];
}

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  cats: {
    modelType: "classifier",
    asset: getModelAsset("classifier"),
    imgDims: [
      classifierPreprocessing.center_crop.height,
      classifierPreprocessing.center_crop.width,
    ],
    numClasses: modelContract.classifier.architecture.num_classes,
    classNames: CLASSIFIER_LABELS,
    mean: classifierPreprocessing.mean,
    std: classifierPreprocessing.std,
  },
};

/** Generator conditions: 12 cat breeds + Other at index 12. */
export const CAT_BREEDS = modelContract.generator.conditions.map(
  (condition: { index: number; name: string }) => ({
    index: condition.index,
    name: condition.name,
  })
);

export const GENERATOR_UNCONDITIONAL_INDEX = modelContract.generator.unconditional_index;

export interface GeneratorConfig {
  modelType: "generator";
  asset: ModelAsset;
  imgDims: number[];
  numBreeds: number;
  defaultSteps: number;
  minSteps: number;
  maxSteps: number;
  defaultCfgScale: number;
  minCfgScale: number;
  maxCfgScale: number;
}

export const GENERATOR_CONFIG: GeneratorConfig = {
  modelType: "generator",
  asset: getModelAsset("generator"),
  imgDims: [128, 128],
  numBreeds: CAT_BREEDS.length,
  defaultSteps: modelContract.generator.sampling.default_steps,
  minSteps: 10,
  maxSteps: 100,
  defaultCfgScale: modelContract.generator.sampling.default_cfg_scale,
  minCfgScale: modelContract.generator.sampling.min_cfg_scale,
  maxCfgScale: modelContract.generator.sampling.max_cfg_scale,
};

/** Versioned preprocessing contract shared by classify, benchmark and workers. */
export const CLASSIFIER_PREPROCESSING = classifierPreprocessing;