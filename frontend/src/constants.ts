export const PAGES = [
  { name: "Cat Classifier", path: "/classify" },
  { name: "Cat Generator", path: "/generate" },
  { name: "Benchmark", path: "/benchmark" },
];

export type ModelType = "cats";

// HuggingFace Hub base URL for model loading (ADR-034)
// Models are loaded from HF Hub CDN with local fallback
export const HF_HUB_BASE_URL =
  "https://huggingface.co/d4oit/tiny-cats-model/resolve/main/";

export interface ModelConfig {
  modelPath: string;
  localFallback: string;
  imgDims: number[];
  numClasses: number;
  classNames: string[];
  mean: number[];
  std: number[];
}

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  cats: {
    modelPath: HF_HUB_BASE_URL + "cats_quantized.onnx",
    localFallback: "/models/cats_quantized.onnx",
    imgDims: [224, 224],
    numClasses: 2,
    classNames: ["cat", "not_cat"],
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
};

export const CAT_BREEDS = [
  { index: 0, name: "Abyssinian" },
  { index: 1, name: "Bengal" },
  { index: 2, name: "Birman" },
  { index: 3, name: "Bombay" },
  { index: 4, name: "British Shorthair" },
  { index: 5, name: "Egyptian Mau" },
  { index: 6, name: "Maine Coon" },
  { index: 7, name: "Persian" },
  { index: 8, name: "Ragdoll" },
  { index: 9, name: "Russian Blue" },
  { index: 10, name: "Siamese" },
  { index: 11, name: "Sphynx" },
  { index: 12, name: "Other" },
];

export interface GeneratorConfig {
  modelPath: string;
  localFallback: string;
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
  modelPath: HF_HUB_BASE_URL + "generator_quantized.onnx",
  localFallback: "/models/generator_quantized.onnx",
  imgDims: [128, 128],
  numBreeds: 13,
  defaultSteps: 50,
  minSteps: 10,
  maxSteps: 100,
  defaultCfgScale: 1.5,
  minCfgScale: 1.0,
  maxCfgScale: 3.0,
};
