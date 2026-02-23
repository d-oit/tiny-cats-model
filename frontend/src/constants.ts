export const PAGES = [
  { name: "Cat Classifier", path: "/classify" },
];

export type ModelType = "cats";

export interface ModelConfig {
  modelPath: string;
  imgDims: number[];
  numClasses: number;
  classNames: string[];
  mean: number[];
  std: number[];
}

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  cats: {
    modelPath: "/tiny-cats-model/models/cats.onnx",
    imgDims: [224, 224],
    numClasses: 2,
    classNames: ["cat", "not_cat"],
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
};
