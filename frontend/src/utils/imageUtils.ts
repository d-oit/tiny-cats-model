import { MODEL_CONFIGS, type ModelType } from "../constants";

export async function loadImageAsTensor(
  base64Input: string,
  modelType: ModelType
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const configs = MODEL_CONFIGS[modelType];
    const [height, width] = configs.imgDims;
    const { mean, std } = configs;
    
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = base64Input;

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }

      ctx.drawImage(img, 0, 0, width, height);

      const imageData = ctx.getImageData(0, 0, width, height).data;
      const totalPixels = width * height;
      
      const tensor = new Float32Array(3 * totalPixels);
      
      for (let i = 0; i < totalPixels; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];

        tensor[i] = (r / 255.0 - mean[0]) / std[0];
        tensor[i + totalPixels] = (g / 255.0 - mean[1]) / std[1];
        tensor[i + totalPixels * 2] = (b / 255.0 - mean[2]) / std[2];
      }

      resolve(tensor);
    };

    img.onerror = (err) => reject(err);
  });
}

export function imageDataToCanvas(
  imageData: ImageData,
  width: number,
  height: number
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.putImageData(imageData, 0, 0);
  }
  return canvas;
}

export async function centerCropAndResize(
  base64Input: string,
  targetWidth: number,
  targetHeight: number
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = base64Input;

    img.onload = () => {
      const minDim = Math.min(img.width, img.height);
      const sx = (img.width - minDim) / 2;
      const sy = (img.height - minDim) / 2;

      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }

      ctx.drawImage(
        img, 
        sx, sy, minDim, minDim,
        0, 0, targetWidth, targetHeight
      );

      resolve(canvas.toDataURL("image/png"));
    };

    img.onerror = (err) => reject(err);
  });
}
