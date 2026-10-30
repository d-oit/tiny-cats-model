import { CLASSIFIER_PREPROCESSING } from "../constants";

export type SupportedImageFormat = "jpeg" | "png" | "webp" | "gif" | "bmp";

export interface ImageLimits {
  maxFileBytes: number;
  maxWidth: number;
  maxHeight: number;
  maxPixels: number;
}

export const DEFAULT_IMAGE_LIMITS: ImageLimits = {
  maxFileBytes: 10 * 1024 * 1024,
  maxWidth: 8192,
  maxHeight: 8192,
  maxPixels: 40_000_000,
};

export const SUPPORTED_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/bmp",
] as const;

const MIME_TO_FORMAT: Record<string, SupportedImageFormat> = {
  "image/jpeg": "jpeg",
  "image/jpg": "jpeg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
  "image/bmp": "bmp",
};

export interface DecodedImage {
  source: CanvasImageSource;
  width: number;
  height: number;
  format: SupportedImageFormat;
  close: () => void;
}


export function detectImageFormat(bytes: Uint8Array): SupportedImageFormat | null {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "jpeg";
  }
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return "png";
  }
  if (
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.slice(0, 4)) === "RIFF" &&
    String.fromCharCode(...bytes.slice(8, 12)) === "WEBP"
  ) {
    return "webp";
  }
  if (
    bytes.length >= 6 &&
    (String.fromCharCode(...bytes.slice(0, 6)) === "GIF87a" ||
      String.fromCharCode(...bytes.slice(0, 6)) === "GIF89a")
  ) {
    return "gif";
  }
  if (bytes.length >= 2 && bytes[0] === 0x42 && bytes[1] === 0x4d) {
    return "bmp";
  }
  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Validate size, declared MIME and magic bytes before decoding. Declared MIME
 * is never trusted alone: a renamed non-image file must fail here.
 */
export async function validateImageFile(
  file: File,
  limits: ImageLimits = DEFAULT_IMAGE_LIMITS
): Promise<SupportedImageFormat> {
  if (file.size === 0) {
    throw new Error("The selected file is empty.");
  }
  if (file.size > limits.maxFileBytes) {
    throw new Error(
      `File is ${formatBytes(file.size)}; the limit is ${formatBytes(
        limits.maxFileBytes
      )}. Choose a smaller image.`
    );
  }
  const declared = file.type.toLowerCase();
  if (declared && !(declared in MIME_TO_FORMAT)) {
    throw new Error(
      "Unsupported file type. Upload a JPEG, PNG, WebP, GIF or BMP image."
    );
  }
  const header = new Uint8Array(await file.slice(0, 16).arrayBuffer());
  const detected = detectImageFormat(header);
  if (!detected) {
    throw new Error(
      "The file contents are not a supported image, regardless of its name or declared type."
    );
  }
  if (declared && MIME_TO_FORMAT[declared] !== detected) {
    throw new Error(
      `File contents are ${detected.toUpperCase()} but the declared type is ${declared}.`
    );
  }
  return detected;
}

function invalidDimensions(
  width: number,
  height: number,
  limits: ImageLimits
): string | null {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width < 1 || height < 1) {
    return "The image has invalid dimensions.";
  }
  if (width > limits.maxWidth || height > limits.maxHeight) {
    return (
      `Image is ${width}x${height}; the per-axis limit is ` +
      `${limits.maxWidth}x${limits.maxHeight}.`
    );
  }
  if (width * height > limits.maxPixels) {
    return (
      `Image has ${width * height} pixels; the limit is ${limits.maxPixels}. ` +
      "Choose a smaller image."
    );
  }
  return null;
}

/**
 * Decode an uploaded file with EXIF orientation applied.
 *
 * Uses createImageBitmap (which honours `imageOrientation: "from-image"`),
 * falling back to an <img> decode when the bitmap API is unavailable. Both
 * paths reject corrupt content and enforce dimension/pixel limits *before*
 * any canvas/tensor allocation.
 */
export async function decodeImageFile(
  file: File,
  limits: ImageLimits = DEFAULT_IMAGE_LIMITS
): Promise<DecodedImage> {
  const format = await validateImageFile(file, limits);
  const mime =
    format === "bmp" ? "image/bmp" : format === "jpeg" ? "image/jpeg" : `image/${format}`;
  const blob = file.type === mime ? file : file.slice(0, file.size, mime);

  try {
    const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
    const problem = invalidDimensions(bitmap.width, bitmap.height, limits);
    if (problem) {
      bitmap.close();
      throw new Error(problem);
    }
    return {
      source: bitmap,
      width: bitmap.width,
      height: bitmap.height,
      format,
      close: () => bitmap.close(),
    };
  } catch (bitmapError) {
    if (bitmapError instanceof Error && bitmapError.message.includes("limit")) {
      throw bitmapError;
    }
    return decodeWithImageElement(blob, format, limits, bitmapError);
  }
}

function decodeWithImageElement(
  blob: Blob,
  format: SupportedImageFormat,
  limits: ImageLimits,
  bitmapError: unknown
): Promise<DecodedImage> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      const problem = invalidDimensions(image.naturalWidth, image.naturalHeight, limits);
      if (problem) {
        URL.revokeObjectURL(url);
        reject(new Error(problem));
        return;
      }
      resolve({
        source: image,
        width: image.naturalWidth,
        height: image.naturalHeight,
        format,
        close: () => URL.revokeObjectURL(url),
      });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      const detail =
        bitmapError instanceof Error ? ` (${bitmapError.message})` : "";
      reject(new Error(`The image could not be decoded; it may be corrupt${detail}`));
    };
    image.src = url;
  });
}

/**
 * Canonical classifier preprocessing: RGB, centre-crop the shorter side,
 * resize to the contract size, scale to [0,1], ImageNet-normalise, NCHW
 * float32. Shared by the classifier page and any future browser benchmark
 * image path so transforms cannot drift.
 */
export function preprocessImageSource(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth = CLASSIFIER_PREPROCESSING.center_crop.width,
  targetHeight = CLASSIFIER_PREPROCESSING.center_crop.height,
  mean: number[] = CLASSIFIER_PREPROCESSING.mean,
  std: number[] = CLASSIFIER_PREPROCESSING.std,
  resizeShortSide = CLASSIFIER_PREPROCESSING.resize.short_side
): Float32Array {
  const minDimension = Math.min(sourceWidth, sourceHeight);
  if (minDimension < 1 || resizeShortSide < 1) {
    throw new Error("Cannot preprocess a zero-sized image");
  }
  // Resize(short side=256) followed by CenterCrop(224) is equivalent to
  // drawing this centred source rectangle directly into the target canvas.
  const cropWidth = (targetWidth * minDimension) / resizeShortSide;
  const cropHeight = (targetHeight * minDimension) / resizeShortSide;
  const sourceX = (sourceWidth - cropWidth) / 2;
  const sourceY = (sourceHeight - cropHeight) / 2;
  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create a 2D canvas context for image preprocessing");
  }
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(
    source,
    sourceX,
    sourceY,
    cropWidth,
    cropHeight,
    0,
    0,
    targetWidth,
    targetHeight
  );

  let imageData: ImageData;
  try {
    imageData = context.getImageData(0, 0, targetWidth, targetHeight);
  } catch {
    throw new Error(
      "Could not read image pixels (the canvas may be tainted by a cross-origin image)"
    );
  }

  const pixels = imageData.data;
  const totalPixels = targetWidth * targetHeight;
  const tensor = new Float32Array(3 * totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const alpha = pixels[i * 4 + 3] / 255;
    // Composite transparent pixels on black, then apply the shared scale.
    const r = pixels[i * 4] * alpha * CLASSIFIER_PREPROCESSING.scale;
    const g = pixels[i * 4 + 1] * alpha * CLASSIFIER_PREPROCESSING.scale;
    const b = pixels[i * 4 + 2] * alpha * CLASSIFIER_PREPROCESSING.scale;
    tensor[i] = (r - mean[0]) / std[0];
    tensor[i + totalPixels] = (g - mean[1]) / std[1];
    tensor[i + totalPixels * 2] = (b - mean[2]) / std[2];
  }
  return tensor;
}

export function createPreviewUrl(file: File): string {
  return URL.createObjectURL(file);
}

export function revokePreviewUrl(url: string | null): void {
  if (url) URL.revokeObjectURL(url);
}

export function imageDataToCanvas(
  imageData: ImageData,
  width: number,
  height: number
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (context) context.putImageData(imageData, 0, 0);
  return canvas;
}