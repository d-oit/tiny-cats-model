import modelManifest from "./generated/modelManifest.json";

export type ModelKey = "classifier" | "generator";

export interface ModelContract {
  contract_version: string;
  task: string;
  architecture: { name: string; num_classes: number; input_size: number };
  input: {
    name: string;
    dtype: string;
    layout: string;
    shape: number[];
    dynamic_axes: Record<string, string>;
  };
  output: {
    name: string;
    dtype: string;
    shape: number[];
    semantics: string;
    labels: string[];
  };
  preprocessing: {
    version: string;
    color_mode: string;
    channel_order: string;
    tensor_layout: string;
    dtype: string;
    resize: { mode: string; short_side: number; interpolation: string };
    center_crop: { height: number; width: number };
    scale: number;
    mean: number[];
    std: number[];
    exif_orientation: string;
    alpha: string;
  };
  dataset: Record<string, unknown>;
  opset?: number;
}

export interface ModelAsset {
  artifact: string;
  localPath: string;
  hubPath: string;
  hubRevision: string | null;
  sha256: string;
  size: number;
  contract: ModelContract;
  onnx?: {
    inputs: Array<{ name: string; dtype: string; shape: Array<number | string | null> }>;
    outputs: Array<{ name: string; dtype: string; shape: Array<number | string | null> }>;
    opset: number | null;
  };
}

export interface ModelAssetOverride {
  sha256?: string;
  size?: number;
  localPath?: string;
  hubPath?: string;
  hubRevision?: string | null;
  /** Test seam for exercising provider fallback without a WebGPU device. */
  forceWebGpuFailure?: boolean;
}

const HUB_REVISION = /^[0-9a-f]{40}$/;

export function getModelAsset(key: ModelKey): ModelAsset {
  // The manifest is generated and validated by scripts/generate_frontend_metadata.py;
  // classifier and generator contract shapes differ, so narrow here once.
  return (modelManifest.models as unknown as Record<ModelKey, ModelAsset>)[key];
}

export function mergeAsset(
  asset: ModelAsset,
  override?: ModelAssetOverride | null
): ModelAsset {
  if (!override) return asset;
  return {
    ...asset,
    ...override,
    contract: asset.contract,
  };
}

/** Resolve a bundle-relative path under the Vite base (GitHub Pages-safe). */
export function resolveLocalUrl(path: string): string {
  const base = import.meta.env.BASE_URL || "/";
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  const normalizedPath = path.startsWith("/") ? path.slice(1) : path;
  return `${normalizedBase}${normalizedPath}`;
}

/**
 * Immutable Hub URL for an asset, or null when no full commit SHA is pinned.
 * The frontend never constructs a mutable `resolve/main` URL.
 */
export function hubUrl(asset: ModelAsset): string | null {
  const revision = asset.hubRevision;
  if (!revision || !HUB_REVISION.test(revision)) return null;
  const base = modelManifest.hub.baseUrl.endsWith("/")
    ? modelManifest.hub.baseUrl
    : `${modelManifest.hub.baseUrl}/`;
  const path = asset.hubPath.startsWith("/") ? asset.hubPath.slice(1) : asset.hubPath;
  return `${base}${revision}/${path}`;
}

export async function sha256Hex(data: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto is unavailable; cannot verify model integrity");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export interface VerifiedModel {
  data: ArrayBuffer;
  source: "hub" | "local";
  sha256: string;
  size: number;
}

async function fetchAndVerify(
  url: string,
  asset: ModelAsset,
  source: "hub" | "local"
): Promise<VerifiedModel> {
  const response = await fetch(url, {
    mode: "cors",
    credentials: "omit",
    cache: source === "hub" ? "no-store" : "default",
  });
  if (!response.ok) {
    throw new Error(`${source} model request failed (${response.status})`);
  }
  const data = await response.arrayBuffer();
  if (asset.size > 0 && data.byteLength !== asset.size) {
    throw new Error(
      `${source} model size mismatch: expected ${asset.size}, got ${data.byteLength}`
    );
  }
  const digest = await sha256Hex(data);
  if (digest !== asset.sha256.toLowerCase()) {
    throw new Error(
      `${source} model digest mismatch: expected ${asset.sha256}, got ${digest}`
    );
  }
  return { data, source, sha256: digest, size: data.byteLength };
}

/**
 * Fetch a model and verify its bytes before any ONNX session is created.
 *
 * Order: immutable Hub revision (when pinned), then bundled local artifact.
 * A Hub failure (network or digest) may fall back to local, but the local
 * bytes must match the release manifest exactly; a stale local model is a hard
 * error. A digest mismatch never silently runs the wrong artifact.
 */
export async function fetchVerifiedModel(
  asset: ModelAsset,
  signal?: AbortSignal
): Promise<VerifiedModel> {
  const errors: string[] = [];
  const remote = hubUrl(asset);
  if (remote) {
    try {
      return await fetchAndVerify(remote, asset, "hub");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`hub: ${message}`);
      console.warn(`[models] immutable Hub fetch failed: ${message}`);
    }
  } else {
    errors.push("hub: no immutable revision pinned");
  }

  if (signal?.aborted) throw new Error("Model download aborted");
  const local = resolveLocalUrl(asset.localPath);
  try {
    return await fetchAndVerify(local, asset, "local");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    errors.push(`local: ${message}`);
  }
  throw new Error(
    `Could not load a verified model artifact (${asset.artifact}): ${errors.join("; ")}`
  );
}

export function modelManifestInfo(): {
  revision: string | null;
  immutable: boolean;
  repoId: string;
} {
  return {
    revision: modelManifest.hub.revision,
    immutable: modelManifest.hub.immutable,
    repoId: modelManifest.hub.repoId,
  };
}

export { modelManifest };