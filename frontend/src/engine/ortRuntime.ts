import * as ort from "onnxruntime-web";

export type ExecutionProvider = "webgpu" | "wasm";

export interface SessionResult {
  ort: typeof ort;
  session: ort.InferenceSession;
  provider: ExecutionProvider;
}

// Vite serves package files from this path in dev; static-copy preserves the
// same package-relative path in the production build.
const baseUrl = import.meta.env.BASE_URL || "/";
ort.env.wasm.wasmPaths = `${baseUrl}node_modules/onnxruntime-web/dist/`;
ort.env.wasm.numThreads =
  typeof navigator !== "undefined" && globalThis.crossOriginIsolated
    ? Math.max(1, navigator.hardwareConcurrency || 2)
    : 1;
ort.env.wasm.proxy = false;

export async function hasWebGpu(): Promise<boolean> {
  if (typeof navigator === "undefined") return false;
  const gpu = (
    navigator as Navigator & {
      gpu?: { requestAdapter: () => Promise<unknown | null> };
    }
  ).gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

async function loadOrt(provider: ExecutionProvider): Promise<typeof ort> {
  if (provider === "webgpu") {
    // The WebGPU build is a separate platform-specific entry point; importing
    // it statically would load its GPU code on WASM-only devices.
    const webgpu = await import("onnxruntime-web/webgpu");
    return webgpu as unknown as typeof ort;
  }
  return ort;
}

/**
 * Create an inference session, preferring WebGPU but falling back to WASM once
 * when adapter/session creation fails (issue #198).
 */
export async function createSessionWithFallback(
  data: Uint8Array,
  forceWebGpuFailure = false
): Promise<SessionResult> {
  const useWebGpu = forceWebGpuFailure || (await hasWebGpu());
  const providers: ExecutionProvider[] = useWebGpu
    ? ["webgpu", "wasm"]
    : ["wasm"];

  let lastError: unknown = null;
  for (const provider of providers) {
    try {
      if (provider === "webgpu" && forceWebGpuFailure) {
        throw new Error("Simulated WebGPU session creation failure");
      }
      const runtime = await loadOrt(provider);
      const session = await runtime.InferenceSession.create(data, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });
      return { ort: runtime, session, provider };
    } catch (error) {
      lastError = error;
      console.warn(`[ort] ${provider} session creation failed; trying fallback`, error);
    }
  }
  throw new Error(
    `Could not create an ONNX Runtime session (${providers.join(" -> ")}): ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`
  );
}

export function resolveInputName(
  session: ort.InferenceSession,
  expected: string
): string {
  const names = session.inputNames;
  if (names.includes(expected)) return expected;
  if (names.length === 1) return names[0];
  throw new Error(
    `ONNX graph inputs [${names.join(", ")}] do not include expected '${expected}'`
  );
}

export function resolveOutputName(
  session: ort.InferenceSession,
  expected: string
): string {
  const names = session.outputNames;
  if (names.includes(expected)) return expected;
  if (names.length === 1) return names[0];
  throw new Error(
    `ONNX graph outputs [${names.join(", ")}] do not include expected '${expected}'`
  );
}

export function stableSoftmax(logits: Float32Array | number[]): Float32Array {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  const probabilities = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp(logits[i] - max);
    probabilities[i] = value;
    sum += value;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new Error("Classifier produced non-finite logits");
  }
  for (let i = 0; i < probabilities.length; i++) probabilities[i] /= sum;
  return probabilities;
}