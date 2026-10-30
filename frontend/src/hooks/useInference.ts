import { useCallback, useEffect, useRef, useState } from "react";
import * as Comlink from "comlink";

import type { ModelType } from "../constants";
import type { ModelAssetOverride } from "../modelAssets";
import type {
  ClassificationTelemetry,
  InferenceEngineApi,
  ModelLoadInfo,
} from "../engine/inference.worker";

export type LoadStatus = "idle" | "loading" | "ready" | "error";

export interface InferenceState {
  status: LoadStatus;
  error: string | null;
  provider: ModelLoadInfo["provider"] | null;
  source: ModelLoadInfo["source"] | null;
  sha256: string | null;
}

const LOAD_TIMEOUT_MS = 60_000;

type TestAssetManifest = Partial<Record<string, ModelAssetOverride>>;

function testAssetOverride(modelType: ModelType): ModelAssetOverride | undefined {
  // Explicit E2E seam: Playwright installs this before the app boots so tests
  // can exercise digest mismatch/retry paths with deterministic bytes.
  const overrides = (
    globalThis as typeof globalThis & {
      __TINY_CATS_TEST_ASSETS__?: TestAssetManifest;
    }
  ).__TINY_CATS_TEST_ASSETS__;
  return overrides?.[modelType];
}

function describeError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "Unknown inference error";
}

export function useInference(modelPath: string, modelType: ModelType) {
  const [state, setState] = useState<InferenceState>({
    status: "loading",
    error: null,
    provider: null,
    source: null,
    sha256: null,
  });
  const [attempt, setAttempt] = useState(0);
  const engineRef = useRef<Comlink.Remote<InferenceEngineApi> | null>(null);

  useEffect(() => {
    let active = true;
    let settled = false;
    let disposing = false;
    setState({
      status: "loading",
      error: null,
      provider: null,
      source: null,
      sha256: null,
    });

    const worker = new Worker(
      new URL("../engine/inference.worker.ts", import.meta.url),
      { type: "module" }
    );
    const engine = Comlink.wrap<InferenceEngineApi>(worker);
    engineRef.current = engine;

    const disposeWorker = () => {
      if (disposing) return;
      disposing = true;
      const hardStop = window.setTimeout(() => worker.terminate(), 1000);
      void engine
        .dispose()
        .catch(() => undefined)
        .finally(() => {
          window.clearTimeout(hardStop);
          worker.terminate();
        });
    };

    const fail = (error: unknown) => {
      if (!active || settled) return;
      settled = true;
      if (engineRef.current === engine) engineRef.current = null;
      setState({
        status: "error",
        error: describeError(error),
        provider: null,
        source: null,
        sha256: null,
      });
      disposeWorker();
    };

    const onWorkerError = (event: ErrorEvent) => {
      event.preventDefault();
      fail(new Error(event.message || "Inference worker crashed"));
    };
    const onMessageError = () => {
      fail(new Error("Inference worker could not deserialize a message"));
    };
    worker.addEventListener("error", onWorkerError);
    worker.addEventListener("messageerror", onMessageError);

    const timeout = window.setTimeout(() => {
      fail(
        new Error(
          `Loading the classifier timed out after ${LOAD_TIMEOUT_MS / 1000}s. ` +
            "Check your connection and retry."
        )
      );
    }, LOAD_TIMEOUT_MS);

    engine
      .loadModel(modelType, testAssetOverride(modelType))
      .then((info) => {
        if (!active || settled) return;
        settled = true;
        window.clearTimeout(timeout);
        setState({
          status: "ready",
          error: null,
          provider: info.provider,
          source: info.source,
          sha256: info.sha256,
        });
      })
      .catch((error: unknown) => {
        if (!active || settled) return;
        window.clearTimeout(timeout);
        fail(error);
      });

    return () => {
      active = false;
      window.clearTimeout(timeout);
      worker.removeEventListener("error", onWorkerError);
      worker.removeEventListener("messageerror", onMessageError);
      engineRef.current = null;
      // Give session release a bounded opportunity before terminating.
      disposeWorker();
    };
    // modelPath participates because callers may swap artifacts without
    // changing modelType; attempt forces a clean retry.
  }, [modelPath, modelType, attempt]);

  const retry = useCallback(() => {
    setAttempt((value) => value + 1);
  }, []);

  const classify = useCallback(
    async (imageTensor: Float32Array): Promise<ClassificationTelemetry> => {
      if (!engineRef.current) {
        throw new Error("Classifier is not ready yet");
      }
      return engineRef.current.classify(imageTensor);
    },
    []
  );

  return { ...state, ready: state.status === "ready", retry, classify };
}