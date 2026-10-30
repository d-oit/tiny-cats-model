import { useCallback, useEffect, useRef, useState } from "react";
import * as Comlink from "comlink";

import type { ModelAssetOverride } from "../modelAssets";
import type {
  GenerationEngineApi,
  GenerationOptions,
  GenerationProgress,
  GenerationResult,
  GeneratorLoadInfo,
} from "../engine/generation.worker";
import type { LoadStatus } from "./useInference";

const LOAD_TIMEOUT_MS = 180_000;

interface GenerationState {
  status: LoadStatus;
  error: string | null;
  provider: GeneratorLoadInfo["provider"] | null;
  source: GeneratorLoadInfo["source"] | null;
  sha256: string | null;
}

function testAssetOverride(): ModelAssetOverride | undefined {
  const overrides = (
    globalThis as typeof globalThis & {
      __TINY_CATS_TEST_ASSETS__?: Partial<Record<string, ModelAssetOverride>>;
    }
  ).__TINY_CATS_TEST_ASSETS__;
  return overrides?.generator;
}

function describeError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "Unknown generation error";
}

export function useGeneration() {
  const [state, setState] = useState<GenerationState>({
    status: "loading",
    error: null,
    provider: null,
    source: null,
    sha256: null,
  });
  const [attempt, setAttempt] = useState(0);
  const engineRef = useRef<Comlink.Remote<GenerationEngineApi> | null>(null);

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
      new URL("../engine/generation.worker.ts", import.meta.url),
      { type: "module" }
    );
    const engine = Comlink.wrap<GenerationEngineApi>(worker);
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
      fail(new Error(event.message || "Generator worker crashed"));
    };
    const onMessageError = () => {
      fail(new Error("Generator worker could not deserialize a message"));
    };
    worker.addEventListener("error", onWorkerError);
    worker.addEventListener("messageerror", onMessageError);

    const timeout = window.setTimeout(() => {
      fail(
        new Error(
          `Loading the generator timed out after ${LOAD_TIMEOUT_MS / 1000}s. ` +
            "Check your connection and retry."
        )
      );
    }, LOAD_TIMEOUT_MS);

    engine
      .loadModel(testAssetOverride())
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
      disposeWorker();
    };
  }, [attempt]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  const generate = useCallback(
    async (
      options: GenerationOptions,
      onProgress?: (progress: GenerationProgress) => void
    ): Promise<GenerationResult> => {
      if (!engineRef.current) throw new Error("Generator is not ready yet");
      const callback = onProgress
        ? Comlink.proxy((progress: GenerationProgress) => onProgress(progress))
        : undefined;
      try {
        return await engineRef.current.generate(options, callback);
      } finally {
        try {
          (callback as unknown as Record<symbol, () => void> | undefined)?.[
            Comlink.releaseProxy
          ]?.();
        } catch {
          // Callback cleanup is best effort.
        }
      }
    },
    []
  );

  return { ...state, ready: state.status === "ready", retry, generate };
}