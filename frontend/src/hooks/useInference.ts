import { useState, useEffect, useRef } from "react";
import * as Comlink from "comlink";
import type { ClassificationTelemetry } from "../types";

export function useInference(modelPath: string, modelType: string) {
  const [ready, setReady] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const engineRef = useRef<any>(null);

  useEffect(() => {
    const worker = new Worker(
      new URL("../engine/inference.worker.ts", import.meta.url),
      { type: "module" }
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const engine: any = Comlink.wrap(worker);
    engineRef.current = engine;

    engine.loadModel(modelPath, modelType).then(() => setReady(true));

    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath]);

  const classify = async (
    imageTensor: Float32Array
  ): Promise<ClassificationTelemetry | null> => {
    if (!engineRef.current) return null;
    return await engineRef.current.classify(imageTensor);
  };

  return { ready, classify };
}
