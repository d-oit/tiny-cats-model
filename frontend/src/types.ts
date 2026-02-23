export interface InferenceTelemetry {
  step: number;
  totalSteps: number;
  image: Float32Array;
  sketch: Float32Array;
  stepTime: number;
  totalTime: number;
}

export interface ClassificationResult {
  classIndex: number;
  className: string;
  confidence: number;
}

export interface ClassificationTelemetry {
  image: Float32Array;
  result: ClassificationResult;
  probabilities: number[];
}
