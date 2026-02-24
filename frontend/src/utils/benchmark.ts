import * as ort from "onnxruntime-web";

import { MODEL_CONFIGS, GENERATOR_CONFIG, type ModelType } from "../constants";

/**
 * Benchmark statistics
 */
export interface BenchmarkStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  p50: number;
  p95: number;
  p99: number;
  samples: number;
}

/**
 * Classification benchmark result
 */
export interface ClassificationBenchmarkResult {
  modelType: ModelType;
  imageSize: number;
  stats: BenchmarkStats;
  timestamp: number;
}

/**
 * Generation benchmark result
 */
export interface GenerationBenchmarkResult {
  steps: number;
  cfgScale: number;
  imageSize: number;
  stats: BenchmarkStats;
  totalStats: BenchmarkStats;
  timestamp: number;
}

/**
 * Complete benchmark report
 */
export interface BenchmarkReport {
  classification: ClassificationBenchmarkResult[];
  generation: GenerationBenchmarkResult[];
  summary: {
    classificationLatency: BenchmarkStats;
    generationLatency: BenchmarkStats;
    meetsGoal: boolean;
    goalThreshold: number;
  };
  timestamp: number;
  userAgent: string;
  hardwareConcurrency: number;
}

/**
 * Calculate statistics from an array of latency measurements
 */
export function calculateStats(latencies: number[]): BenchmarkStats {
  if (latencies.length === 0) {
    return {
      mean: 0,
      std: 0,
      min: 0,
      max: 0,
      p50: 0,
      p95: 0,
      p99: 0,
      samples: 0,
    };
  }

  const sorted = [...latencies].sort((a, b) => a - b);
  const n = sorted.length;

  // Mean
  const mean = sorted.reduce((a, b) => a + b, 0) / n;

  // Standard deviation
  const variance = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  // Min and Max
  const min = sorted[0];
  const max = sorted[n - 1];

  // Percentiles
  const percentile = (p: number): number => {
    const index = Math.ceil((p / 100) * n) - 1;
    return sorted[Math.max(0, index)];
  };

  return {
    mean,
    std,
    min,
    max,
    p50: percentile(50),
    p95: percentile(95),
    p99: percentile(99),
    samples: n,
  };
}

/**
 * Generate synthetic image tensor for benchmarking
 */
export function generateSyntheticTensor(width: number, height: number): Float32Array {
  const totalPixels = width * height;
  const tensor = new Float32Array(3 * totalPixels);

  const configs = MODEL_CONFIGS.cats;
  const { mean, std } = configs;

  for (let i = 0; i < totalPixels; i++) {
    // Generate random values in normalized range
    const r = Math.random();
    const g = Math.random();
    const b = Math.random();

    tensor[i] = (r - mean[0]) / std[0];
    tensor[i + totalPixels] = (g - mean[1]) / std[1];
    tensor[i + totalPixels * 2] = (b - mean[2]) / std[2];
  }

  return tensor;
}

/**
 * Benchmark classification inference
 */
export async function benchmarkClassification(
  session: ort.InferenceSession,
  numRuns: number = 10,
  imageSize: number = 224
): Promise<BenchmarkStats> {
  const latencies: number[] = [];
  const inputDims = [1, 3, imageSize, imageSize];

  // Warmup run
  const warmupTensor = generateSyntheticTensor(imageSize, imageSize);
  const warmupInput = new ort.Tensor("float32", warmupTensor, inputDims);
  await session.run({ input: warmupInput });

  // Benchmark runs
  for (let i = 0; i < numRuns; i++) {
    const imageTensor = generateSyntheticTensor(imageSize, imageSize);
    const inputTensor = new ort.Tensor("float32", imageTensor, inputDims);

    const startTime = performance.now();
    await session.run({ input: inputTensor });
    const endTime = performance.now();

    latencies.push(endTime - startTime);
  }

  return calculateStats(latencies);
}

/**
 * Benchmark generation inference for a single step
 */
export async function benchmarkGenerationStep(
  session: ort.InferenceSession,
  numRuns: number = 5,
  imageSize: number = 128
): Promise<BenchmarkStats> {
  const latencies: number[] = [];
  const inputSize = 3 * imageSize * imageSize;

  // Warmup run
  const warmupNoise = new Float32Array(inputSize);
  const warmupNoiseTensor = new ort.Tensor("float32", warmupNoise, [1, 3, imageSize, imageSize]);
  const warmupTimestep = new ort.Tensor("float32", new Float32Array([0.5]), [1]);
  const warmupBreed = new ort.Tensor("int64", new BigInt64Array([BigInt(0)]), [1]);

  await session.run({
    noise: warmupNoiseTensor,
    timestep: warmupTimestep,
    breed: warmupBreed,
  });

  // Benchmark runs
  for (let i = 0; i < numRuns; i++) {
    const noise = new Float32Array(inputSize);
    for (let j = 0; j < inputSize; j++) {
      noise[j] = (Math.random() - 0.5) * 2;
    }

    const noiseTensor = new ort.Tensor("float32", noise, [1, 3, imageSize, imageSize]);
    const timestepTensor = new ort.Tensor("float32", new Float32Array([0.5]), [1]);
    const breedTensor = new ort.Tensor("int64", new BigInt64Array([BigInt(0)]), [1]);

    const startTime = performance.now();
    await session.run({
      noise: noiseTensor,
      timestep: timestepTensor,
      breed: breedTensor,
    });
    const endTime = performance.now();

    latencies.push(endTime - startTime);
  }

  return calculateStats(latencies);
}

/**
 * Benchmark full generation with CFG
 */
export async function benchmarkFullGeneration(
  session: ort.InferenceSession,
  steps: number,
  cfgScale: number,
  breedIndex: number,
  imageSize: number = 128
): Promise<{ stepStats: BenchmarkStats; totalStats: BenchmarkStats }> {
  const stepLatencies: number[] = [];
  const totalLatencies: number[] = [];
  const inputSize = 3 * imageSize * imageSize;
  const dt = 1.0 / steps;

  // Warmup
  const warmupNoise = new Float32Array(inputSize);
  const warmupNoiseTensor = new ort.Tensor("float32", warmupNoise, [1, 3, imageSize, imageSize]);
  const warmupTimestep = new ort.Tensor("float32", new Float32Array([0]), [1]);
  const warmupBreed = new ort.Tensor("int64", new BigInt64Array([BigInt(breedIndex)]), [1]);
  const warmupUncondBreed = new ort.Tensor("int64", new BigInt64Array([BigInt(-1)]), [1]);

  await session.run({ noise: warmupNoiseTensor, timestep: warmupTimestep, breed: warmupBreed });
  if (cfgScale > 1.0) {
    await session.run({ noise: warmupNoiseTensor, timestep: warmupTimestep, breed: warmupUncondBreed });
  }

  // Benchmark runs (3 full generations)
  const numGenerations = 3;

  for (let gen = 0; gen < numGenerations; gen++) {
    let x = new Float32Array(inputSize);
    for (let i = 0; i < inputSize; i++) {
      x[i] = (Math.random() - 0.5) * 2;
    }

    const genStartTime = performance.now();

    for (let step = 0; step < steps; step++) {
      const t = step / steps;

      const noiseTensor = new ort.Tensor("float32", x, [1, 3, imageSize, imageSize]);
      const timestepTensor = new ort.Tensor("float32", new Float32Array([t]), [1]);
      const breedTensor = new ort.Tensor("int64", new BigInt64Array([BigInt(breedIndex)]), [1]);

      const stepStartTime = performance.now();

      const result = await session.run({
        noise: noiseTensor,
        timestep: timestepTensor,
        breed: breedTensor,
      });

      let velocity = (result.velocity as ort.Tensor).data as Float32Array;

      // Apply CFG if scale > 1
      if (cfgScale > 1.0) {
        const uncondBreedTensor = new ort.Tensor("int64", new BigInt64Array([BigInt(-1)]), [1]);
        const uncondResult = await session.run({
          noise: noiseTensor,
          timestep: timestepTensor,
          breed: uncondBreedTensor,
        });
        const uncondVelocity = (uncondResult.velocity as ort.Tensor).data as Float32Array;

        const guidedVelocity = new Float32Array(inputSize);
        for (let i = 0; i < inputSize; i++) {
          guidedVelocity[i] = uncondVelocity[i] + cfgScale * (velocity[i] - uncondVelocity[i]);
        }
        velocity = guidedVelocity;
      }

      // Euler step
      for (let i = 0; i < inputSize; i++) {
        x[i] += dt * velocity[i];
      }

      const stepEndTime = performance.now();
      stepLatencies.push(stepEndTime - stepStartTime);
    }

    const genEndTime = performance.now();
    totalLatencies.push(genEndTime - genStartTime);
  }

  return {
    stepStats: calculateStats(stepLatencies),
    totalStats: calculateStats(totalLatencies),
  };
}

/**
 * Load model for benchmarking
 */
export async function loadModelForBenchmark(modelPath: string): Promise<ort.InferenceSession> {
  const response = await fetch(modelPath, { mode: "cors", credentials: "omit" });
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status}`);
  }
  const modelData = await response.arrayBuffer();
  const modelDataUint8 = new Uint8Array(modelData);

  return ort.InferenceSession.create(modelDataUint8, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

/**
 * Run complete benchmark suite
 */
export async function runBenchmarkSuite(
  onProgress?: (status: string) => void
): Promise<BenchmarkReport> {
  const report: BenchmarkReport = {
    classification: [],
    generation: [],
    summary: {
      classificationLatency: calculateStats([]),
      generationLatency: calculateStats([]),
      meetsGoal: false,
      goalThreshold: 2000, // 2 seconds as per GOAP success metric
    },
    timestamp: Date.now(),
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
  };

  const progress = (msg: string) => {
    console.log(`[Benchmark] ${msg}`);
    onProgress?.(msg);
  };

  try {
    // Benchmark classification
    progress("Loading classification model...");
    const classifierSession = await loadModelForBenchmark(MODEL_CONFIGS.cats.modelPath);
    progress("Benchmarking classification inference...");

    const imageSizes = [128, 224, 256];
    for (const size of imageSizes) {
      const stats = await benchmarkClassification(classifierSession, 10, size);
      report.classification.push({
        modelType: "cats",
        imageSize: size,
        stats,
        timestamp: Date.now(),
      });
      progress(`Classification (${size}x${size}): mean=${stats.mean.toFixed(2)}ms`);
    }

    // Benchmark generation
    progress("Loading generation model...");
    const generatorSession = await loadModelForBenchmark(GENERATOR_CONFIG.modelPath);
    progress("Benchmarking generation inference...");

    const stepsOptions = [10, 25, 50, 100];
    const cfgOptions = [1.0, 1.5, 2.0, 3.0];
    const breedIndex = 0; // Abyssinian

    for (const steps of stepsOptions) {
      for (const cfg of cfgOptions) {
        const result = await benchmarkFullGeneration(generatorSession, steps, cfg, breedIndex);
        report.generation.push({
          steps,
          cfgScale: cfg,
          imageSize: GENERATOR_CONFIG.imgDims[0],
          stats: result.stepStats,
          totalStats: result.totalStats,
          timestamp: Date.now(),
        });
        progress(`Generation (${steps} steps, CFG ${cfg}): total=${result.totalStats.mean.toFixed(2)}ms`);
      }
    }

    // Calculate summary
    const allClassificationLatencies = report.classification.flatMap(r =>
      Array(r.stats.samples).fill(r.stats.mean)
    );
    const allGenerationLatencies = report.generation.flatMap(r => r.totalStats.mean);

    report.summary.classificationLatency = calculateStats(allClassificationLatencies);
    report.summary.generationLatency = calculateStats(allGenerationLatencies);
    report.summary.meetsGoal = report.summary.generationLatency.p95 < report.summary.goalThreshold;

    progress("Benchmark complete!");
  } catch (error) {
    progress(`Benchmark failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    throw error;
  }

  return report;
}

/**
 * Format latency for display
 */
export function formatLatency(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${ms.toFixed(2)}ms`;
}

/**
 * Get performance recommendation based on benchmark results
 */
export function getPerformanceRecommendation(report: BenchmarkReport): string[] {
  const recommendations: string[] = [];
  const { summary } = report;

  if (summary.meetsGoal) {
    recommendations.push("Performance meets GOAP success metric (<2s for full generation)");
  } else {
    recommendations.push("Performance does not meet GOAP success metric (<2s for full generation)");
  }

  // Check classification latency
  if (summary.classificationLatency.mean > 500) {
    recommendations.push("Consider using smaller image sizes for faster classification");
  }

  // Check generation latency
  if (summary.generationLatency.mean > 5000) {
    recommendations.push("Consider reducing sampling steps or using WebGPU backend");
  }

  // Hardware recommendations
  if (report.hardwareConcurrency < 4) {
    recommendations.push("Performance may improve on systems with more CPU cores");
  }

  // CFG recommendations
  const highCfgResults = report.generation.filter(r => r.cfgScale >= 2.0);
  const lowCfgResults = report.generation.filter(r => r.cfgScale < 2.0);
  if (highCfgResults.length > 0 && lowCfgResults.length > 0) {
    const highCfgAvg = highCfgResults.reduce((sum, r) => sum + r.totalStats.mean, 0) / highCfgResults.length;
    const lowCfgAvg = lowCfgResults.reduce((sum, r) => sum + r.totalStats.mean, 0) / lowCfgResults.length;
    if (highCfgAvg > lowCfgAvg * 1.5) {
      recommendations.push("Using CFG scale < 2.0 significantly improves performance");
    }
  }

  // Steps recommendations
  const step10Results = report.generation.filter(r => r.steps === 10);
  const step50Results = report.generation.filter(r => r.steps === 50);
  if (step10Results.length > 0 && step50Results.length > 0) {
    const step10Avg = step10Results.reduce((sum, r) => sum + r.totalStats.mean, 0) / step10Results.length;
    const step50Avg = step50Results.reduce((sum, r) => sum + r.totalStats.mean, 0) / step50Results.length;
    if (step50Avg > step10Avg * 4) {
      recommendations.push("Consider using 10-25 steps for faster generation with acceptable quality");
    }
  }

  return recommendations;
}

/**
 * Export benchmark report to markdown
 */
export function exportReportToMarkdown(report: BenchmarkReport): string {
  const lines: string[] = [];

  lines.push("# Benchmark Results");
  lines.push("");
  lines.push(`Generated: ${new Date(report.timestamp).toISOString()}`);
  lines.push(`User Agent: ${report.userAgent}`);
  lines.push(`CPU Cores: ${report.hardwareConcurrency}`);
  lines.push("");

  lines.push("## Classification Latency");
  lines.push("");
  lines.push("| Image Size | Mean | Std | Min | Max | P50 | P95 | P99 |");
  lines.push("|------------|------|-----|-----|-----|-----|-----|-----|");
  for (const result of report.classification) {
    const s = result.stats;
    lines.push(
      `| ${result.imageSize}x${result.imageSize} | ${formatLatency(s.mean)} | ${formatLatency(s.std)} | ${formatLatency(s.min)} | ${formatLatency(s.max)} | ${formatLatency(s.p50)} | ${formatLatency(s.p95)} | ${formatLatency(s.p99)} |`
    );
  }
  lines.push("");

  lines.push("## Generation Latency");
  lines.push("");
  lines.push("| Steps | CFG Scale | Mean Total | P95 Total | Mean Step | P95 Step |");
  lines.push("|-------|-----------|------------|-----------|-----------|----------|");
  for (const result of report.generation) {
    lines.push(
      `| ${result.steps} | ${result.cfgScale.toFixed(1)} | ${formatLatency(result.totalStats.mean)} | ${formatLatency(result.totalStats.p95)} | ${formatLatency(result.stats.mean)} | ${formatLatency(result.stats.p95)} |`
    );
  }
  lines.push("");

  lines.push("## Summary");
  lines.push("");
  lines.push(`- **Classification Mean Latency**: ${formatLatency(report.summary.classificationLatency.mean)}`);
  lines.push(`- **Generation Mean Latency**: ${formatLatency(report.summary.generationLatency.mean)}`);
  lines.push(`- **Generation P95 Latency**: ${formatLatency(report.summary.generationLatency.p95)}`);
  lines.push(`- **GOAP Goal (<2s)**: ${report.summary.meetsGoal ? "PASSED" : "FAILED"}`);
  lines.push("");

  lines.push("## Recommendations");
  lines.push("");
  const recommendations = getPerformanceRecommendation(report);
  for (const rec of recommendations) {
    lines.push(`- ${rec}`);
  }
  lines.push("");

  return lines.join("\n");
}
