import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { DeployerLookup } from "../features/deployer_lookup.ts";
import { TABULAR_FEATURES } from "../features/feature_schema.ts";
import { SEQUENCE_LENGTH, TEMPORAL_EMBEDDING_DIM, type ModelSequence } from "../features/sequence_buffer.ts";
import { clamp, summarizeMultiTask, type MultiTaskUncertainty } from "./uncertainty.ts";

export const ONNX_FEATURE_ORDER = TABULAR_FEATURES;

export interface OnnxScoreInput {
  features: Record<string, number>;
  tabularFeatures?: number[];
  deployerId?: number;
  sequence?: ModelSequence;
  temporalEmbedding?: Float32Array | number[];
}

export interface OnnxScoreResult {
  rugProb: number;
  xgbRugProb: number | null;
  torchRugProb: number;
  riskProbability: number;
  timeToRug: number;
  maxDrawdown: number;
  pump2xProb: number;
  uncertainty: number;
  highUncertainty: boolean;
  distributions: MultiTaskUncertainty;
  elapsedMs: number;
}

export interface OnnxScorerConfig {
  xgbModelPath?: string;
  xgbEnsembleWeight?: number;
  torchEnsembleWeight?: number;
}

interface OnnxMetadata {
  feature_names?: string[];
}

type OrtModule = typeof import("onnxruntime-node");
type OrtSession = import("onnxruntime-node").InferenceSession;

const sessionCache = new Map<string, Promise<OrtSession>>();
let xgbFallbackLogged = false;

export class OnnxRugScorer {
  modelPath: string;
  session: OrtSession;
  ort: OrtModule;
  featureOrder: string[];
  deployerLookup?: DeployerLookup;
  mcPasses: number;
  private xgbSession: OrtSession | null = null;
  private xgbEnsembleWeight: number;
  private torchEnsembleWeight: number;

  private constructor(
    modelPath: string,
    session: OrtSession,
    ort: OrtModule,
    featureOrder: string[],
    mcPasses: number,
    config: OnnxScorerConfig,
    deployerLookup?: DeployerLookup,
    xgbSession: OrtSession | null = null
  ) {
    this.modelPath = modelPath;
    this.session = session;
    this.ort = ort;
    this.featureOrder = featureOrder;
    this.mcPasses = mcPasses;
    this.deployerLookup = deployerLookup;
    this.xgbSession = xgbSession;
    this.xgbEnsembleWeight = config.xgbEnsembleWeight ?? 0.4;
    this.torchEnsembleWeight = config.torchEnsembleWeight ?? 0.6;
  }

  static async load(modelPath: string, options: {
    metadataPath?: string;
    mcPasses?: number;
    deployerLookup?: DeployerLookup;
    xgbModelPath?: string;
    xgbEnsembleWeight?: number;
    torchEnsembleWeight?: number;
    logger?: { info: (obj: object, msg: string) => void; warn: (obj: object, msg: string) => void };
  } = {}): Promise<OnnxRugScorer> {
    const ort = await import("onnxruntime-node");
    const resolved = resolve(modelPath);
    const sessionOptions = {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all" as const,
      intraOpNumThreads: 1,
      interOpNumThreads: 1
    };
    let sessionPromise = sessionCache.get(resolved);
    if (!sessionPromise) {
      sessionPromise = ort.InferenceSession.create(resolved, sessionOptions);
      sessionCache.set(resolved, sessionPromise);
    }
    const session = await sessionPromise;
    const metadataPath = options.metadataPath || resolved.replace(/\.onnx$/i, "_meta.json");
    const featureOrder = await loadFeatureOrder(metadataPath);

    let xgbSession: OrtSession | null = null;
    const xgbPath = options.xgbModelPath ? resolve(options.xgbModelPath) : undefined;
    if (xgbPath && existsSync(xgbPath)) {
      try {
        xgbSession = await ort.InferenceSession.create(xgbPath, sessionOptions);
        options.logger?.info({ xgbPath }, "xgb_model_loaded");
      } catch (error) {
        options.logger?.warn({ xgbPath, err: error }, "xgb_model_load_failed_pytorch_only");
      }
    } else if (options.xgbModelPath) {
      options.logger?.warn({ xgbPath: options.xgbModelPath }, "xgb_model_not_found_pytorch_only");
    }

    return new OnnxRugScorer(
      resolved,
      session,
      ort,
      featureOrder,
      options.mcPasses ?? 15,
      {
        xgbModelPath: options.xgbModelPath,
        xgbEnsembleWeight: options.xgbEnsembleWeight,
        torchEnsembleWeight: options.torchEnsembleWeight
      },
      options.deployerLookup,
      xgbSession
    );
  }

  async score(input: OnnxScoreInput, passes = this.mcPasses): Promise<OnnxScoreResult> {
    const started = performance.now();
    const tabular = this.resolveTabular(input);
    const xgbProb = await this.runXgb(tabular);
    const torchResult = await this.runTorch(input, passes);

    let ensembleRugProb: number;
    if (xgbProb !== null) {
      ensembleRugProb = (xgbProb * this.xgbEnsembleWeight) + (torchResult.rugProb * this.torchEnsembleWeight);
    } else {
      ensembleRugProb = torchResult.rugProb;
      if (!xgbFallbackLogged) {
        xgbFallbackLogged = true;
        console.warn("xgb_unavailable_pytorch_only_mode");
      }
    }

    return {
      ...torchResult,
      rugProb: ensembleRugProb,
      xgbRugProb: xgbProb,
      torchRugProb: torchResult.rugProb,
      riskProbability: ensembleRugProb,
      elapsedMs: performance.now() - started
    };
  }

  async scoreFast(input: OnnxScoreInput): Promise<OnnxScoreResult> {
    return this.score(input, 1);
  }

  private resolveTabular(input: OnnxScoreInput): number[] {
    if (input.tabularFeatures && input.tabularFeatures.length === this.featureOrder.length) {
      return input.tabularFeatures;
    }
    return this.featureOrder.map((name) => Number.isFinite(input.features[name]) ? input.features[name] : 0);
  }

  private async runXgb(tabular: number[]): Promise<number | null> {
    if (!this.xgbSession) {
      return null;
    }

    try {
      const input = new this.ort.Tensor("float32", Float32Array.from(tabular), [1, 14]);
      if (input.dims[1] !== 14) {
        throw new Error(`XGB tabular shape mismatch: expected [1,14] got [1,${input.dims[1]}]`);
      }

      const result = await this.xgbSession.run({ tabular_input: input });
      const outputName = (this.xgbSession as unknown as { outputNames: string[] }).outputNames.find((name: string) => name.includes("prob")) ?? (this.xgbSession as unknown as { outputNames: string[] }).outputNames[1];
      const output = result[outputName];
      if (!output) {
        return null;
      }

      const data = output.data as ArrayLike<number> | Record<string, number>;
      if (Array.isArray(data) || data instanceof Float32Array) {
        const values = Array.from(data);
        return clamp(values.length > 1 ? Number(values[1]) : Number(values[0] ?? 0));
      }
      if (typeof data === "object") {
        const map = data as Record<string, number>;
        if (map[1] !== undefined) {
          return clamp(Number(map[1]));
        }
        if (map["1"] !== undefined) {
          return clamp(Number(map["1"]));
        }
        const values = Object.values(map);
        return clamp(Number(values[1] ?? values[0] ?? 0));
      }
      return clamp(Number(data));
    } catch (err) {
      if (!xgbFallbackLogged) {
        xgbFallbackLogged = true;
        console.warn("xgb_inference_failed_falling_back_to_pytorch", err instanceof Error ? err.message : String(err));
      }
      return null;
    }
  }

  private async runTorch(input: OnnxScoreInput, passes: number): Promise<Omit<OnnxScoreResult, "xgbRugProb" | "torchRugProb" | "riskProbability" | "elapsedMs">> {
    const samples: Array<{ rugProb: number; timeToRug: number; maxDrawdown: number; pump2xProb: number }> = [];
    const runCount = Math.max(1, passes);
    for (let index = 0; index < runCount; index += 1) {
      const outputs = await this.session.run(this.feeds(input));
      samples.push({
        rugProb: clamp(readScalar(outputs.rug_prob)),
        timeToRug: Math.max(0, readScalar(outputs.time_to_rug_hours ?? outputs.time_to_rug)),
        maxDrawdown: clamp(readScalar(outputs.max_drawdown_pct ?? outputs.max_drawdown), 0, 1),
        pump2xProb: clamp(readScalar(outputs.pump_2x_prob ?? outputs.pump_2x))
      });
    }
    const distributions = summarizeMultiTask(samples);
    return {
      rugProb: distributions.rugProb.mean,
      timeToRug: distributions.timeToRug.mean,
      maxDrawdown: distributions.maxDrawdown.mean,
      pump2xProb: distributions.pump2xProb.mean,
      uncertainty: distributions.rugProb.std,
      highUncertainty: Object.values(distributions).some((summary) => summary.std > 0.08),
      distributions
    };
  }

  feeds(input: OnnxScoreInput): Record<string, import("onnxruntime-node").Tensor> {
    const featureValues = this.resolveTabular(input);
    const deployerId = BigInt(Math.max(0, input.deployerId ?? 0));
    const baseFeeds: Record<string, import("onnxruntime-node").Tensor> = {
      tabular: new this.ort.Tensor("float32", Float32Array.from(featureValues), [1, this.featureOrder.length]),
      deployer_id: new this.ort.Tensor("int64", BigInt64Array.from([deployerId]), [1])
    };
    if (this.inputNames().includes("temporal_embedding")) {
      baseFeeds.temporal_embedding = new this.ort.Tensor("float32", normalizeTemporalEmbedding(input.temporalEmbedding), [1, TEMPORAL_EMBEDDING_DIM]);
      return baseFeeds;
    }
    const sequenceWidth = this.sequenceWidth();
    const sequence = normalizeLegacySequence(input.sequence, sequenceWidth);
    baseFeeds.sequence = new this.ort.Tensor("float32", Float32Array.from(sequence.flat()), [1, SEQUENCE_LENGTH, sequenceWidth]);

    const [batchSize, seqLen, features] = baseFeeds.sequence.dims as number[];
    if (batchSize !== 1 || seqLen !== SEQUENCE_LENGTH || features !== sequenceWidth) {
      throw new Error(
        `sequence tensor shape mismatch: expected [1,${SEQUENCE_LENGTH},${sequenceWidth}] got [${batchSize},${seqLen},${features}]. ` +
        "Check SEQUENCE_LENGTH alignment between training and inference."
      );
    }

    return baseFeeds;
  }

  inputNames(): string[] {
    return (this.session as unknown as { inputNames?: string[] }).inputNames || [];
  }

  sequenceWidth(): number {
    const metadata = (this.session as unknown as { inputMetadata?: Record<string, { dimensions?: Array<number | string> }> }).inputMetadata;
    const width = metadata?.sequence?.dimensions?.[2];
    return typeof width === "number" && width > 0 ? width : 6;
  }
}

const normalizeLegacySequence = (sequence?: ModelSequence, width = 6): ModelSequence => {
  const empty = Array.from({ length: width }, (_, index) => index === 3 ? 1 : 0);
  const rows = (sequence || []).slice(-SEQUENCE_LENGTH).map((row) => [
    ...Array.from({ length: width }, (_, index) => finite(row[index], index === 3 ? 1 : 0))
  ]);
  while (rows.length < SEQUENCE_LENGTH) {
    rows.unshift([...empty]);
  }
  return rows;
};

const normalizeTemporalEmbedding = (embedding?: Float32Array | number[]): Float32Array => {
  const values = Array.from(embedding || []);
  while (values.length < TEMPORAL_EMBEDDING_DIM) {
    values.push(0);
  }
  return Float32Array.from(values.slice(0, TEMPORAL_EMBEDDING_DIM).map((value) => finite(value, 0)));
};

const finite = (value: number | undefined, fallback: number): number => Number.isFinite(value) ? Number(value) : fallback;

const readScalar = (tensor: import("onnxruntime-node").Tensor | undefined): number => {
  if (!tensor) {
    return 0;
  }
  const data = tensor.data as Float32Array | number[];
  return Number(data[0] ?? 0);
};

const loadFeatureOrder = async (metadataPath: string): Promise<string[]> => {
  try {
    const raw = await readFile(metadataPath, "utf8");
    const parsed = JSON.parse(raw) as OnnxMetadata;
    if (Array.isArray(parsed.feature_names) && parsed.feature_names.length > 0) {
      return parsed.feature_names;
    }
  } catch {
    return [...ONNX_FEATURE_ORDER];
  }
  return [...ONNX_FEATURE_ORDER];
};
