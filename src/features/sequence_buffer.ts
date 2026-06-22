import type { TokenLaunchEvent } from "../token_risk_scorer.ts";

export const SEQUENCE_LENGTH = 16;
export const TEMPORAL_EMBEDDING_DIM = 16;
export const SEQUENCE_FEATURES = [
  "holder_count",
  "liquidity_sol",
  "volume_sol",
  "buy_sell_ratio",
  "price_velocity",
  "tx_count"
] as const;

export type SequenceFeatureName = typeof SEQUENCE_FEATURES[number];
export type ModelSequence = number[][];

export interface TokenStateSnapshot {
  mint: string;
  timestamp: number;
  holderCount: number;
  liquiditySol: number;
  volumeSol: number;
  buySellRatio: number;
  priceVelocity: number;
  txCount: number;
}

interface SequenceSlot {
  hour: number;
  values: [number, number, number, number, number, number];
}

export interface SequenceWindow {
  sequence: ModelSequence;
  insufficientHistory: boolean;
}

type OrtModule = typeof import("onnxruntime-node");
type OrtSession = import("onnxruntime-node").InferenceSession;

const emptyStep = (): [number, number, number, number, number, number] => [0, 0, 0, 1, 0, 0];

export class SequenceBuffer {
  buffers = new Map<string, SequenceSlot[]>();
  maxTokens: number;
  encoderSession?: OrtSession;
  ort?: OrtModule;

  constructor(maxTokens = 50_000) {
    this.maxTokens = maxTokens;
  }

  async loadEncoder(path = "models/sequence_encoder.onnx"): Promise<void> {
    const ort = await import("onnxruntime-node");
    this.ort = ort;
    this.encoderSession = await ort.InferenceSession.create(path, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
      intraOpNumThreads: 1,
      interOpNumThreads: 1
    });
  }

  update(snapshot: TokenStateSnapshot): void {
    const hour = Math.floor(snapshot.timestamp / 3_600_000);
    const values: [number, number, number, number, number, number] = [
      Math.max(0, snapshot.holderCount),
      Math.max(0, snapshot.liquiditySol),
      Math.max(0, snapshot.volumeSol),
      Math.max(0, snapshot.buySellRatio),
      snapshot.priceVelocity,
      Math.max(0, snapshot.txCount)
    ];
    const buffer = this.buffers.get(snapshot.mint) || [];
    const existing = buffer.find((slot) => slot.hour === hour);
    if (existing) {
      existing.values = values;
    } else {
      buffer.push({ hour, values });
      buffer.sort((a, b) => a.hour - b.hour);
      while (buffer.length > SEQUENCE_LENGTH) {
        buffer.shift();
      }
    }
    this.buffers.set(snapshot.mint, buffer);
    this.evictIfNeeded();
  }

  updateFromEvent(event: TokenLaunchEvent, holderCount = event.uniqueBuyers): void {
    this.update({
      mint: event.mint,
      timestamp: event.timestamp,
      holderCount,
      liquiditySol: event.liquiditySol,
      volumeSol: event.totalVolumeSol,
      buySellRatio: event.buySellRatio,
      priceVelocity: event.priceVelocity1m,
      txCount: Math.max(0, Math.round(event.uniqueBuyers * Math.max(event.buySellRatio, 0.1)))
    });
  }

  windowFor(mint: string, now = Date.now()): SequenceWindow {
    const currentHour = Math.floor(now / 3_600_000);
    const buffer = this.buffers.get(mint) || [];
    const byHour = new Map(buffer.map((slot) => [slot.hour, slot.values] as const));
    const output: ModelSequence = [];
    for (let offset = SEQUENCE_LENGTH - 1; offset >= 0; offset -= 1) {
      output.push([...(byHour.get(currentHour - offset) || emptyStep())]);
    }
    return {
      sequence: output,
      insufficientHistory: buffer.length < 6
    };
  }

  sequenceFor(mint: string, now = Date.now()): ModelSequence {
    return this.windowFor(mint, now).sequence;
  }

  async embeddingFor(mint: string, now = Date.now()): Promise<Float32Array> {
    const window = this.windowFor(mint, now);
    if (this.encoderSession && this.ort) {
      const feeds = {
        sequence: new this.ort.Tensor("float32", Float32Array.from(window.sequence.flat()), [1, SEQUENCE_LENGTH, SEQUENCE_FEATURES.length])
      };
      const outputs = await this.encoderSession.run(feeds);
      const tensor = outputs.temporal_embedding || outputs.embedding || Object.values(outputs)[0];
      if (tensor) {
        return Float32Array.from(Array.from(tensor.data as Float32Array | number[]).slice(0, TEMPORAL_EMBEDDING_DIM));
      }
    }
    return fallbackEmbedding(window.sequence);
  }

  flattenFor(mint: string, now = Date.now()): Float32Array {
    return Float32Array.from(this.sequenceFor(mint, now).flat());
  }

  insufficientHistory(mint: string): boolean {
    return (this.buffers.get(mint) || []).length < 6;
  }

  evict(mint: string): void {
    this.buffers.delete(mint);
  }

  private evictIfNeeded(): void {
    if (this.buffers.size <= this.maxTokens) {
      return;
    }
    const first = this.buffers.keys().next().value as string | undefined;
    if (first) {
      this.buffers.delete(first);
    }
  }
}

const fallbackEmbedding = (sequence: ModelSequence): Float32Array => {
  const columns = SEQUENCE_FEATURES.map((_, column) => sequence.map((row) => finite(row[column], column === 3 ? 1 : 0)));
  const values: number[] = [];
  for (const column of columns) {
    const mean = column.reduce((sum, value) => sum + value, 0) / Math.max(column.length, 1);
    const last = column[column.length - 1] || 0;
    values.push(mean, last);
  }
  const velocity = columns[4] || [];
  const txCount = columns[5] || [];
  values.push(
    Math.max(...velocity, 0),
    Math.min(...velocity, 0),
    txCount.reduce((sum, value) => sum + value, 0),
    sequence.filter((row) => row.some((value) => value !== 0)).length / SEQUENCE_LENGTH
  );
  return Float32Array.from(values.slice(0, TEMPORAL_EMBEDDING_DIM));
};

const finite = (value: number | undefined, fallback: number): number => Number.isFinite(value) ? Number(value) : fallback;
