import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { MarketRegime, ScorerConfig, SupportedChain } from "./config.ts";
import type { Logger } from "./utils/logger.ts";
import { DeployerLookup } from "./features/deployer_lookup.ts";
import { SequenceBuffer } from "./features/sequence_buffer.ts";
import { computeTabularFeatures } from "./features/feature_schema.ts";
import {
  evaluateLpProtection,
  lpProtectionConfigFromScorer,
  summarizeLockerVaults,
  type RugCheckStatus
} from "./lp_protection_gate.ts";
import { OnnxRugScorer } from "./ml/onnx_scorer.ts";
import { shouldBlockRugRisk } from "./ml/uncertainty.ts";

export interface TokenLaunchEvent {
  mint: string;
  deployer: string;
  timestamp: number;
  chain?: SupportedChain;
  contractAddress?: string;
  poolAddress?: string;
  txSignature?: string;
  liquiditySol: number;
  previousLiquiditySol?: number;
  liquiditySpikePct?: number;
  lpBurnPct: number;
  ageSeconds: number;
  uniqueBuyers: number;
  totalVolumeSol: number;
  previousVolumeSol?: number;
  volumeSpikeRatio?: number;
  marketCapSol: number;
  rugPullRisk: number;
  honeypotRisk: number;
  transferTaxPct: number;
  topHolderPct: number;
  top10HolderPct?: number;
  devHoldPct: number;
  mutableMetadata: boolean;
  mintAuthorityRenounced: boolean;
  freezeAuthorityRenounced: boolean;
  ownerRenounced?: boolean;
  proxyContract?: boolean;
  blacklistFunction?: boolean;
  tradingPaused?: boolean;
  volatility1m: number;
  priceVelocity1m: number;
  buySellRatio: number;
  jitoCompetition: number;
  launchRatePerMinute: number;
  predictedWinProb: number;
  rewardRiskRatio: number;
  launchPlatform?: string;
  memeVolatilityIndex?: number;
  memeAlphaScore?: number;
  sentimentScore?: number;
  whaleAccumulationScore?: number;
  retailFomoScore?: number;
  botSpamScore?: number;
  volumeBottleneckRatio?: number;
  dataQuality?: "complete" | "incomplete";
  synthetic?: boolean;
  cachedRugcheck?: RugCheckSummary;
  lpLockedPct?: number;
  lpLockExpiryMs?: number;
  lpProtectionCachedAt?: number;
}

export interface TokenRiskResult {
  accepted: boolean;
  riskProbability: number;
  mlConfidence: number;
  regime: MarketRegime;
  reasons: string[];
  rugcheck?: RugCheckSummary;
  timeToRugHours?: number;
  maxDrawdownPct?: number;
  pump2xProbability?: number;
  uncertainty?: number;
}

export interface RugCheckSummary {
  mint?: string;
  score?: number;
  riskLevel?: string;
  mintAuthority?: string | null;
  freezeAuthority?: string | null;
  lpLocked?: boolean;
  lpLockedPct?: number;
  lpLockExpiryMs?: number;
  lpLockerTypes?: string[];
  topHoldersPct?: number;
  risks?: Array<{ name?: string; level?: string; description?: string }>;
}

export interface RugCheckFetchResult {
  status: RugCheckStatus;
  summary?: RugCheckSummary;
}

interface LinearModel {
  bias: number;
  weights: Record<string, number>;
  featureOrder?: string[];
}

export interface NeuralRiskPrediction {
  riskProbability: number;
  timeToRugHours: number;
  maxDrawdownPct: number;
  pump2xProbability: number;
  uncertainty: number;
}

const clamp = (value: number, low = 0, high = 1): number => Math.max(low, Math.min(high, value));
const sigmoid = (value: number): number => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
const emptyLinearModel = (): LinearModel => ({ bias: 0, weights: {} });

export class TokenRiskScorer {
  config: ScorerConfig;
  logger: Logger;
  model: LinearModel;
  onnxScorer?: OnnxRugScorer;
  sequenceBuffer: SequenceBuffer;
  deployerLookup?: DeployerLookup;

  constructor(
    config: ScorerConfig,
    logger: Logger,
    model: LinearModel,
    onnxScorer?: OnnxRugScorer,
    sequenceBuffer = new SequenceBuffer(),
    deployerLookup?: DeployerLookup
  ) {
    this.config = config;
    this.logger = logger.child({ component: "token_risk_scorer" });
    this.model = model;
    this.onnxScorer = onnxScorer;
    this.sequenceBuffer = sequenceBuffer;
    this.deployerLookup = deployerLookup;
  }

  static async load(modelPath: string, config: ScorerConfig, logger: Logger): Promise<TokenRiskScorer> {
    const resolvedPath = await resolveModelPath(modelPath);
    let linearModelPath = resolvedPath;
    if (resolvedPath.toLowerCase().endsWith(".onnx")) {
      const allowOnnx = ["1", "true", "yes", "on"].includes(
        (process.env.ALLOW_ONNX_RUG_MODEL || "").trim().toLowerCase()
      );
      if (!allowOnnx) {
        throw new Error(
          "ONNX rug model is disabled pending retraining on 2025+ data. " +
          "Set ALLOW_ONNX_RUG_MODEL=true to override."
        );
      }
      try {
        const deployerLookup = await DeployerLookup.load();
        const onnxScorer = await OnnxRugScorer.load(resolvedPath, { deployerLookup, mcPasses: 20 });
        logger.info({ modelPath: resolvedPath }, "onnx_rug_model_loaded");
        return new TokenRiskScorer(config, logger, emptyLinearModel(), onnxScorer, new SequenceBuffer(), deployerLookup);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        logger.warn({ modelPath: resolvedPath, error: message }, "onnx_model_unavailable_linear_fallback");
        linearModelPath = await resolveModelPath("rug_model.json");
      }
    }

    const raw = await readFile(linearModelPath, "utf8");
    const parsed = JSON.parse(raw) as LinearModel;
    if (typeof parsed.bias !== "number" || typeof parsed.weights !== "object") {
      throw new Error("invalid_rug_model");
    }
    logger.info({ modelPath: linearModelPath }, "rug_model_loaded");
    return new TokenRiskScorer(config, logger, parsed);
  }

  async evaluate(event: TokenLaunchEvent): Promise<TokenRiskResult> {
    const reasons: string[] = [];
    if (this.config.deployerBlacklist.has(event.deployer)) {
      reasons.push("blacklisted_deployer");
    }
    if (event.rugPullRisk > this.config.rugPullBlockThreshold) {
      reasons.push("rug_threshold");
    }

    const rugcheckFetch = await this.fetchRugCheck(event);
    const rugcheck = rugcheckFetch.summary;
    const lpGate = evaluateLpProtection(
      {
        lpBurnPct: event.lpBurnPct,
        rugcheck,
        rugcheckStatus: rugcheckFetch.status,
        synthetic: event.synthetic
      },
      lpProtectionConfigFromScorer(this.config)
    );
    if (!lpGate.accepted) {
      reasons.push(...lpGate.reasons);
    }
    if (event.honeypotRisk > this.config.honeypotRiskThreshold) {
      reasons.push("honeypot_risk");
    }
    if (event.transferTaxPct > this.config.maxTransferTaxPct) {
      reasons.push("transfer_tax");
    }

    if (rugcheck) {
      if (rugcheck.mintAuthority !== null && rugcheck.mintAuthority !== undefined) {
        reasons.push("rugcheck_mint_authority");
      }
      if (rugcheck.freezeAuthority !== null && rugcheck.freezeAuthority !== undefined) {
        reasons.push("rugcheck_freeze_authority");
      }
      if (rugcheck.risks?.some((risk) => ["danger", "critical"].includes(String(risk.level || "").toLowerCase()))) {
        reasons.push("rugcheck_danger");
      }
    }

    const prediction = await this.predictFull(event, rugcheck);
    const riskProbability = prediction.riskProbability;
    if (shouldBlockRugRisk({ riskProbability, uncertaintyStd: prediction.uncertainty, threshold: this.config.rugProbBlockThreshold })) {
      reasons.push("ml_risk_probability");
    }

    const regime = this.detectRegime(event, riskProbability);
    const mlConfidence = clamp(1 - riskProbability - prediction.uncertainty);
    return {
      accepted: reasons.length === 0,
      riskProbability,
      mlConfidence,
      regime,
      reasons,
      rugcheck,
      timeToRugHours: prediction.timeToRugHours,
      maxDrawdownPct: prediction.maxDrawdownPct,
      pump2xProbability: prediction.pump2xProbability,
      uncertainty: prediction.uncertainty
    };
  }

  async predict(event: TokenLaunchEvent, rugcheck?: RugCheckSummary): Promise<number> {
    return (await this.predictFull(event, rugcheck)).riskProbability;
  }

  async predictFull(event: TokenLaunchEvent, rugcheck?: RugCheckSummary): Promise<NeuralRiskPrediction> {
    const features = this.features(event, rugcheck);
    if (this.onnxScorer) {
      this.sequenceBuffer.updateFromEvent(event);
      const scored = await this.onnxScorer.score({
        features,
        deployerId: this.deployerLookup?.idFor(event.deployer),
        sequence: this.sequenceBuffer.sequenceFor(event.mint, event.timestamp),
        temporalEmbedding: await this.sequenceBuffer.embeddingFor(event.mint, event.timestamp)
      });
      return {
        riskProbability: clamp(scored.rugProb),
        timeToRugHours: scored.timeToRug,
        maxDrawdownPct: scored.maxDrawdown,
        pump2xProbability: clamp(scored.pump2xProb),
        uncertainty: scored.uncertainty
      };
    }
    let score = this.model.bias;
    for (const [key, weight] of Object.entries(this.model.weights)) {
      score += (features[key] || 0) * weight;
    }
    return {
      riskProbability: clamp(sigmoid(score)),
      timeToRugHours: 24,
      maxDrawdownPct: 0,
      pump2xProbability: 0,
      uncertainty: 0
    };
  }

  features(event: TokenLaunchEvent, rugcheck?: RugCheckSummary): Record<string, number> {
    return computeTabularFeatures({ event, rugcheck });
  }

  detectRegime(event: TokenLaunchEvent, riskProbability: number): MarketRegime {
    if (event.rugPullRisk > 0.08 || riskProbability > 0.1 || event.lpBurnPct < 0.95 || event.volatility1m > 0.72) {
      return "stress";
    }
    if (event.launchRatePerMinute > 1200 || event.jitoCompetition > 0.82) {
      return "burst";
    }
    if (event.volatility1m > 0.48 || event.buySellRatio < 0.82) {
      return "caution";
    }
    return "normal";
  }

  async fetchRugCheck(event: TokenLaunchEvent): Promise<RugCheckFetchResult> {
    if (event.synthetic) {
      return { status: "synthetic" };
    }
    if (!this.config.rugcheckEnabled) {
      return { status: "disabled" };
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 900);
    try {
      const response = await fetch(`${this.config.rugcheckApiUrl.replace(/\/$/, "")}/v1/tokens/${event.mint}/report/summary`, {
        headers: this.config.rugcheckApiKey ? { authorization: `Bearer ${this.config.rugcheckApiKey}` } : {},
        signal: controller.signal
      });
      if (!response.ok) {
        this.logger.warn({ mint: event.mint, status: response.status }, "rugcheck_request_failed");
        return { status: "failed" };
      }
      const summary = await response.json() as RugCheckSummary;
      const burnConfirmed = Number.isFinite(event.lpBurnPct) && event.lpBurnPct >= this.config.minLpBurnPct;
      if (!burnConfirmed) {
        const lockers = await this.fetchLockerVaults(event.mint, controller.signal);
        if (lockers) {
          summary.lpLockExpiryMs = lockers.lpLockExpiryMs;
          summary.lpLockerTypes = lockers.lpLockerTypes;
        }
      }
      return { status: "ok", summary };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.warn({ mint: event.mint, error: message }, "rugcheck_unavailable");
      return { status: "failed" };
    } finally {
      clearTimeout(timeout);
    }
  }

  private async fetchLockerVaults(
    mint: string,
    signal?: AbortSignal
  ): Promise<{ lpLockExpiryMs?: number; lpLockerTypes: string[] } | undefined> {
    try {
      const response = await fetch(`${this.config.rugcheckApiUrl.replace(/\/$/, "")}/v1/tokens/${mint}/lockers`, {
        headers: this.config.rugcheckApiKey ? { authorization: `Bearer ${this.config.rugcheckApiKey}` } : {},
        signal
      });
      if (!response.ok) {
        this.logger.warn({ mint, status: response.status }, "rugcheck_lockers_failed");
        return undefined;
      }
      const payload = await response.json() as { lockers?: Record<string, { unlockDate?: number | string; type?: string }> | Array<{ unlockDate?: number | string; type?: string }> };
      return summarizeLockerVaults(payload.lockers);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.warn({ mint, error: message }, "rugcheck_lockers_unavailable");
      return undefined;
    }
  }
}

const resolveModelPath = async (modelPath: string): Promise<string> => {
  const candidates = [
    resolve(modelPath),
    resolve(`${modelPath}.onnx`),
    resolve(`${modelPath}.json`),
    resolve("models", modelPath),
    resolve("models", `${modelPath}.onnx`),
    resolve("models", `${modelPath}.json`)
  ];
  if (modelPath.toLowerCase().endsWith(".onnx")) {
    candidates.push(resolve("models", "rug_model.json"));
  }
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {}
  }
  return candidates[0];
};
