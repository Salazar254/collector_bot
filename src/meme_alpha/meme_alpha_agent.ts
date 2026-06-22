import type { MemeAlphaConfig } from "../config.ts";
import type { LpProtectionConfig } from "../lp_protection_gate.ts";
import type { TokenLaunchEvent } from "../token_risk_scorer.ts";
import type { Logger } from "../utils/logger.ts";
import { AntiRugGuard, type AntiRugAuditResult } from "./anti_rug_guard.ts";
import { LiquiditySpikeDetector, type LiquiditySpikeSignal } from "./liquidity_spike_detector.ts";
import { SocialSentimentBook, firstKey, type SentimentScore, type SentimentSummary, type SocialPost } from "./sentiment_engine.ts";

export interface MemeAlphaDecision {
  accepted: boolean;
  score: number;
  confidence: "low" | "medium" | "high";
  reasons: string[];
  enrichedEvent: TokenLaunchEvent;
  audit: AntiRugAuditResult;
  liquidity: LiquiditySpikeSignal;
  sentiment: SentimentSummary;
}

const clamp01 = (value: number): number => Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

export class MemeAlphaAgent {
  config: MemeAlphaConfig;
  logger: Logger;
  antiRug: AntiRugGuard;
  liquidity: LiquiditySpikeDetector;
  sentiment: SocialSentimentBook;

  constructor(config: MemeAlphaConfig, logger: Logger, lpProtection: LpProtectionConfig) {
    this.config = config;
    this.logger = logger.child({ component: "meme_alpha_agent" });
    this.antiRug = new AntiRugGuard(config, lpProtection);
    this.liquidity = new LiquiditySpikeDetector(config);
    this.sentiment = new SocialSentimentBook({
      halfLifeMs: config.sentimentHalfLifeMs,
      staleMs: config.sentimentStaleMs
    });
  }

  ingestSocialPost(post: SocialPost): SentimentScore | undefined {
    const score = this.sentiment.ingest(post);
    if (score) {
      this.logger.debug({
        key: firstKey(post.mint, post.contractAddress, post.symbol),
        source: post.source,
        sentimentScore: score.sentimentScore,
        whaleAccumulationScore: score.whaleAccumulationScore,
        retailFomoScore: score.retailFomoScore,
        botSpamScore: score.botSpamScore
      }, "social_signal_ingested");
    }
    return score;
  }

  async evaluate(event: TokenLaunchEvent): Promise<MemeAlphaDecision> {
    const audit = this.antiRug.audit(event);
    const liquidity = this.liquidity.detect(event);
    const sentiment = this.sentiment.summaryFor([
      event.mint,
      event.contractAddress,
      event.poolAddress
    ], event.timestamp);
    const score = this.alphaScore(liquidity, sentiment, audit);
    const reasons = [
      ...audit.reasons,
      ...liquidity.reasons,
      ...sentiment.reasons.map((reason) => `sentiment_${reason}`)
    ];
    if (sentiment.samples === 0) {
      reasons.push("social_context_missing");
    }
    if (score < this.config.minScore) {
      reasons.push("meme_alpha_below_threshold");
    }
    if (!liquidity.accepted) {
      reasons.push("liquidity_spike_unverified");
    }
    const accepted = audit.accepted && liquidity.accepted && score >= this.config.minScore;
    const enrichedEvent: TokenLaunchEvent = {
      ...event,
      liquiditySpikePct: liquidity.spikePct,
      volumeSpikeRatio: liquidity.volumeSpikeRatio,
      memeAlphaScore: score,
      sentimentScore: sentiment.sentimentScore,
      whaleAccumulationScore: sentiment.whaleAccumulationScore,
      retailFomoScore: sentiment.retailFomoScore,
      botSpamScore: sentiment.botSpamScore,
      volumeBottleneckRatio: liquidity.volumeBottleneckRatio,
      memeVolatilityIndex: Math.max(event.memeVolatilityIndex ?? 0, liquidity.volumeBottleneckRatio, event.volatility1m)
    };
    return {
      accepted,
      score,
      confidence: score >= this.config.highConvictionScore ? "high" : score >= this.config.minScore ? "medium" : "low",
      reasons: dedupe(reasons),
      enrichedEvent,
      audit,
      liquidity,
      sentiment
    };
  }

  alphaScore(liquidity: LiquiditySpikeSignal, sentiment: SentimentSummary, audit: AntiRugAuditResult): number {
    const social = sentiment.samples > 0 ? sentiment.sentimentScore : 0;
    const whale = sentiment.samples > 0 ? sentiment.whaleAccumulationScore : 0;
    const retailPenalty = clamp01(sentiment.retailFomoScore - sentiment.whaleAccumulationScore) * 0.12;
    const spamPenalty = sentiment.botSpamScore * 0.18;
    const auditPenalty = audit.riskScore * 0.42;
    return clamp01(
      liquidity.score * 0.48
      + social * 0.2
      + whale * 0.18
      + liquidity.volumeBottleneckRatio * 0.14
      - retailPenalty
      - spamPenalty
      - auditPenalty
    );
  }
}

const dedupe = (items: string[]): string[] => [...new Set(items.filter(Boolean))];
