import type { BotConfig } from "./config.ts";
import {
  evaluateLpProtection,
  lpProtectionConfigFromScorer
} from "./lp_protection_gate.ts";
import { checkMevProtection } from "./mev_guard.ts";
import type { PositionManager } from "./position_manager.ts";
import type { EntryPriceFeed } from "./position_backends.ts";
import { checkJupiterSlippage } from "./slippage_guard.ts";
import type { TokenLaunchEvent, TokenRiskResult, TokenRiskScorer } from "./token_risk_scorer.ts";
import { RiskManager } from "./risk_manager.ts";
import type { Logger } from "./utils/logger.ts";
import { JitoClient } from "./utils/jito_client.ts";
import type { RpcPool } from "./utils/rpc_pool.ts";
import { WalletRotator, type WalletRef } from "./wallet_rotator.ts";
import { MemeAlphaAgent } from "./meme_alpha/meme_alpha_agent.ts";
import type { SocialPost } from "./meme_alpha/sentiment_engine.ts";

const SOL_MINT = "So11111111111111111111111111111111111111112";
const LAMPORTS_PER_SOL = 1_000_000_000n;

export interface SwapBundleRequest {
  event: TokenLaunchEvent;
  wallet: WalletRef;
  amountSol: number;
  stopLossPct: number;
  takeProfitPct: number;
  tipAccount?: string;
}

export interface SwapBundle {
  transactionsBase64: string[];
  quoteId: string;
  expectedOutAmount?: string;
}

export type SwapBundleBuilder = (request: SwapBundleRequest) => Promise<SwapBundle>;

export interface TradeReceipt {
  mint: string;
  mode: string;
  walletId: string;
  amountSol: number;
  accepted: boolean;
  reason: string;
  pnlSol?: number;
  bundleId?: string;
}

export interface EngineStats {
  received: number;
  processed: number;
  accepted: number;
  rejected: number;
  executed: number;
  failed: number;
  staleEventsDropped: number;
  lpRevalidationRejected: number;
  noFreeWallet: number;
}

export interface SniperEngineOptions {
  buildSwapBundle?: SwapBundleBuilder;
  memeAlpha?: MemeAlphaAgent;
  rpc?: RpcPool;
  positionManager?: PositionManager;
  priceFeed?: EntryPriceFeed;
}

export class SniperEngine {
  config: BotConfig;
  scorer: TokenRiskScorer;
  risk: RiskManager;
  wallets: WalletRotator;
  jito: JitoClient;
  rpc?: RpcPool;
  positionManager?: PositionManager;
  priceFeed?: EntryPriceFeed;
  memeAlpha?: MemeAlphaAgent;
  logger: Logger;
  buildSwapBundle?: SwapBundleBuilder;
  queue: TokenLaunchEvent[] = [];
  running = false;
  stats: EngineStats = {
    received: 0,
    processed: 0,
    accepted: 0,
    rejected: 0,
    executed: 0,
    failed: 0,
    staleEventsDropped: 0,
    lpRevalidationRejected: 0,
    noFreeWallet: 0
  };

  constructor(config: BotConfig, scorer: TokenRiskScorer, logger: Logger, options: SniperEngineOptions = {}) {
    this.config = config;
    this.scorer = scorer;
    this.risk = new RiskManager(config.risk);
    this.wallets = new WalletRotator(config.wallets, logger);
    this.jito = new JitoClient(config.jito, logger);
    this.rpc = options.rpc;
    this.positionManager = options.positionManager;
    this.priceFeed = options.priceFeed;
    this.memeAlpha = options.memeAlpha ?? (config.memeAlpha.enabled
      ? new MemeAlphaAgent(config.memeAlpha, logger, lpProtectionConfigFromScorer(config.scorer))
      : undefined);
    this.logger = logger.child({ component: "sniper_engine" });
    this.buildSwapBundle = options.buildSwapBundle;
  }

  attachPositionManager(positionManager: PositionManager, priceFeed?: EntryPriceFeed): void {
    this.positionManager = positionManager;
    if (priceFeed) {
      this.priceFeed = priceFeed;
    }
  }

  async start(): Promise<void> {
    if (this.config.mode === "live" && !this.config.liveTrading) {
      throw new Error("live_mode_requires_LIVE_TRADING_true");
    }
    await this.wallets.start();
    this.running = true;
    this.pump();
    this.logger.info({ mode: this.config.mode, queueDepth: this.queue.length }, "engine_started");
  }

  async stop(): Promise<void> {
    this.running = false;
    await this.wallets.stop();
    this.logger.info({ stats: this.stats, risk: this.risk.snapshot() }, "engine_stopped");
  }

  submit(event: TokenLaunchEvent): boolean {
    this.stats.received += 1;
    if (this.queue.length >= this.config.throughput.maxQueueDepth) {
      this.stats.rejected += 1;
      this.logger.warn({ mint: event.mint, queueDepth: this.queue.length }, "event_queue_full");
      return false;
    }
    this.queue.push(event);
    return true;
  }

  ingestSocialPost(post: SocialPost): void {
    this.memeAlpha?.ingestSocialPost(post);
  }

  async drain(): Promise<void> {
    while (this.queue.length > 0) {
      await this.processBatch();
    }
  }

  async processBatch(): Promise<void> {
    const batch = this.queue.splice(0, this.config.throughput.eventLoopBatchSize);
    await Promise.all(batch.map((event) => this.processEvent(event)));
  }

  pump(): void {
    if (!this.running) {
      return;
    }
    void this.processBatch().catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error({ error: message }, "batch_processing_failed");
    }).finally(() => {
      if (this.running) {
        setImmediate(() => this.pump());
      }
    });
  }

  eventAgeMs(event: TokenLaunchEvent, nowMs = Date.now()): number {
    return nowMs - event.timestamp;
  }

  isEventStale(event: TokenLaunchEvent, nowMs = Date.now()): boolean {
    return this.eventAgeMs(event, nowMs) > this.config.throughput.maxEventAgeMs;
  }

  isLpCacheStale(event: TokenLaunchEvent, nowMs = Date.now()): boolean {
    if (!event.lpProtectionCachedAt) {
      return true;
    }
    return nowMs - event.lpProtectionCachedAt > this.config.throughput.maxEventAgeMs;
  }

  attachLpProtectionCache(event: TokenLaunchEvent, riskResult: TokenRiskResult): TokenLaunchEvent {
    const rugcheck = riskResult.rugcheck;
    return {
      ...event,
      cachedRugcheck: rugcheck,
      lpLockedPct: event.lpLockedPct ?? rugcheck?.lpLockedPct,
      lpLockExpiryMs: event.lpLockExpiryMs ?? rugcheck?.lpLockExpiryMs,
      lpProtectionCachedAt: Date.now()
    };
  }

  revalidateLpProtection(event: TokenLaunchEvent): { accepted: boolean; reasons: string[] } {
    if (this.isLpCacheStale(event)) {
      return { accepted: false, reasons: ["lp_cache_stale"] };
    }
    const rugcheckStatus = event.cachedRugcheck ? "ok" as const : event.synthetic ? "synthetic" as const : "disabled" as const;
    const result = evaluateLpProtection(
      {
        lpBurnPct: event.lpBurnPct,
        rugcheck: event.cachedRugcheck,
        rugcheckStatus,
        synthetic: event.synthetic
      },
      lpProtectionConfigFromScorer(this.config.scorer)
    );
    return { accepted: result.accepted, reasons: result.reasons };
  }

  estimateEntryPriceSol(event: TokenLaunchEvent): number {
    if (event.liquiditySol > 0 && event.marketCapSol > 0) {
      return event.marketCapSol / event.liquiditySol;
    }
    return 1;
  }

  estimateExpectedOutAmount(event: TokenLaunchEvent, amountSol: number): bigint {
    const entryPriceSol = this.estimateEntryPriceSol(event);
    if (entryPriceSol <= 0) {
      return 0n;
    }
    return BigInt(Math.max(1, Math.floor(amountSol / entryPriceSol * 1_000_000_000)));
  }

  async processEvent(event: TokenLaunchEvent): Promise<TradeReceipt | undefined> {
    this.stats.processed += 1;
    let wallet: WalletRef | null = null;
    try {
      if (this.isEventStale(event)) {
        this.stats.staleEventsDropped += 1;
        this.stats.rejected += 1;
        this.logger.debug({
          mint: event.mint,
          ageMs: this.eventAgeMs(event),
          maxEventAgeMs: this.config.throughput.maxEventAgeMs
        }, "event_stale_dropped");
        return undefined;
      }

      let candidate = event;
      if (this.memeAlpha) {
        const alpha = await this.memeAlpha.evaluate(event);
        if (!alpha.accepted) {
          this.stats.rejected += 1;
          this.logger.debug({ mint: event.mint, score: alpha.score, reasons: alpha.reasons }, "meme_alpha_rejected");
          return undefined;
        }
        candidate = alpha.enrichedEvent;
        this.logger.debug({
          mint: candidate.mint,
          score: alpha.score,
          confidence: alpha.confidence,
          auditMs: alpha.audit.elapsedMs,
          sentimentSamples: alpha.sentiment.samples
        }, "meme_alpha_accepted");
      }

      const riskResult = await this.scorer.evaluate(candidate);
      if (!riskResult.accepted) {
        this.stats.rejected += 1;
        this.logger.debug({ mint: candidate.mint, reasons: riskResult.reasons, riskProbability: riskResult.riskProbability }, "token_rejected");
        return undefined;
      }

      candidate = this.attachLpProtectionCache(candidate, riskResult);

      const plan = this.risk.planPosition({
        mint: candidate.mint,
        timestamp: candidate.timestamp,
        regime: riskResult.regime,
        riskProbability: riskResult.riskProbability,
        mlConfidence: riskResult.mlConfidence,
        winProbability: candidate.predictedWinProb,
        rewardRiskRatio: candidate.rewardRiskRatio,
        liquiditySol: candidate.liquiditySol,
        volatility: candidate.volatility1m,
        deployer: candidate.deployer,
        launchPlatform: candidate.launchPlatform,
        rugUncertaintyStd: riskResult.uncertainty,
        memeVolatilityIndex: candidate.memeVolatilityIndex,
        memeAlphaScore: candidate.memeAlphaScore,
        whaleAccumulationScore: candidate.whaleAccumulationScore,
        retailFomoScore: candidate.retailFomoScore,
        volumeBottleneckRatio: candidate.volumeBottleneckRatio
      });

      if (!plan.accepted) {
        this.stats.rejected += 1;
        this.logger.debug({ mint: candidate.mint, reason: plan.reason }, "position_rejected");
        return undefined;
      }

      wallet = this.wallets.nextWallet();
      if (!wallet) {
        this.stats.noFreeWallet += 1;
        this.stats.rejected += 1;
        this.logger.warn({ mint: candidate.mint }, "no_free_wallet");
        return undefined;
      }

      const receipt = await this.execute(candidate, wallet, plan.amountSol, plan.stopLossPct, plan.takeProfitPct, riskResult.regime);
      if (!receipt.accepted) {
        this.stats.rejected += 1;
        return receipt;
      }

      this.stats.accepted += 1;
      await this.wallets.publishAllocation(wallet, candidate.mint, plan.amountSol);
      this.risk.recordEntry({
        mint: candidate.mint,
        amountSol: plan.amountSol,
        openedAt: Date.now(),
        riskMode: plan.riskMode,
        cluster: candidate.launchPlatform,
        platform: candidate.launchPlatform,
        deployer: candidate.deployer
      });
      await this.trackPosition(candidate, wallet, plan.amountSol, plan.stopLossPct, plan.takeProfitPct, riskResult.timeToRugHours);
      return receipt;
    } catch (error) {
      this.stats.failed += 1;
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error({ mint: event.mint, error: message }, "event_processing_failed");
      return undefined;
    } finally {
      if (wallet) {
        this.wallets.complete(wallet.id);
      }
    }
  }

  async trackPosition(
    event: TokenLaunchEvent,
    wallet: WalletRef,
    amountSol: number,
    stopLossPct: number,
    takeProfitPct: number,
    timeToRugHours?: number
  ): Promise<void> {
    if (!this.positionManager) {
      return;
    }
    const entryPriceSol = this.estimateEntryPriceSol(event);
    this.priceFeed?.track(event.mint, entryPriceSol);
    const tokenAmount = entryPriceSol > 0 ? amountSol / entryPriceSol : amountSol;
    await this.positionManager.track({
      mint: event.mint,
      walletId: wallet.id,
      amountSol,
      tokenAmount,
      entryPriceSol,
      openedAt: Date.now(),
      dynamicStop: stopLossPct,
      dynamicTakeProfit: takeProfitPct,
      maxHoldMs: this.config.scorer.maxHoldHorizonMs,
      timeToRugHours,
      cluster: event.launchPlatform
    });
  }

  async execute(
    event: TokenLaunchEvent,
    wallet: WalletRef,
    amountSol: number,
    stopLossPct: number,
    takeProfitPct: number,
    regime: TokenRiskResult["regime"] = "normal"
  ): Promise<TradeReceipt> {
    if (this.isEventStale(event)) {
      this.stats.staleEventsDropped += 1;
      this.logger.warn({
        mint: event.mint,
        ageMs: this.eventAgeMs(event),
        maxEventAgeMs: this.config.throughput.maxEventAgeMs
      }, "execute_event_stale_rejected");
      return {
        mint: event.mint,
        mode: this.config.mode,
        walletId: wallet.id,
        amountSol,
        accepted: false,
        reason: "event_stale"
      };
    }

    const lpRevalidation = this.revalidateLpProtection(event);
    if (!lpRevalidation.accepted) {
      this.stats.lpRevalidationRejected += 1;
      this.logger.warn({
        mint: event.mint,
        reasons: lpRevalidation.reasons,
        lpProtectionCachedAt: event.lpProtectionCachedAt
      }, "execute_lp_revalidation_rejected");
      return {
        mint: event.mint,
        mode: this.config.mode,
        walletId: wallet.id,
        amountSol,
        accepted: false,
        reason: lpRevalidation.reasons[0] || "lp_revalidation_failed"
      };
    }

    if (this.config.mode === "paper") {
      this.logger.debug({ mint: event.mint }, "paper_mode_guard_simulation_skipped");
      this.stats.executed += 1;
      return {
        mint: event.mint,
        mode: "paper",
        walletId: wallet.id,
        amountSol,
        accepted: true,
        reason: "paper_entry_recorded"
      };
    }

    if (!this.buildSwapBundle) {
      throw new Error("live_swap_bundle_builder_required");
    }
    if (!this.rpc) {
      throw new Error("live_rpc_pool_required");
    }

    const expectedOutAmount = this.estimateExpectedOutAmount(event, amountSol);
    const slippage = await checkJupiterSlippage({
      inputMint: SOL_MINT,
      outputMint: event.mint,
      amountLamports: BigInt(Math.floor(amountSol * Number(LAMPORTS_PER_SOL))),
      expectedOutAmount,
      maxDeviationPct: this.config.execution.maxSlippagePct,
      quoteApiBaseUrl: this.config.execution.jupiterQuoteApiUrl
    });
    if (!slippage.accepted) {
      this.logger.warn({ mint: event.mint, deviationPct: slippage.deviationPct }, "slippage_guard_rejected");
      return {
        mint: event.mint,
        mode: "live",
        walletId: wallet.id,
        amountSol,
        accepted: false,
        reason: "slippage_exceeded"
      };
    }

    const tipAccount = await this.jito.nextTipAccount();
    const bundle = await this.buildSwapBundle({
      event,
      wallet,
      amountSol,
      stopLossPct,
      takeProfitPct,
      tipAccount
    });

    const swapTx = bundle.transactionsBase64[0];
    if (!swapTx) {
      throw new Error("swap_bundle_missing_transaction");
    }

    const quotedOutAmount = bundle.expectedOutAmount
      ? BigInt(bundle.expectedOutAmount)
      : slippage.quotedOutAmount;
    const mev = await checkMevProtection({
      quotedOutAmount,
      transactionBase64: swapTx,
      regime,
      rpc: this.rpc
    });
    if (!mev.accepted) {
      this.logger.warn({ mint: event.mint, reason: mev.reason }, "mev_guard_rejected");
      return {
        mint: event.mint,
        mode: "live",
        walletId: wallet.id,
        amountSol,
        accepted: false,
        reason: "mev_risk_detected"
      };
    }

    const result = await this.jito.sendBundle(bundle.transactionsBase64, event.jitoCompetition);
    if (!result.accepted) {
      return {
        mint: event.mint,
        mode: "live",
        walletId: wallet.id,
        amountSol,
        accepted: false,
        reason: result.reason || "bundle_not_landed",
        bundleId: result.bundleId
      };
    }

    this.stats.executed += 1;
    return {
      mint: event.mint,
      mode: "live",
      walletId: wallet.id,
      amountSol,
      accepted: true,
      reason: "jito_bundle_landed",
      bundleId: result.bundleId
    };
  }
}
