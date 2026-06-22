import type { MarketRegime, RiskConfig } from "./config.ts";
import { shouldBlockRugRisk } from "./ml/uncertainty.ts";

export interface RiskSignal {
  mint: string;
  timestamp: number;
  regime: MarketRegime;
  riskProbability: number;
  rugUncertaintyStd?: number;
  mlConfidence: number;
  winProbability: number;
  rewardRiskRatio: number;
  liquiditySol: number;
  volatility: number;
  deployer?: string;
  launchPlatform?: string;
  sector?: string;
  memeVolatilityIndex?: number;
  memeAlphaScore?: number;
  whaleAccumulationScore?: number;
  retailFomoScore?: number;
  volumeBottleneckRatio?: number;
}

export interface PositionPlan {
  accepted: boolean;
  reason: string;
  amountSol: number;
  positionFraction: number;
  stopLossPct: number;
  takeProfitPct: number;
  riskMode: MarketRegime;
}

export interface PositionRecord {
  mint: string;
  amountSol: number;
  openedAt: number;
  riskMode: MarketRegime;
  cluster?: string;
  platform?: string;
  sector?: string;
  deployer?: string;
}

export interface RiskSnapshot {
  equitySol: number;
  cashSol: number;
  openExposureSol: number;
  clusterExposureSol: Record<string, number>;
  maxDrawdownPct: number;
  dailyDrawdownPct: number;
  circuitBreakerOpen: boolean;
  circuitReason: string;
  consecutiveLosses: number;
  openPositions: number;
  compoundingReserveSol: number;
}

const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));

export class RiskManager {
  config: RiskConfig;
  cashSol: number;
  realizedPnlSol = 0;
  peakEquitySol: number;
  dayPeakEquitySol: number;
  maxDrawdownPct = 0;
  dailyDrawdownPct = 0;
  dailyPnlSol = 0;
  circuitBreakerOpen = false;
  circuitReason = "";
  consecutiveLosses = 0;
  compoundingReserveSol = 0;
  openPositions = new Map<string, PositionRecord>();
  rollingReturns: number[] = [];

  constructor(config: RiskConfig) {
    this.config = config;
    this.cashSol = config.startingCapitalSol;
    this.peakEquitySol = config.startingCapitalSol;
    this.dayPeakEquitySol = config.startingCapitalSol;
  }

  planPosition(signal: RiskSignal): PositionPlan {
    this.updateCircuitBreakers(signal.volatility);
    if (this.circuitBreakerOpen) {
      return this.rejected(this.circuitReason || "circuit_breaker_open", signal.regime);
    }
    if (this.openPositions.size >= this.config.maxOpenPositions) {
      return this.rejected("max_open_positions", signal.regime);
    }
    if (this.openPositions.has(signal.mint)) {
      return this.rejected("duplicate_mint_position", signal.regime);
    }
    const uncertaintyStd = signal.rugUncertaintyStd ?? 0;
    if (shouldBlockRugRisk({ riskProbability: signal.riskProbability, uncertaintyStd, threshold: this.config.rugProbBlockThreshold })) {
      return this.rejected("risk_probability_block", signal.regime);
    }
    if (signal.volatility >= this.config.volatilitySpikeBlock) {
      return this.rejected("volatility_spike_block", signal.regime);
    }

    const equity = this.equitySol();
    const cluster = this.clusterFor(signal.launchPlatform);
    const platform = this.platformFor(signal.launchPlatform);
    const exposureAfterMinimum = this.openExposureSol() + this.config.minTradeSol;
    if (equity <= 0 || exposureAfterMinimum / equity > this.config.maxTotalExposureFraction) {
      return this.rejected("total_exposure_cap", signal.regime);
    }
    const clusterExposureAfterMinimum = this.clusterExposureSol(cluster) + this.config.minTradeSol;
    if (equity <= 0 || clusterExposureAfterMinimum / equity > this.config.maxClusterExposureFraction) {
      return this.rejected("cluster_exposure_cap", signal.regime);
    }
    const platformLimit = this.config.maxPositionPerPlatform[platform] ?? this.config.maxPositionPerPlatform.other ?? 0.2;
    if (equity <= 0 || (this.platformExposureSol(platform) + this.config.minTradeSol) / equity > platformLimit) {
      return this.rejected("platform_exposure_cap", signal.regime);
    }
    if (signal.deployer && this.deployerPositionCount(signal.deployer) >= this.config.maxDeployerPositions) {
      return this.rejected("deployer_concentration_cap", signal.regime);
    }

    const kelly = this.kelly(signal.winProbability, signal.rewardRiskRatio);
    if (kelly <= 0) {
      return this.rejected("negative_kelly", signal.regime);
    }
    const regimeFactor = this.regimeFactor(signal.regime);
    const confidence = clamp(signal.mlConfidence, 0, 1);
    const clusterPenalty = this.clusterPenalty(cluster);
    const uncertaintyPenalty = uncertaintyStd > 0.08 ? 0.5 : 1;
    const tailRiskFactor = (signal.memeVolatilityIndex ?? signal.volatility) > this.config.tailRiskVolatilityThreshold ? 0.5 : 1;
    const alphaBoost = this.alphaBoost(signal);
    const compoundingFactor = this.compoundingFactor(signal, equity);
    const positionFraction = clamp(kelly * this.config.kellyFraction * regimeFactor * confidence * clusterPenalty * uncertaintyPenalty * tailRiskFactor * alphaBoost * compoundingFactor, 0, this.config.maxPositionFraction);
    const maxByExposure = Math.max(0, equity * this.config.maxTotalExposureFraction - this.openExposureSol());
    const maxByCluster = Math.max(0, equity * this.config.maxClusterExposureFraction - this.clusterExposureSol(cluster));
    const maxByPlatform = Math.max(0, equity * platformLimit - this.platformExposureSol(platform));
    const maxByCapital = equity * this.config.maxPositionFraction;
    const liquidityFraction = (signal.volumeBottleneckRatio ?? 0) >= 0.55
      ? this.config.volumeBottleneckLiquidityFraction
      : this.config.maxLiquidityPositionFraction;
    const maxByLiquidity = Math.max(this.config.minTradeSol, signal.liquiditySol * liquidityFraction);
    const amountSol = Math.min(equity * positionFraction, maxByExposure, maxByCluster, maxByPlatform, maxByCapital, maxByLiquidity, this.cashSol);

    if (amountSol < this.config.minTradeSol) {
      return this.rejected("below_min_trade_size", signal.regime);
    }

    return {
      accepted: true,
      reason: "accepted",
      amountSol,
      positionFraction: amountSol / Math.max(equity, 1e-9),
      stopLossPct: this.dynamicStop(signal.regime, signal.volatility),
      takeProfitPct: this.dynamicTakeProfit(signal.regime, signal.volatility),
      riskMode: signal.regime
    };
  }

  recordEntry(position: PositionRecord): void {
    this.cashSol -= position.amountSol;
    this.openPositions.set(position.mint, position);
    this.refreshDrawdown();
  }

  recordExit(mint: string, pnlSol: number): void {
    const position = this.openPositions.get(mint);
    if (!position) {
      return;
    }
    this.openPositions.delete(mint);
    this.cashSol += position.amountSol + pnlSol;
    this.realizedPnlSol += pnlSol;
    this.dailyPnlSol += pnlSol;
    if (pnlSol > 0) {
      const cap = this.equitySol() * this.config.compoundingMaxReserveFraction;
      this.compoundingReserveSol = Math.min(cap, this.compoundingReserveSol + pnlSol * this.config.compoundingReinvestWinPct);
    } else if (pnlSol < 0) {
      this.compoundingReserveSol = Math.max(0, this.compoundingReserveSol + pnlSol);
    }
    this.rollingReturns.push(pnlSol / Math.max(position.amountSol, 1e-9));
    if (this.rollingReturns.length > 120) {
      this.rollingReturns.shift();
    }
    this.consecutiveLosses = pnlSol < 0 ? this.consecutiveLosses + 1 : 0;
    this.refreshDrawdown();
    this.updateCircuitBreakers(this.realizedVolatility());
  }

  snapshot(): RiskSnapshot {
    return {
      equitySol: this.equitySol(),
      cashSol: this.cashSol,
      openExposureSol: this.openExposureSol(),
      clusterExposureSol: this.clusterExposureByName(),
      maxDrawdownPct: this.maxDrawdownPct,
      dailyDrawdownPct: this.dailyDrawdownPct,
      circuitBreakerOpen: this.circuitBreakerOpen,
      circuitReason: this.circuitReason,
      consecutiveLosses: this.consecutiveLosses,
      openPositions: this.openPositions.size,
      compoundingReserveSol: this.compoundingReserveSol
    };
  }

  equitySol(): number {
    return this.cashSol + this.openExposureSol();
  }

  openExposureSol(): number {
    let total = 0;
    for (const position of this.openPositions.values()) {
      total += position.amountSol;
    }
    return total;
  }

  clusterExposureSol(cluster: string): number {
    let total = 0;
    for (const position of this.openPositions.values()) {
      if (this.clusterFor(position.cluster) === cluster) {
        total += position.amountSol;
      }
    }
    return total;
  }

  clusterExposureByName(): Record<string, number> {
    const output: Record<string, number> = {};
    for (const position of this.openPositions.values()) {
      const cluster = this.clusterFor(position.cluster);
      output[cluster] = (output[cluster] || 0) + position.amountSol;
    }
    return output;
  }

  platformExposureSol(platform: string): number {
    let total = 0;
    for (const position of this.openPositions.values()) {
      if (this.platformFor(position.platform || position.cluster) === platform) {
        total += position.amountSol;
      }
    }
    return total;
  }

  deployerPositionCount(deployer: string): number {
    let count = 0;
    for (const position of this.openPositions.values()) {
      if (position.deployer === deployer) {
        count += 1;
      }
    }
    return count;
  }

  realizedVolatility(): number {
    if (this.rollingReturns.length < 2) {
      return 0;
    }
    const mean = this.rollingReturns.reduce((sum, value) => sum + value, 0) / this.rollingReturns.length;
    const variance = this.rollingReturns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / this.rollingReturns.length;
    return Math.sqrt(variance);
  }

  resetDailyPeak(): void {
    this.dayPeakEquitySol = this.equitySol();
    this.dailyDrawdownPct = 0;
    this.dailyPnlSol = 0;
  }

  updateCircuitBreakers(volatility: number): void {
    this.refreshDrawdown();
    if (this.maxDrawdownPct >= this.config.maxDrawdownCircuitBreakerPct) {
      this.openCircuit("max_drawdown");
    } else if (this.dailyDrawdownPct >= this.config.dailyDrawdownCircuitBreakerPct) {
      this.openCircuit("daily_drawdown");
    } else if (this.dailyPnlSol <= -Math.abs(this.config.dailyLossLimitSol)) {
      this.openCircuit("daily_loss_limit");
    } else if (volatility >= this.config.volatilitySpikeBlock) {
      this.openCircuit("volatility_spike");
    } else if (this.consecutiveLosses >= this.config.consecutiveLossCircuitBreaker) {
      this.openCircuit("consecutive_losses");
    }
  }

  openCircuit(reason: string): void {
    this.circuitBreakerOpen = true;
    this.circuitReason = reason;
  }

  closeCircuit(): void {
    this.circuitBreakerOpen = false;
    this.circuitReason = "";
    this.consecutiveLosses = 0;
    this.resetDailyPeak();
  }

  refreshDrawdown(): void {
    const equity = this.equitySol();
    this.peakEquitySol = Math.max(this.peakEquitySol, equity);
    this.dayPeakEquitySol = Math.max(this.dayPeakEquitySol, equity);
    this.maxDrawdownPct = this.peakEquitySol > 0 ? Math.max(this.maxDrawdownPct, ((this.peakEquitySol - equity) / this.peakEquitySol) * 100) : 100;
    this.dailyDrawdownPct = this.dayPeakEquitySol > 0 ? ((this.dayPeakEquitySol - equity) / this.dayPeakEquitySol) * 100 : 100;
  }

  kelly(winProbability: number, rewardRiskRatio: number): number {
    const p = clamp(winProbability, 0.01, 0.99);
    const b = Math.max(rewardRiskRatio, 0.1);
    const q = 1 - p;
    return clamp((b * p - q) / b, 0, 1);
  }

  regimeFactor(regime: MarketRegime): number {
    if (regime === "stress") {
      return 0.25;
    }
    if (regime === "caution") {
      return 0.5;
    }
    if (regime === "burst") {
      return 0.75;
    }
    return 0.6;
  }

  dynamicStop(regime: MarketRegime, volatility: number): number {
    const regimeMultiplier = regime === "stress" ? 0.55 : regime === "caution" ? 0.75 : 1;
    return clamp(this.config.baseStopLossPct * regimeMultiplier * (1 - Math.min(volatility, 0.6) * 0.35), 0.035, this.config.baseStopLossPct);
  }

  dynamicTakeProfit(regime: MarketRegime, volatility: number): number {
    const regimeMultiplier = regime === "burst" ? 1.24 : regime === "stress" ? 0.78 : 1;
    return clamp(this.config.baseTakeProfitPct * regimeMultiplier * (1 + Math.min(volatility, 0.5) * 0.2), 0.08, 0.42);
  }

  clusterPenalty(cluster: string): number {
    let count = 0;
    for (const position of this.openPositions.values()) {
      if (this.clusterFor(position.cluster) === cluster) {
        count += 1;
      }
    }
    if (count < this.config.correlationClusterPositionThreshold) {
      return 1;
    }
    return clamp(1 - (count - this.config.correlationClusterPositionThreshold + 1) * 0.25, 0.25, 1);
  }

  alphaBoost(signal: RiskSignal): number {
    const alpha = clamp(signal.memeAlphaScore ?? 0, 0, 1);
    const whale = clamp(signal.whaleAccumulationScore ?? 0, 0, 1);
    const retail = clamp(signal.retailFomoScore ?? 0, 0, 1);
    const fomoPenalty = clamp(retail - whale, 0, 1) * 0.45;
    return 1 + this.config.highAlphaKellyBoost * clamp((alpha - 0.55) / 0.45, 0, 1) * (1 - fomoPenalty);
  }

  compoundingFactor(signal: RiskSignal, equity: number): number {
    const alpha = signal.memeAlphaScore ?? 0;
    const bottleneck = signal.volumeBottleneckRatio ?? 0;
    if (alpha < 0.65 || bottleneck < 0.55 || equity <= 0 || this.compoundingReserveSol <= 0) {
      return 1;
    }
    const reserveBoost = Math.min(this.config.compoundingMaxBoost, this.compoundingReserveSol / equity);
    return 1 + reserveBoost * (0.5 + clamp(bottleneck, 0, 1) * 0.5);
  }

  clusterFor(value: string | undefined): string {
    const normalized = String(value || "unknown").toLowerCase();
    if (normalized.includes("pump")) {
      return "pump.fun";
    }
    if (normalized.includes("raydium")) {
      return "raydium";
    }
    if (normalized.includes("moonshot")) {
      return "moonshot";
    }
    return normalized || "unknown";
  }

  platformFor(value: string | undefined): string {
    const normalized = String(value || "other").toLowerCase();
    if (normalized.includes("pump")) {
      return "pump_fun";
    }
    if (normalized.includes("raydium")) {
      return "raydium";
    }
    return "other";
  }

  rejected(reason: string, regime: MarketRegime): PositionPlan {
    return {
      accepted: false,
      reason,
      amountSol: 0,
      positionFraction: 0,
      stopLossPct: 0,
      takeProfitPct: 0,
      riskMode: regime
    };
  }
}
