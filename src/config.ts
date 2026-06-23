export type RuntimeMode = "paper" | "live";
export type MarketRegime = "normal" | "caution" | "stress" | "burst";
export type LogLevel = "debug" | "info" | "warn" | "error";

// Single source of truth for the rug-probability block threshold. Both the scorer
// and the risk manager read this same default so the two gates can never drift.
export const RUG_PROB_BLOCK_THRESHOLD_DEFAULT = 0.15;

export interface RpcConfig {
  httpUrls: string[];
  wsUrl?: string;
  requestTimeoutMs: number;
  maxRetries: number;
}

export type SupportedChain = "solana" | "base";

export interface JitoConfig {
  blockEngineUrl: string;
  authUuid?: string;
  minTipSol: number;
  maxTipSol: number;
  tipFloorUrl: string;
  bundleOnly: boolean;
  maxLandingPollAttempts: number;
  landingPollIntervalMs: number;
}

export interface ExecutionConfig {
  maxSlippagePct: number;
  jupiterQuoteApiUrl: string;
  positionPollIntervalMs: number;
}

export interface RiskConfig {
  startingCapitalSol: number;
  maxPositionFraction: number;
  kellyFraction: number;
  compoundingReinvestWinPct: number;
  compoundingMaxBoost: number;
  compoundingMaxReserveFraction: number;
  highAlphaKellyBoost: number;
  maxLiquidityPositionFraction: number;
  volumeBottleneckLiquidityFraction: number;
  maxTotalExposureFraction: number;
  maxClusterExposureFraction: number;
  maxPositionPerPlatform: Record<string, number>;
  correlationClusterPositionThreshold: number;
  tailRiskVolatilityThreshold: number;
  maxDrawdownCircuitBreakerPct: number;
  dailyDrawdownCircuitBreakerPct: number;
  dailyLossLimitSol: number;
  volatilitySpikeBlock: number;
  minTradeSol: number;
  maxOpenPositions: number;
  baseStopLossPct: number;
  baseTakeProfitPct: number;
  consecutiveLossCircuitBreaker: number;
  maxDeployerPositions: number;
  rugProbBlockThreshold: number;
}

export interface ScorerConfig {
  modelPath: string;
  xgbModelPath: string;
  xgbEnsembleWeight: number;
  torchEnsembleWeight: number;
  rugcheckEnabled: boolean;
  rugcheckApiUrl: string;
  rugcheckApiKey?: string;
  rugProbBlockThreshold: number;
  rugPullBlockThreshold: number;
  minLpBurnPct: number;
  minLpLockPct: number;
  maxHoldHorizonMs: number;
  honeypotRiskThreshold: number;
  maxTransferTaxPct: number;
  deployerBlacklist: Set<string>;
}

export interface WalletConfig {
  masterPublicKey?: string;
  satelliteWalletsJson?: string;
  rotationCount: number;
  redisUrl?: string;
}

export interface ThroughputConfig {
  targetEventsPerHour: number;
  targetTradesPerHour: number;
  streamFanout: number;
  eventLoopBatchSize: number;
  maxQueueDepth: number;
  maxEventAgeMs: number;
}

export interface MemeAlphaConfig {
  enabled: boolean;
  streamsEnabled: boolean;
  minScore: number;
  highConvictionScore: number;
  sentimentHalfLifeMs: number;
  sentimentStaleMs: number;
  socialWsUrls: string[];
  solanaWsUrl?: string;
  solanaLogMentions: string[];
  baseWsUrl?: string;
  baseLogAddresses: string[];
  baseLogTopics: string[];
  liquiditySpikePct: number;
  minLiquidityDeltaSol: number;
  volumeSpikeRatio: number;
  auditBudgetMs: number;
  blockOnAuditBudgetOverrun: boolean;
  requireMintAuthorityRenounced: boolean;
  requireFreezeAuthorityRenounced: boolean;
  requireSocialContext: boolean;
  maxTopHolderPct: number;
  maxTop10HolderPct: number;
  maxDevHoldPct: number;
}

export interface BotConfig {
  mode: RuntimeMode;
  liveTrading: boolean;
  logLevel: LogLevel;
  emergencyHaltPort: number;
  modelRegistryPath: string;
  rpc: RpcConfig;
  jito: JitoConfig;
  risk: RiskConfig;
  scorer: ScorerConfig;
  wallets: WalletConfig;
  throughput: ThroughputConfig;
  memeAlpha: MemeAlphaConfig;
  execution: ExecutionConfig;
}

const asNumber = (value: string | undefined, fallback: number): number => {
  if (value === undefined || value.trim() === "") {
    return fallback;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asInt = (value: string | undefined, fallback: number): number => {
  if (value === undefined || value.trim() === "") {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asBool = (value: string | undefined, fallback: boolean): boolean => {
  if (value === undefined || value.trim() === "") {
    return fallback;
  }
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
};

const splitList = (value: string | undefined): string[] => {
  if (!value) {
    return [];
  }
  return value.split(",").map((item) => item.trim()).filter(Boolean);
};

const asJsonRecord = (value: string | undefined, fallback: Record<string, number>): Record<string, number> => {
  if (!value || value.trim() === "") {
    return fallback;
  }
  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    return Object.fromEntries(Object.entries(parsed).map(([key, item]) => [key, asNumber(String(item), fallback[key] ?? 0)]));
  } catch {
    return fallback;
  }
};

const normalizeMode = (value: string | undefined): RuntimeMode => {
  return value?.toLowerCase() === "live" ? "live" : "paper";
};

const normalizeLogLevel = (value: string | undefined): LogLevel => {
  const resolved = value?.toLowerCase();
  if (resolved === "debug" || resolved === "warn" || resolved === "error") {
    return resolved;
  }
  return "info";
};

export const loadConfig = (env: NodeJS.ProcessEnv = process.env): BotConfig => {
  const primaryRpc = env.SOLANA_RPC_URL || "https://api.mainnet-beta.solana.com";
  const rpcUrls = [...splitList(env.SOLANA_RPC_URLS), env.TITAN_RPC_URL, primaryRpc].filter((url): url is string => Boolean(url));
  const uniqueRpcUrls = Array.from(new Set(rpcUrls));
  const startingCapitalSol = asNumber(env.STARTING_CAPITAL_SOL, 10);
  const mode = normalizeMode(env.BOT_MODE);
  const liveTrading = asBool(env.LIVE_TRADING, false);

  return {
    mode,
    liveTrading,
    logLevel: normalizeLogLevel(env.LOG_LEVEL),
    emergencyHaltPort: asInt(env.EMERGENCY_HALT_PORT, 9090),
    modelRegistryPath: env.MODEL_REGISTRY_PATH || "./models/registry.json",
    rpc: {
      httpUrls: uniqueRpcUrls,
      wsUrl: env.SOLANA_WS_URL,
      requestTimeoutMs: asInt(env.RPC_TIMEOUT_MS, 1200),
      maxRetries: asInt(env.RPC_MAX_RETRIES, 3)
    },
    jito: {
      blockEngineUrl: env.JITO_BLOCK_ENGINE_URL || "https://frankfurt.mainnet.block-engine.jito.wtf",
      authUuid: env.JITO_AUTH_UUID,
      minTipSol: asNumber(env.JITO_MIN_TIP_SOL, 0.0001),
      maxTipSol: asNumber(env.JITO_MAX_TIP_SOL, 0.001),
      tipFloorUrl: env.JITO_TIP_FLOOR_URL || "https://bundles.jito.wtf/api/v1/bundles/tip_floor",
      bundleOnly: true,
      maxLandingPollAttempts: asInt(env.JITO_MAX_LANDING_POLL_ATTEMPTS, 8),
      landingPollIntervalMs: asInt(env.JITO_LANDING_POLL_INTERVAL_MS, 500)
    },
    risk: {
      startingCapitalSol,
      maxPositionFraction: Math.min(asNumber(env.MAX_POSITION_FRACTION, 0.1), 0.1),
      kellyFraction: asNumber(env.KELLY_FRACTION, 0.15),
      compoundingReinvestWinPct: Math.min(Math.max(asNumber(env.COMPOUNDING_REINVEST_WIN_PCT, 0.8), 0), 1),
      compoundingMaxBoost: Math.min(Math.max(asNumber(env.COMPOUNDING_MAX_BOOST, 0.35), 0), 1),
      compoundingMaxReserveFraction: Math.min(Math.max(asNumber(env.COMPOUNDING_MAX_RESERVE_FRACTION, 0.4), 0), 1),
      highAlphaKellyBoost: Math.min(Math.max(asNumber(env.HIGH_ALPHA_KELLY_BOOST, 0.25), 0), 1),
      maxLiquidityPositionFraction: Math.min(Math.max(asNumber(env.MAX_LIQUIDITY_POSITION_FRACTION, 0.015), 0.001), 0.05),
      volumeBottleneckLiquidityFraction: Math.min(Math.max(asNumber(env.VOLUME_BOTTLENECK_LIQUIDITY_FRACTION, 0.008), 0.001), 0.03),
      maxTotalExposureFraction: asNumber(env.MAX_TOTAL_EXPOSURE_FRACTION, 0.55),
      maxClusterExposureFraction: asNumber(env.MAX_CLUSTER_EXPOSURE_FRACTION, 0.25),
      maxPositionPerPlatform: asJsonRecord(env.MAX_POSITION_PER_PLATFORM, { pump_fun: 0.3, raydium: 0.4, other: 0.2 }),
      correlationClusterPositionThreshold: asInt(env.CORRELATION_CLUSTER_POSITION_THRESHOLD, 3),
      tailRiskVolatilityThreshold: asNumber(env.TAIL_RISK_VOLATILITY_THRESHOLD, 0.75),
      maxDrawdownCircuitBreakerPct: asNumber(env.MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT, 12),
      dailyDrawdownCircuitBreakerPct: asNumber(env.DAILY_DRAWDOWN_CIRCUIT_BREAKER_PCT, 8),
      dailyLossLimitSol: asNumber(env.DAILY_LOSS_LIMIT_SOL, 5),
      volatilitySpikeBlock: asNumber(env.VOLATILITY_SPIKE_BLOCK, 0.75),
      minTradeSol: asNumber(env.MIN_TRADE_SOL, 0.001),
      maxOpenPositions: asInt(env.MAX_OPEN_POSITIONS, 200),
      baseStopLossPct: asNumber(env.BASE_STOP_LOSS_PCT, 0.12),
      baseTakeProfitPct: asNumber(env.BASE_TAKE_PROFIT_PCT, 0.24),
      consecutiveLossCircuitBreaker: asInt(env.CONSECUTIVE_LOSS_CIRCUIT_BREAKER, 5),
      maxDeployerPositions: asInt(env.MAX_DEPLOYER_POSITIONS, 2),
      rugProbBlockThreshold: asNumber(env.RUG_PROB_BLOCK_THRESHOLD, RUG_PROB_BLOCK_THRESHOLD_DEFAULT)
    },
    scorer: {
      modelPath: env.RUG_MODEL_PATH || "./models/rug_model.json",
      xgbModelPath: env.XGB_MODEL_PATH || "./models/xgb_model.onnx",
      xgbEnsembleWeight: asNumber(env.XGB_ENSEMBLE_WEIGHT, 0.4),
      torchEnsembleWeight: asNumber(env.TORCH_ENSEMBLE_WEIGHT, 0.6),
      rugcheckEnabled: asBool(env.RUGCHECK_ENABLED, false),
      rugcheckApiUrl: env.RUGCHECK_API_URL || "https://api.rugcheck.xyz",
      rugcheckApiKey: env.RUGCHECK_API_KEY,
      rugProbBlockThreshold: asNumber(env.RUG_PROB_BLOCK_THRESHOLD, RUG_PROB_BLOCK_THRESHOLD_DEFAULT),
      rugPullBlockThreshold: asNumber(env.RUG_PULL_BLOCK_THRESHOLD, 0.12),
      minLpBurnPct: asNumber(env.MIN_LP_BURN_PCT, 0.9),
      minLpLockPct: asNumber(env.MIN_LP_LOCK_PCT, 90),
      maxHoldHorizonMs: asInt(env.MAX_HOLD_HORIZON_MS, 24 * 3_600_000),
      honeypotRiskThreshold: asNumber(env.HONEYPOT_RISK_THRESHOLD, 0.1),
      maxTransferTaxPct: asNumber(env.MAX_TRANSFER_TAX_PCT, 0.08),
      deployerBlacklist: new Set(splitList(env.DEPLOYER_BLACKLIST))
    },
    wallets: {
      masterPublicKey: env.MASTER_WALLET_PUBLIC_KEY,
      satelliteWalletsJson: env.SATELLITE_WALLETS_JSON,
      rotationCount: asInt(env.WALLET_ROTATION_COUNT, 200),
      redisUrl: env.REDIS_URL
    },
    throughput: {
      targetEventsPerHour: asInt(env.TARGET_EVENTS_PER_HOUR, 500000),
      targetTradesPerHour: asInt(env.TARGET_TRADES_PER_HOUR, 500),
      streamFanout: asInt(env.STREAM_FANOUT, 20),
      eventLoopBatchSize: asInt(env.EVENT_LOOP_BATCH_SIZE, 256),
      maxQueueDepth: asInt(env.MAX_QUEUE_DEPTH, 50000),
      maxEventAgeMs: asInt(env.MAX_EVENT_AGE_MS, 5000)
    },
    memeAlpha: {
      enabled: asBool(env.MEME_ALPHA_ENABLED, true),
      streamsEnabled: asBool(env.MEME_ALPHA_STREAMS_ENABLED, false),
      minScore: asNumber(env.MEME_ALPHA_MIN_SCORE, 0.58),
      highConvictionScore: asNumber(env.MEME_ALPHA_HIGH_CONVICTION_SCORE, 0.78),
      sentimentHalfLifeMs: asInt(env.SENTIMENT_HALF_LIFE_MS, 120_000),
      sentimentStaleMs: asInt(env.SENTIMENT_STALE_MS, 300_000),
      socialWsUrls: splitList(env.SOCIAL_WS_URLS),
      solanaWsUrl: env.SOLANA_WS_URL,
      solanaLogMentions: splitList(env.SOLANA_LOG_MENTIONS),
      baseWsUrl: env.BASE_WS_URL,
      baseLogAddresses: splitList(env.BASE_LOG_ADDRESSES),
      baseLogTopics: splitList(env.BASE_LOG_TOPICS),
      liquiditySpikePct: asNumber(env.LIQUIDITY_SPIKE_PCT, 0.35),
      minLiquidityDeltaSol: asNumber(env.MIN_LIQUIDITY_DELTA_SOL, 3),
      volumeSpikeRatio: asNumber(env.VOLUME_SPIKE_RATIO, 2.2),
      auditBudgetMs: asNumber(env.ANTI_RUG_AUDIT_BUDGET_MS, 10),
      blockOnAuditBudgetOverrun: asBool(env.ANTI_RUG_BLOCK_ON_BUDGET_OVERRUN, true),
      requireMintAuthorityRenounced: asBool(env.REQUIRE_MINT_AUTHORITY_RENOUNCED, true),
      requireFreezeAuthorityRenounced: asBool(env.REQUIRE_FREEZE_AUTHORITY_RENOUNCED, true),
      requireSocialContext: asBool(env.REQUIRE_SOCIAL_CONTEXT, false),
      maxTopHolderPct: asNumber(env.MAX_TOP_HOLDER_PCT, 0.22),
      maxTop10HolderPct: asNumber(env.MAX_TOP10_HOLDER_PCT, 0.55),
      maxDevHoldPct: asNumber(env.MAX_DEV_HOLD_PCT, 0.08)
    },
    execution: {
      maxSlippagePct: asNumber(env.MAX_SLIPPAGE_PCT, 1),
      jupiterQuoteApiUrl: env.JUPITER_QUOTE_API_URL || "https://quote-api.jup.ag/v6/quote",
      positionPollIntervalMs: asInt(env.POSITION_POLL_INTERVAL_MS, 30_000)
    }
  };
};
