import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadConfig } from "../src/config.ts";
import { SniperEngine } from "../src/sniper_engine.ts";
import { TokenRiskScorer } from "../src/token_risk_scorer.ts";
import type { TokenLaunchEvent, TokenRiskResult } from "../src/token_risk_scorer.ts";
import { createLogger } from "../src/utils/logger.ts";

const baseConfig = () => loadConfig({
  BOT_MODE: "paper",
  STARTING_CAPITAL_SOL: "100",
  RUGCHECK_ENABLED: "false",
  MAX_TOTAL_EXPOSURE_FRACTION: "1",
  MAX_CLUSTER_EXPOSURE_FRACTION: "1",
  MAX_POSITION_PER_PLATFORM: "{\"pump_fun\":1,\"raydium\":1,\"other\":1}",
  MIN_TRADE_SOL: "0.001",
  MAX_EVENT_AGE_MS: "5000",
  MEME_ALPHA_ENABLED: "false"
});

const launch = (overrides: Partial<TokenLaunchEvent> = {}): TokenLaunchEvent => ({
  mint: "stale_mint",
  deployer: "deployer_a",
  timestamp: Date.now(),
  chain: "solana",
  liquiditySol: 30,
  lpBurnPct: 0.99,
  ageSeconds: 3,
  uniqueBuyers: 44,
  totalVolumeSol: 90,
  marketCapSol: 420,
  rugPullRisk: 0.01,
  honeypotRisk: 0.01,
  transferTaxPct: 0,
  topHolderPct: 0.06,
  top10HolderPct: 0.24,
  devHoldPct: 0.02,
  mutableMetadata: false,
  mintAuthorityRenounced: true,
  freezeAuthorityRenounced: true,
  volatility1m: 0.24,
  priceVelocity1m: 0.22,
  buySellRatio: 1.8,
  jitoCompetition: 0.35,
  launchRatePerMinute: 500,
  predictedWinProb: 0.64,
  rewardRiskRatio: 1.7,
  launchPlatform: "raydium",
  synthetic: true,
  ...overrides
});

class StubScorer extends TokenRiskScorer {
  constructor() {
    super(baseConfig().scorer, createLogger("error"), { bias: 0, weights: {} });
  }

  override async evaluate(): Promise<TokenRiskResult> {
    return {
      accepted: true,
      riskProbability: 0.02,
      mlConfidence: 0.9,
      regime: "normal",
      reasons: [],
      rugcheck: {
        lpLockedPct: 95,
        lpLockExpiryMs: 0,
        lpLockerTypes: ["raydium_locker"]
      }
    };
  }
}

export const runSniperEngineTests = async (): Promise<void> => {
  const scorer = new StubScorer();
  const engine = new SniperEngine(baseConfig(), scorer, createLogger("error"));

  const stale = launch({ timestamp: Date.now() - 10_000 });
  const staleResult = await engine.processEvent(stale);
  assert.equal(staleResult, undefined);
  assert.equal(engine.stats.staleEventsDropped, 1);

  const fresh = launch({ mint: "fresh_mint", timestamp: Date.now() });
  const receipt = await engine.processEvent(fresh);
  assert.ok(receipt);
  assert.equal(receipt?.accepted, true);
  assert.equal(engine.stats.executed, 1);

  const cachedEvent = engine.attachLpProtectionCache(fresh, {
    accepted: true,
    riskProbability: 0.02,
    mlConfidence: 0.9,
    regime: "normal",
    reasons: [],
    rugcheck: { lpLockedPct: 95, lpLockExpiryMs: 0, lpLockerTypes: ["raydium_locker"] }
  });
  cachedEvent.lpProtectionCachedAt = Date.now() - 10_000;
  const wallet = { id: "w1", publicKey: "pk1", disabled: false, lastUsedAt: 0, inFlight: 0 };
  const rejected = await engine.execute(cachedEvent, wallet, 0.1, 0.12, 0.24);
  assert.equal(rejected.accepted, false);
  assert.equal(rejected.reason, "lp_cache_stale");
  assert.equal(engine.stats.lpRevalidationRejected, 1);
};

const entryUrl = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";

if (import.meta.url === entryUrl) {
  await runSniperEngineTests();
}
