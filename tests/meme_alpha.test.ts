import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadConfig } from "../src/config.ts";
import { AntiRugGuard } from "../src/meme_alpha/anti_rug_guard.ts";
import { MemeAlphaAgent } from "../src/meme_alpha/meme_alpha_agent.ts";
import { lpProtectionConfigFromScorer } from "../src/lp_protection_gate.ts";
import { DegenspeakSentimentEngine } from "../src/meme_alpha/sentiment_engine.ts";
import { RiskManager, type RiskSignal } from "../src/risk_manager.ts";
import type { TokenLaunchEvent } from "../src/token_risk_scorer.ts";
import { createLogger } from "../src/utils/logger.ts";

const baseConfig = () => loadConfig({
  BOT_MODE: "paper",
  STARTING_CAPITAL_SOL: "100",
  RUGCHECK_ENABLED: "false",
  MAX_TOTAL_EXPOSURE_FRACTION: "1",
  MAX_CLUSTER_EXPOSURE_FRACTION: "1",
  MAX_POSITION_PER_PLATFORM: "{\"pump_fun\":1,\"raydium\":1,\"other\":1}",
  MIN_TRADE_SOL: "0.001"
});

const launch = (overrides: Partial<TokenLaunchEvent> = {}): TokenLaunchEvent => ({
  mint: "alpha_mint",
  deployer: "deployer_a",
  timestamp: Date.now(),
  chain: "solana",
  liquiditySol: 30,
  previousLiquiditySol: 10,
  lpBurnPct: 0.99,
  ageSeconds: 3,
  uniqueBuyers: 44,
  totalVolumeSol: 90,
  previousVolumeSol: 20,
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

const riskSignal = (overrides: Partial<RiskSignal> = {}): RiskSignal => ({
  mint: "risk_alpha",
  timestamp: Date.now(),
  regime: "normal",
  riskProbability: 0.02,
  mlConfidence: 0.9,
  winProbability: 0.65,
  rewardRiskRatio: 1.8,
  liquiditySol: 10_000,
  volatility: 0.2,
  launchPlatform: "raydium",
  ...overrides
});

export const runMemeAlphaTests = async (): Promise<void> => {
  const sentiment = new DegenspeakSentimentEngine();
  const whale = sentiment.score({
    source: "x",
    text: "Smart money accumulating, fresh wallet cluster swept the floor after LP added.",
    timestamp: Date.now(),
    authorFollowers: 50_000,
    authorVerified: true
  });
  const retail = sentiment.score({
    source: "telegram",
    text: "100x moon gem ape now, send it, don't fade!",
    timestamp: Date.now()
  });
  assert.ok(whale.whaleAccumulationScore > retail.whaleAccumulationScore);
  assert.ok(retail.retailFomoScore > whale.retailFomoScore);

  const lpProtection = lpProtectionConfigFromScorer(baseConfig().scorer);
  const guard = new AntiRugGuard(baseConfig().memeAlpha, lpProtection);
  const rugAudit = guard.audit(launch({ mintAuthorityRenounced: false, topHolderPct: 0.42 }));
  assert.equal(rugAudit.accepted, false);
  assert.ok(rugAudit.reasons.includes("mint_authority_active"));
  assert.ok(rugAudit.reasons.includes("top_holder_concentration"));

  const agent = new MemeAlphaAgent(baseConfig().memeAlpha, createLogger("error"), lpProtection);
  agent.ingestSocialPost({
    source: "x",
    mint: "alpha_mint",
    text: "Whale wallet cluster accumulating and absorbing sells, smart money swept.",
    timestamp: Date.now(),
    authorFollowers: 80_000,
    authorVerified: true
  });
  const decision = await agent.evaluate(launch());
  assert.equal(decision.accepted, true);
  assert.ok(decision.enrichedEvent.memeAlphaScore && decision.enrichedEvent.memeAlphaScore >= baseConfig().memeAlpha.minScore);
  assert.ok((decision.enrichedEvent.volumeBottleneckRatio ?? 0) > 0.55);

  const coldRisk = new RiskManager(baseConfig().risk);
  const hotRisk = new RiskManager(baseConfig().risk);
  hotRisk.recordEntry({ mint: "winner", amountSol: 1, openedAt: Date.now(), riskMode: "normal", platform: "raydium" });
  hotRisk.recordExit("winner", 1);
  assert.equal(Number(hotRisk.snapshot().compoundingReserveSol.toFixed(6)), 0.8);
  const coldPlan = coldRisk.planPosition(riskSignal());
  const hotPlan = hotRisk.planPosition(riskSignal({
    memeAlphaScore: 0.9,
    whaleAccumulationScore: 0.8,
    retailFomoScore: 0.2,
    volumeBottleneckRatio: 0.8
  }));
  assert.equal(hotPlan.accepted, true);
  assert.ok(hotPlan.positionFraction > coldPlan.positionFraction);
};

const entryUrl = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";

if (import.meta.url === entryUrl) {
  await runMemeAlphaTests();
}
