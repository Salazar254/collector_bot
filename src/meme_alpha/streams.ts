import WebSocket from "ws";
import type { MemeAlphaConfig } from "../config.ts";
import type { TokenLaunchEvent } from "../token_risk_scorer.ts";
import type { Logger } from "../utils/logger.ts";
import type { SocialPost, SocialSource } from "./sentiment_engine.ts";

export interface MemeAlphaStreamHandlers {
  onLaunch(event: TokenLaunchEvent): void | Promise<void>;
  onSocial(post: SocialPost): void | Promise<void>;
}

interface StreamRef {
  name: string;
  url: string;
  socket: WebSocket;
  attempts: number;
}

export class MemeAlphaStreamHub {
  config: MemeAlphaConfig;
  logger: Logger;
  handlers: MemeAlphaStreamHandlers;
  streams: StreamRef[] = [];
  stopped = false;

  constructor(config: MemeAlphaConfig, logger: Logger, handlers: MemeAlphaStreamHandlers) {
    this.config = config;
    this.logger = logger.child({ component: "meme_alpha_streams" });
    this.handlers = handlers;
  }

  start(): void {
    this.stopped = false;
    for (const url of this.config.socialWsUrls) {
      this.connect("social", url, (socket) => {
        socket.send(JSON.stringify({ type: "subscribe", streams: ["x", "telegram"] }));
      });
    }
    if (this.config.solanaWsUrl) {
      this.connect("solana", this.config.solanaWsUrl, (socket) => {
        const mentions = this.config.solanaLogMentions.length > 0 ? this.config.solanaLogMentions : ["TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"];
        for (const mention of mentions) {
          socket.send(JSON.stringify({
            jsonrpc: "2.0",
            id: `logs-${mention}`,
            method: "logsSubscribe",
            params: [{ mentions: [mention] }, { commitment: "processed" }]
          }));
        }
      });
    }
    if (this.config.baseWsUrl) {
      this.connect("base", this.config.baseWsUrl, (socket) => {
        socket.send(JSON.stringify({
          jsonrpc: "2.0",
          id: "base-logs",
          method: "eth_subscribe",
          params: ["logs", {
            address: this.config.baseLogAddresses.length > 0 ? this.config.baseLogAddresses : undefined,
            topics: this.config.baseLogTopics.length > 0 ? this.config.baseLogTopics : undefined
          }]
        }));
      });
    }
  }

  stop(): void {
    this.stopped = true;
    for (const stream of this.streams) {
      stream.socket.close();
    }
    this.streams = [];
  }

  connect(name: string, url: string, subscribe: (socket: WebSocket) => void): void {
    const stream: StreamRef = { name, url, socket: new WebSocket(url), attempts: 0 };
    this.streams.push(stream);
    stream.socket.on("open", () => {
      stream.attempts = 0;
      subscribe(stream.socket);
      this.logger.info({ name, url: redactUrl(url) }, "stream_connected");
    });
    stream.socket.on("message", (data) => {
      void this.handleMessage(name, data.toString()).catch((error) => {
        this.logger.warn({ name, error: error instanceof Error ? error.message : String(error) }, "stream_message_rejected");
      });
    });
    stream.socket.on("close", () => {
      this.logger.warn({ name, url: redactUrl(url) }, "stream_closed");
      if (!this.stopped) {
        this.scheduleReconnect(stream, subscribe);
      }
    });
    stream.socket.on("error", (error) => {
      this.logger.warn({ name, url: redactUrl(url), error: error.message }, "stream_error");
    });
  }

  scheduleReconnect(stream: StreamRef, subscribe: (socket: WebSocket) => void): void {
    stream.attempts += 1;
    const delayMs = Math.min(30_000, 500 * 2 ** Math.min(stream.attempts, 6));
    setTimeout(() => {
      if (!this.stopped) {
        const index = this.streams.indexOf(stream);
        if (index >= 0) {
          this.streams.splice(index, 1);
        }
        this.connect(stream.name, stream.url, subscribe);
      }
    }, delayMs);
  }

  async handleMessage(name: string, raw: string): Promise<void> {
    const parsed = parseJson(raw);
    if (!parsed) {
      return;
    }
    const social = normalizeSocial(parsed);
    if (social) {
      await this.handlers.onSocial(social);
      return;
    }
    const launch = normalizeLaunch(parsed, name);
    if (launch) {
      if (launch.dataQuality === "incomplete") {
        this.logger.warn({ stream: name, mint: launch.mint, dataQuality: launch.dataQuality }, "launch_event_rejected_incomplete_data");
        return;
      }
      await this.handlers.onLaunch(launch);
      return;
    }
    const embedded = extractEmbeddedJson(parsed);
    if (embedded) {
      await this.handleMessage(name, JSON.stringify(embedded));
    }
  }
}

const normalizeSocial = (payload: Record<string, unknown>): SocialPost | undefined => {
  const text = stringValue(payload.text ?? payload.message ?? payload.body);
  if (!text) {
    return undefined;
  }
  const source = normalizeSource(stringValue(payload.source) || "feed");
  return {
    source,
    text,
    timestamp: numberValue(payload.timestamp) || Date.now(),
    mint: stringValue(payload.mint),
    symbol: stringValue(payload.symbol),
    contractAddress: stringValue(payload.contractAddress ?? payload.contract_address),
    author: stringValue(payload.author ?? payload.username),
    authorFollowers: numberValue(payload.authorFollowers ?? payload.followers),
    authorVerified: boolValue(payload.authorVerified ?? payload.verified)
  };
};
const detectMissingSafetyFields = (payload: Record<string, unknown>): string[] => {
  const incompleteFields: string[] = [];
  const raw = (key: string, fallback: string): unknown => payload[key] ?? payload[fallback];
  if (numberValue(raw("lpBurnPct", "lp_burn_pct")) === undefined) incompleteFields.push("lpBurnPct");
  if (numberValue(raw("rugPullRisk", "rug_pull_risk")) === undefined) incompleteFields.push("rugPullRisk");
  if (numberValue(raw("honeypotRisk", "honeypot_risk")) === undefined) incompleteFields.push("honeypotRisk");
  if (numberValue(raw("transferTaxPct", "transfer_tax_pct")) === undefined) incompleteFields.push("transferTaxPct");
  if (numberValue(raw("topHolderPct", "top_holder_pct")) === undefined) incompleteFields.push("topHolderPct");
  if (numberValue(raw("top10HolderPct", "top10_holder_pct")) === undefined) incompleteFields.push("top10HolderPct");
  if (numberValue(raw("devHoldPct", "dev_hold_pct")) === undefined) incompleteFields.push("devHoldPct");
  if (boolValue(raw("mintAuthorityRenounced", "mint_authority_renounced")) === undefined) incompleteFields.push("mintAuthorityRenounced");
  if (boolValue(raw("freezeAuthorityRenounced", "freeze_authority_renounced")) === undefined) incompleteFields.push("freezeAuthorityRenounced");
  return incompleteFields;
};

const validateLaunchEvent = (event: TokenLaunchEvent, missingFields: string[]): TokenLaunchEvent => {
  if (missingFields.length === 0) {
    return { ...event, dataQuality: "complete" };
  }
  const patched = { ...event, dataQuality: "incomplete" as const };
  for (const field of missingFields) {
    switch (field) {
      case "lpBurnPct": patched.lpBurnPct = 0; break;
      case "rugPullRisk": patched.rugPullRisk = 1; break;
      case "honeypotRisk": patched.honeypotRisk = 1; break;
      case "transferTaxPct": patched.transferTaxPct = 1; break;
      case "topHolderPct": patched.topHolderPct = 1; break;
      case "top10HolderPct": patched.top10HolderPct = 1; break;
      case "devHoldPct": patched.devHoldPct = 1; break;
      case "mintAuthorityRenounced": patched.mintAuthorityRenounced = false; break;
      case "freezeAuthorityRenounced": patched.freezeAuthorityRenounced = false; break;
    }
  }
  return patched;
};

const normalizeLaunch = (payload: Record<string, unknown>, streamName: string): TokenLaunchEvent | undefined => {
  const mint = stringValue(payload.mint ?? payload.contractAddress ?? payload.contract_address);
  const liquiditySol = numberValue(payload.liquiditySol ?? payload.liquidity_sol);
  if (!mint || liquiditySol === undefined) {
    return undefined;
  }
  const missingFields = detectMissingSafetyFields(payload);
  const event: TokenLaunchEvent = {
    mint,
    deployer: stringValue(payload.deployer) || "unknown",
    timestamp: numberValue(payload.timestamp) || Date.now(),
    chain: streamName === "base" ? "base" : "solana",
    contractAddress: stringValue(payload.contractAddress ?? payload.contract_address),
    poolAddress: stringValue(payload.poolAddress ?? payload.pool_address),
    txSignature: stringValue(payload.txSignature ?? payload.tx_signature),
    liquiditySol,
    previousLiquiditySol: numberValue(payload.previousLiquiditySol ?? payload.previous_liquidity_sol),
    liquiditySpikePct: numberValue(payload.liquiditySpikePct ?? payload.liquidity_spike_pct),
    lpBurnPct: numberValue(payload.lpBurnPct ?? payload.lp_burn_pct) ?? 0,
    ageSeconds: numberValue(payload.ageSeconds ?? payload.age_seconds) ?? 0,
    uniqueBuyers: numberValue(payload.uniqueBuyers ?? payload.unique_buyers) ?? 0,
    totalVolumeSol: numberValue(payload.totalVolumeSol ?? payload.total_volume_sol) ?? liquiditySol,
    previousVolumeSol: numberValue(payload.previousVolumeSol ?? payload.previous_volume_sol),
    marketCapSol: numberValue(payload.marketCapSol ?? payload.market_cap_sol) ?? liquiditySol * 12,
    rugPullRisk: numberValue(payload.rugPullRisk ?? payload.rug_pull_risk) ?? 1,
    honeypotRisk: numberValue(payload.honeypotRisk ?? payload.honeypot_risk) ?? 1,
    transferTaxPct: numberValue(payload.transferTaxPct ?? payload.transfer_tax_pct) ?? 1,
    topHolderPct: numberValue(payload.topHolderPct ?? payload.top_holder_pct) ?? 1,
    top10HolderPct: numberValue(payload.top10HolderPct ?? payload.top10_holder_pct) ?? 1,
    devHoldPct: numberValue(payload.devHoldPct ?? payload.dev_hold_pct) ?? 1,
    mutableMetadata: boolValue(payload.mutableMetadata ?? payload.mutable_metadata) ?? false,
    mintAuthorityRenounced: boolValue(payload.mintAuthorityRenounced ?? payload.mint_authority_renounced) ?? false,
    freezeAuthorityRenounced: boolValue(payload.freezeAuthorityRenounced ?? payload.freeze_authority_renounced) ?? false,
    ownerRenounced: boolValue(payload.ownerRenounced ?? payload.owner_renounced),
    proxyContract: boolValue(payload.proxyContract ?? payload.proxy_contract),
    blacklistFunction: boolValue(payload.blacklistFunction ?? payload.blacklist_function),
    tradingPaused: boolValue(payload.tradingPaused ?? payload.trading_paused),
    volatility1m: numberValue(payload.volatility1m ?? payload.volatility_1m) ?? 0.2,
    priceVelocity1m: numberValue(payload.priceVelocity1m ?? payload.price_velocity_1m) ?? 0,
    buySellRatio: numberValue(payload.buySellRatio ?? payload.buy_sell_ratio) ?? 1,
    jitoCompetition: numberValue(payload.jitoCompetition ?? payload.jito_competition) ?? 0.5,
    launchRatePerMinute: numberValue(payload.launchRatePerMinute ?? payload.launch_rate_per_minute) ?? 0,
    predictedWinProb: numberValue(payload.predictedWinProb ?? payload.predicted_win_prob) ?? 0,
    rewardRiskRatio: numberValue(payload.rewardRiskRatio ?? payload.reward_risk_ratio) ?? 0.1,
    launchPlatform: stringValue(payload.launchPlatform ?? payload.launch_platform ?? streamName)
  };
  return validateLaunchEvent(event, missingFields);
};

const extractEmbeddedJson = (payload: Record<string, unknown>): Record<string, unknown> | undefined => {
  const logs = payload.params && typeof payload.params === "object"
    ? (payload.params as { result?: { value?: { logs?: unknown } } }).result?.value?.logs
    : undefined;
  if (!Array.isArray(logs)) {
    return undefined;
  }
  for (const log of logs) {
    if (typeof log !== "string") {
      continue;
    }
    const marker = "meme-alpha:";
    const index = log.indexOf(marker);
    if (index >= 0) {
      return parseJson(log.slice(index + marker.length).trim());
    }
  }
  return undefined;
};

const parseJson = (raw: string): Record<string, unknown> | undefined => {
  try {
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : undefined;
  } catch {
    return undefined;
  }
};

const stringValue = (value: unknown): string | undefined => typeof value === "string" && value.trim() ? value.trim() : undefined;
const numberValue = (value: unknown): number | undefined => {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number.parseFloat(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : undefined;
};
const boolValue = (value: unknown): boolean | undefined => {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return ["1", "true", "yes", "on"].includes(value.toLowerCase());
  }
  return undefined;
};
const normalizeSource = (value: string): SocialSource => value === "x" || value === "telegram" || value === "manual" ? value : "feed";
const redactUrl = (url: string): string => url.replace(/([?&](?:token|key|api_key)=)[^&]+/gi, "$1redacted");
