import type { JitoConfig } from "../config.ts";
import type { Logger } from "./logger.ts";

export interface BundleResult {
  bundleId: string;
  accepted: boolean;
  landed?: boolean;
  tipSol: number;
  reason?: string;
}

interface JitoRpcResponse<T> {
  jsonrpc: string;
  id: number;
  result?: T;
  error?: {
    code: number;
    message: string;
  };
}

interface TipFloorEntry {
  landed_tips_50th_percentile?: number;
  landed_tips_75th_percentile?: number;
  landed_tips_95th_percentile?: number;
  ema_landed_tips_50th_percentile?: number;
}

interface BundleStatusEntry {
  bundle_id?: string;
  status?: string;
  confirmation_status?: string;
  landed_slot?: number;
  slot?: number;
  err?: unknown;
}

interface BundleStatusesResult {
  value?: BundleStatusEntry[];
}

interface InflightBundleStatusEntry {
  bundle_id?: string;
  status?: string;
  landed_slot?: number;
  err?: unknown;
}

interface InflightBundleStatusesResult {
  value?: InflightBundleStatusEntry[];
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

const isLandedStatus = (entry?: BundleStatusEntry | InflightBundleStatusEntry): boolean => {
  if (!entry) {
    return false;
  }
  const status = String(entry.status || "").toLowerCase();
  if (status === "landed" || status === "confirmed") {
    return true;
  }
  const confirmation = String((entry as BundleStatusEntry).confirmation_status || "").toLowerCase();
  if (confirmation === "confirmed" || confirmation === "finalized") {
    return true;
  }
  return typeof entry.landed_slot === "number" || typeof (entry as BundleStatusEntry).slot === "number";
};

const isFailedStatus = (entry?: BundleStatusEntry | InflightBundleStatusEntry): boolean => {
  if (!entry) {
    return false;
  }
  if (entry.err) {
    return true;
  }
  const status = String(entry.status || "").toLowerCase();
  return status === "failed" || status === "invalid";
};

export class JitoClient {
  config: JitoConfig;
  logger: Logger;
  cachedTipAccounts: string[] = [];
  tipCursor = 0;

  constructor(config: JitoConfig, logger: Logger) {
    this.config = config;
    this.logger = logger.child({ component: "jito_client" });
  }

  async sendBundle(transactionsBase64: string[], competition = 0.5): Promise<BundleResult> {
    if (transactionsBase64.length === 0 || transactionsBase64.length > 5) {
      throw new Error("jito_bundle_requires_1_to_5_transactions");
    }
    const tipSol = await this.adaptiveTipSol(competition);
    const bundleId = await this.rpc<string>("/api/v1/bundles", "sendBundle", [
      transactionsBase64,
      { encoding: "base64" }
    ]);
    this.logger.info({ bundleId, txCount: transactionsBase64.length, tipSol }, "jito_bundle_submitted");
    return this.pollBundleLanding(bundleId, tipSol);
  }

  async pollBundleLanding(bundleId: string, tipSol: number): Promise<BundleResult> {
    const attempts = Math.max(1, this.config.maxLandingPollAttempts);
    const intervalMs = Math.max(50, this.config.landingPollIntervalMs);

    for (let attempt = 0; attempt < attempts; attempt += 1) {
      if (attempt > 0) {
        await sleep(intervalMs);
      }

      const landed = await this.fetchLandedBundle(bundleId);
      if (landed) {
        this.logger.info({ bundleId, attempt: attempt + 1 }, "jito_bundle_landed");
        return { bundleId, accepted: true, landed: true, tipSol };
      }

      const inflight = await this.fetchInflightBundle(bundleId);
      if (isFailedStatus(inflight)) {
        this.logger.warn({ bundleId, status: inflight?.status, err: inflight?.err }, "jito_bundle_not_landed");
        return { bundleId, accepted: false, landed: false, tipSol, reason: "bundle_not_landed" };
      }
    }

    this.logger.warn({ bundleId, attempts }, "jito_bundle_status_timeout");
    return { bundleId, accepted: false, landed: false, tipSol, reason: "bundle_status_timeout" };
  }

  private async fetchLandedBundle(bundleId: string): Promise<boolean> {
    try {
      const result = await this.rpc<BundleStatusesResult>("/api/v1/bundles", "getBundleStatuses", [[bundleId]]);
      const entry = result.value?.[0];
      return isLandedStatus(entry);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.debug({ bundleId, error: message }, "jito_get_bundle_statuses_failed");
      return false;
    }
  }

  private async fetchInflightBundle(bundleId: string): Promise<InflightBundleStatusEntry | undefined> {
    try {
      const result = await this.rpc<InflightBundleStatusesResult>("/api/v1/bundles", "getInflightBundleStatuses", [[bundleId]]);
      return result.value?.[0];
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.debug({ bundleId, error: message }, "jito_get_inflight_bundle_statuses_failed");
      return undefined;
    }
  }

  async getTipAccounts(): Promise<string[]> {
    if (this.cachedTipAccounts.length > 0) {
      return this.cachedTipAccounts;
    }
    try {
      this.cachedTipAccounts = await this.rpc<string[]>("/api/v1/bundles", "getTipAccounts", []);
    } catch {
      this.cachedTipAccounts = await this.rpc<string[]>("/api/v1/getTipAccounts", "getTipAccounts", []);
    }
    return this.cachedTipAccounts;
  }

  async nextTipAccount(): Promise<string> {
    const accounts = await this.getTipAccounts();
    if (accounts.length === 0) {
      throw new Error("no_jito_tip_accounts");
    }
    const account = accounts[this.tipCursor % accounts.length];
    this.tipCursor += 1;
    return account;
  }

  async adaptiveTipSol(competition = 0.5): Promise<number> {
    const floor = await this.fetchTipFloor();
    const floorTip = floor?.ema_landed_tips_50th_percentile || floor?.landed_tips_50th_percentile || this.config.minTipSol;
    const p75 = floor?.landed_tips_75th_percentile || floorTip * 1.8;
    const p95 = floor?.landed_tips_95th_percentile || p75 * 2.5;
    const boundedCompetition = Math.max(0, Math.min(1, competition));
    const raw = boundedCompetition < 0.55 ? floorTip : boundedCompetition < 0.85 ? p75 : p95;
    return Math.max(this.config.minTipSol, Math.min(this.config.maxTipSol, raw));
  }

  async fetchTipFloor(): Promise<TipFloorEntry | undefined> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 800);
      const response = await fetch(this.config.tipFloorUrl, { signal: controller.signal });
      clearTimeout(timeout);
      if (!response.ok) {
        return undefined;
      }
      const data = await response.json() as TipFloorEntry[];
      return Array.isArray(data) ? data[0] : undefined;
    } catch {
      return undefined;
    }
  }

  async rpc<T>(path: string, method: string, params: unknown[]): Promise<T> {
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (this.config.authUuid) {
      headers["x-jito-auth"] = this.config.authUuid;
    }
    const url = `${this.config.blockEngineUrl.replace(/\/$/, "")}${path}`;
    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: Date.now(),
        method,
        params
      })
    });
    if (!response.ok) {
      throw new Error(`jito_http_${response.status}`);
    }
    const payload = await response.json() as JitoRpcResponse<T>;
    if (payload.error) {
      throw new Error(`jito_rpc_${payload.error.code}:${payload.error.message}`);
    }
    if (payload.result === undefined) {
      throw new Error("jito_missing_result");
    }
    return payload.result;
  }
}
