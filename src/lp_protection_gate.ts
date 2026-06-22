import type { RugCheckSummary } from "./token_risk_scorer.ts";

export type RugCheckStatus = "ok" | "failed" | "disabled" | "synthetic";

export interface LpProtectionConfig {
  minLpBurnPct: number;
  minLpLockPct: number;
  maxHoldHorizonMs: number;
}

export interface LpProtectionInput {
  lpBurnPct: number;
  rugcheck?: RugCheckSummary;
  rugcheckStatus: RugCheckStatus;
  synthetic?: boolean;
  nowMs?: number;
}

export interface LpProtectionResult {
  accepted: boolean;
  reasons: string[];
  burnConfirmed: boolean;
  lockConfirmed: boolean;
  lpLockedPct?: number;
  lpLockExpiryMs?: number;
}

export const KNOWN_LP_LOCKER_TYPES = new Set([
  "raydium_locker",
  "fluxbeam",
  "streamflow",
  "meteora",
  "burned",
  "goosefx",
  "uncx",
  "team_finance"
]);

export const lpProtectionConfigFromScorer = (scorer: {
  minLpBurnPct: number;
  minLpLockPct: number;
  maxHoldHorizonMs: number;
}): LpProtectionConfig => ({
  minLpBurnPct: scorer.minLpBurnPct,
  minLpLockPct: scorer.minLpLockPct,
  maxHoldHorizonMs: scorer.maxHoldHorizonMs
});

export const resolveLpLockedPct = (rugcheck?: RugCheckSummary): number => {
  if (!rugcheck) {
    return 0;
  }
  if (typeof rugcheck.lpLockedPct === "number" && Number.isFinite(rugcheck.lpLockedPct)) {
    return rugcheck.lpLockedPct;
  }
  return rugcheck.lpLocked ? 100 : 0;
};

export const parseLockerUnlockMs = (value: number | string | undefined | null): number | undefined => {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "number") {
    if (value === 0) {
      return 0;
    }
    return value < 1_000_000_000_000 ? value * 1000 : value;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const numeric = Number(trimmed);
  if (Number.isFinite(numeric)) {
    return parseLockerUnlockMs(numeric);
  }
  const parsed = Date.parse(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
};

export const lockCoversHoldHorizon = (
  expiryMs: number | undefined,
  nowMs: number,
  maxHoldHorizonMs: number
): boolean => {
  if (expiryMs === 0) {
    return true;
  }
  if (expiryMs === undefined) {
    return false;
  }
  return expiryMs >= nowMs + maxHoldHorizonMs;
};

export const evaluateLpProtection = (
  input: LpProtectionInput,
  config: LpProtectionConfig
): LpProtectionResult => {
  const nowMs = input.nowMs ?? Date.now();
  const reasons: string[] = [];
  const burnConfirmed = Number.isFinite(input.lpBurnPct) && input.lpBurnPct >= config.minLpBurnPct;

  let lockConfirmed = false;
  let lpLockedPct: number | undefined;
  let lpLockExpiryMs: number | undefined;

  if (burnConfirmed) {
    return {
      accepted: true,
      reasons: [],
      burnConfirmed: true,
      lockConfirmed: false,
      lpLockedPct: input.rugcheck ? resolveLpLockedPct(input.rugcheck) : undefined,
      lpLockExpiryMs: input.rugcheck?.lpLockExpiryMs
    };
  }

  if (input.rugcheckStatus === "ok" && input.rugcheck) {
    lpLockedPct = resolveLpLockedPct(input.rugcheck);
    lpLockExpiryMs = input.rugcheck.lpLockExpiryMs;
    const hasKnownLocker = (input.rugcheck.lpLockerTypes?.length ?? 0) > 0 || lpLockExpiryMs !== undefined;
    const pctOk = lpLockedPct >= config.minLpLockPct;
    const expiryOk = lockCoversHoldHorizon(lpLockExpiryMs, nowMs, config.maxHoldHorizonMs);
    lockConfirmed = pctOk && hasKnownLocker && expiryOk;

    if (!lockConfirmed) {
      if (!pctOk) {
        reasons.push("lp_lock_below_threshold");
      } else if (!hasKnownLocker) {
        reasons.push("lp_lock_unverified");
      } else if (!expiryOk) {
        reasons.push("lp_lock_expiry_within_hold");
      }
    }
  } else if (input.rugcheckStatus === "failed") {
    reasons.push("rugcheck_unavailable");
  } else if (input.rugcheckStatus === "synthetic" || input.synthetic) {
    reasons.push("synthetic_no_lp_verification");
  } else if (input.rugcheckStatus === "disabled") {
    reasons.push("lp_lock_verification_disabled");
  } else {
    reasons.push("lp_protection_missing");
  }

  const accepted = lockConfirmed;
  if (!accepted && reasons.length === 0) {
    reasons.push("lp_protection_missing");
  }

  return {
    accepted,
    reasons,
    burnConfirmed,
    lockConfirmed,
    lpLockedPct,
    lpLockExpiryMs
  };
};

export interface LockerVaultRecord {
  unlockDate?: number | string;
  type?: string;
}

export const summarizeLockerVaults = (
  lockers: Record<string, LockerVaultRecord> | LockerVaultRecord[] | undefined
): { lpLockExpiryMs?: number; lpLockerTypes: string[] } => {
  const entries = Array.isArray(lockers)
    ? lockers
    : lockers
      ? Object.values(lockers)
      : [];

  const lpLockerTypes: string[] = [];
  let lpLockExpiryMs: number | undefined;

  for (const locker of entries) {
    const lockerType = String(locker.type || "").trim().toLowerCase();
    if (lockerType && KNOWN_LP_LOCKER_TYPES.has(lockerType)) {
      lpLockerTypes.push(lockerType);
    }
    const unlockMs = parseLockerUnlockMs(locker.unlockDate);
    if (unlockMs === 0) {
      lpLockExpiryMs = 0;
      continue;
    }
    if (unlockMs !== undefined) {
      lpLockExpiryMs = lpLockExpiryMs === undefined ? unlockMs : Math.min(lpLockExpiryMs, unlockMs);
    }
  }

  return {
    lpLockExpiryMs,
    lpLockerTypes: [...new Set(lpLockerTypes)]
  };
};
