import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  evaluateLpProtection,
  lockCoversHoldHorizon,
  parseLockerUnlockMs,
  summarizeLockerVaults
} from "../src/lp_protection_gate.ts";
import type { RugCheckSummary } from "../src/token_risk_scorer.ts";

const lpConfig = {
  minLpBurnPct: 0.9,
  minLpLockPct: 90,
  maxHoldHorizonMs: 3_600_000
};

const nowMs = 1_700_000_000_000;

export const runLpProtectionGateTests = (): void => {
  const burnPass = evaluateLpProtection(
    { lpBurnPct: 0.95, rugcheckStatus: "disabled", synthetic: true, nowMs },
    lpConfig
  );
  assert.equal(burnPass.accepted, true);
  assert.equal(burnPass.burnConfirmed, true);

  const syntheticReject = evaluateLpProtection(
    { lpBurnPct: 0.5, rugcheckStatus: "synthetic", synthetic: true, nowMs },
    lpConfig
  );
  assert.equal(syntheticReject.accepted, false);
  assert.ok(syntheticReject.reasons.includes("synthetic_no_lp_verification"));

  const rugcheckFail = evaluateLpProtection(
    { lpBurnPct: 0.5, rugcheckStatus: "failed", nowMs },
    lpConfig
  );
  assert.equal(rugcheckFail.accepted, false);
  assert.ok(rugcheckFail.reasons.includes("rugcheck_unavailable"));

  const lockPass: RugCheckSummary = {
    lpLockedPct: 95,
    lpLockExpiryMs: 0,
    lpLockerTypes: ["raydium_locker"]
  };
  const locked = evaluateLpProtection(
    { lpBurnPct: 0.2, rugcheck: lockPass, rugcheckStatus: "ok", nowMs },
    lpConfig
  );
  assert.equal(locked.accepted, true);
  assert.equal(locked.lockConfirmed, true);

  const shortLock: RugCheckSummary = {
    lpLockedPct: 95,
    lpLockExpiryMs: nowMs + 1_000,
    lpLockerTypes: ["streamflow"]
  };
  const shortLocked = evaluateLpProtection(
    { lpBurnPct: 0.2, rugcheck: shortLock, rugcheckStatus: "ok", nowMs },
    lpConfig
  );
  assert.equal(shortLocked.accepted, false);
  assert.ok(shortLocked.reasons.includes("lp_lock_expiry_within_hold"));

  assert.equal(parseLockerUnlockMs(0), 0);
  assert.equal(parseLockerUnlockMs(1_700_000_000), 1_700_000_000_000);
  assert.equal(lockCoversHoldHorizon(0, nowMs, lpConfig.maxHoldHorizonMs), true);
  assert.equal(lockCoversHoldHorizon(nowMs + 500_000, nowMs, lpConfig.maxHoldHorizonMs), false);

  const lockerSummary = summarizeLockerVaults({
    lockerA: { type: "raydium_locker", unlockDate: 0 },
    lockerB: { type: "streamflow", unlockDate: nowMs + 9_000_000 }
  });
  assert.deepEqual(lockerSummary.lpLockerTypes, ["raydium_locker", "streamflow"]);
  assert.equal(lockerSummary.lpLockExpiryMs, 0);
};

const entryUrl = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";

if (import.meta.url === entryUrl) {
  runLpProtectionGateTests();
}
