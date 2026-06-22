import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadConfig } from "../src/config.ts";
import { WalletRotator } from "../src/wallet_rotator.ts";
import { createLogger } from "../src/utils/logger.ts";

export const runWalletRotatorTests = (): void => {
  const rotator = new WalletRotator(loadConfig({
    WALLET_ROTATION_COUNT: "2"
  }).wallets, createLogger("error"));

  const first = rotator.nextWallet();
  const second = rotator.nextWallet();
  assert.ok(first);
  assert.ok(second);
  assert.notEqual(first?.id, second?.id);
  assert.equal(rotator.nextWallet(), null);

  rotator.complete(first!.id);
  const third = rotator.nextWallet();
  assert.ok(third);
  assert.equal(third?.id, first?.id);
  rotator.complete(third!.id);
  rotator.complete(second!.id);
};

const entryUrl = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";

if (import.meta.url === entryUrl) {
  runWalletRotatorTests();
}
