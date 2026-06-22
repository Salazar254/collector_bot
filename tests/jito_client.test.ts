import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { loadConfig } from "../src/config.ts";
import { JitoClient } from "../src/utils/jito_client.ts";
import { createLogger } from "../src/utils/logger.ts";

const jitoConfig = () => loadConfig({
  JITO_MAX_LANDING_POLL_ATTEMPTS: "2",
  JITO_LANDING_POLL_INTERVAL_MS: "1"
}).jito;

export const runJitoClientTests = async (): Promise<void> => {
  const originalFetch = globalThis.fetch;
  let pollCount = 0;
  globalThis.fetch = async (input, init) => {
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : {};
    if (body.method === "sendBundle") {
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: "bundle_test_id" }), { status: 200 });
    }
    if (body.method === "getBundleStatuses") {
      pollCount += 1;
      const landed = pollCount >= 2;
      return new Response(JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        result: {
          value: landed ? [{ bundle_id: "bundle_test_id", confirmation_status: "confirmed", slot: 123 }] : []
        }
      }), { status: 200 });
    }
    if (body.method === "getInflightBundleStatuses") {
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { value: [] } }), { status: 200 });
    }
    return originalFetch(input, init);
  };

  try {
    const client = new JitoClient(jitoConfig(), createLogger("error"));
    const landed = await client.sendBundle(["dGVzdA=="], 0.2);
    assert.equal(landed.accepted, true);
    assert.equal(landed.landed, true);

    pollCount = 0;
    globalThis.fetch = async (input, init) => {
      const body = typeof init?.body === "string" ? JSON.parse(init.body) : {};
      if (body.method === "sendBundle") {
        return new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: "bundle_fail_id" }), { status: 200 });
      }
      if (body.method === "getInflightBundleStatuses") {
        return new Response(JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          result: { value: [{ bundle_id: "bundle_fail_id", status: "failed", err: "simulation_failed" }] }
        }), { status: 200 });
      }
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { value: [] } }), { status: 200 });
    };

    const failed = await client.sendBundle(["dGVzdA=="], 0.2);
    assert.equal(failed.accepted, false);
    assert.equal(failed.reason, "bundle_not_landed");
  } finally {
    globalThis.fetch = originalFetch;
  }
};

const entryUrl = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";

if (import.meta.url === entryUrl) {
  await runJitoClientTests();
}
