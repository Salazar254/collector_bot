import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { RiskManager } from "./risk_manager.ts";
import type { Logger } from "./utils/logger.ts";

export interface EmergencyHaltOptions {
  port: number;
  risk: RiskManager;
  logger: Logger;
  cancelPendingBundles?: () => Promise<void> | void;
  resumeSecret?: string;
}

export class EmergencyHaltServer {
  options: EmergencyHaltOptions;
  server?: Server;

  constructor(options: EmergencyHaltOptions) {
    this.options = { ...options, logger: options.logger.child({ component: "emergency_halt" }) };
  }

  start(): Promise<void> {
    this.server = createServer((request, response) => {
      void this.route(request, response).catch((error) => {
        this.json(response, 500, { error: error instanceof Error ? error.message : String(error) });
      });
    });
    return new Promise((resolveListen) => {
      this.server?.listen(this.options.port, () => {
        this.options.logger.info({ port: this.options.port }, "emergency_halt_started");
        resolveListen();
      });
    });
  }

  stop(): Promise<void> {
    return new Promise((resolveClose) => {
      if (!this.server) {
        resolveClose();
        return;
      }
      this.server.close(() => resolveClose());
      this.server = undefined;
    });
  }

  async route(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const method = request.method || "GET";
    const url = request.url || "/";
    if (method === "POST" && url === "/halt") {
      this.options.risk.openCircuit("manual_emergency_halt");
      await this.options.cancelPendingBundles?.();
      this.json(response, 200, { halted: true, risk: this.options.risk.snapshot() });
      return;
    }
    if (method === "POST" && url === "/resume") {
      const secret = this.options.resumeSecret || process.env.EMERGENCY_HALT_SECRET;
      if (secret) {
        const provided = request.headers["x-halt-secret"] || request.headers["authorization"];
        if (provided !== secret && provided !== `Bearer ${secret}`) {
          this.options.logger.warn({ ip: request.socket.remoteAddress }, "resume_unauthorized");
          this.json(response, 401, { error: "unauthorized" });
          return;
        }
      }
      // TODO: Replace shared-secret with proper auth (mTLS or JWT) before production
      this.options.risk.closeCircuit();
      this.options.logger.info({ risk: this.options.risk.snapshot() }, "emergency_halt_resumed");
      this.json(response, 200, { halted: false, risk: this.options.risk.snapshot(), reviewRequired: true });
      return;
    }
    if (method === "GET" && url === "/health") {
      const snapshot = this.options.risk.snapshot();
      this.json(response, 200, {
        equitySol: snapshot.equitySol,
        openPositions: snapshot.openPositions,
        circuitBreakerOpen: snapshot.circuitBreakerOpen,
        circuitReason: snapshot.circuitReason
      });
      return;
    }
    this.json(response, 404, { error: "not_found" });
  }

  json(response: ServerResponse, status: number, payload: unknown): void {
    response.writeHead(status, { "content-type": "application/json" });
    response.end(JSON.stringify(payload));
  }
}
