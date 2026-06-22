import type {
  ExitExecutionBackend,
  ExitExecutionReceipt,
  ExitExecutionRequest,
  PriceFeed,
  PriceQuote
} from "./position_manager.ts";
import type { Logger } from "./utils/logger.ts";

/**
 * Placeholder exit backend — logs exit signals but does NOT submit sells.
 * Replace with a real ExitExecutionBackend before deploying live capital.
 */
export class NullExitBackend implements ExitExecutionBackend {
  logger: Logger;

  constructor(logger: Logger) {
    this.logger = logger.child({ component: "null_exit_backend" });
  }

  async executeExit(request: ExitExecutionRequest): Promise<ExitExecutionReceipt> {
    this.logger.warn({
      mint: request.position.mint,
      walletId: request.position.walletId,
      exitReason: request.decision.reason,
      targetPriceSol: request.decision.targetPriceSol,
      quotePriceSol: request.quote.priceSol
    }, "exit_backend_not_configured");
    return {
      accepted: false,
      reason: "exit_backend_not_configured",
      realizedSol: 0
    };
  }
}

/** Returns the last tracked entry price so time-based exits can still evaluate. */
export class EntryPriceFeed implements PriceFeed {
  private readonly quotes = new Map<string, PriceQuote>();

  track(mint: string, entryPriceSol: number, timestamp = Date.now()): void {
    this.quotes.set(mint, { mint, priceSol: entryPriceSol, timestamp });
  }

  untrack(mint: string): void {
    this.quotes.delete(mint);
  }

  async quote(mint: string): Promise<PriceQuote | undefined> {
    return this.quotes.get(mint);
  }
}
