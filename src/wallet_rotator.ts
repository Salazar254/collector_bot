import type { WalletConfig } from "./config.ts";
import type { Logger } from "./utils/logger.ts";

export interface WalletRef {
  id: string;
  publicKey: string;
  keypairPath?: string;
  disabled: boolean;
  lastUsedAt: number;
  inFlight: number;
}

export class WalletRotator {
  config: WalletConfig;
  logger: Logger;
  wallets: WalletRef[];
  cursor = 0;
  redisPublisher: { publish(channel: string, message: string): Promise<number>; quit?(): Promise<string> } | undefined;

  constructor(config: WalletConfig, logger: Logger) {
    this.config = config;
    this.logger = logger.child({ component: "wallet_rotator" });
    this.wallets = this.loadWallets();
  }

  async start(): Promise<void> {
    if (!this.config.redisUrl) {
      return;
    }
    try {
      const module = await import("ioredis");
      type RedisConstructor = new (url: string) => NonNullable<WalletRotator["redisPublisher"]>;
      const RedisCtor = (module.default as unknown) as RedisConstructor;
      this.redisPublisher = new RedisCtor(this.config.redisUrl);
      this.logger.info({ redisUrl: this.config.redisUrl }, "wallet_pubsub_connected");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.warn({ error: message }, "wallet_pubsub_disabled");
    }
  }

  async stop(): Promise<void> {
    if (this.redisPublisher?.quit) {
      await this.redisPublisher.quit();
    }
  }

  nextWallet(): WalletRef | null {
    if (this.wallets.length === 0) {
      throw new Error("no_wallets_available");
    }
    const poolSize = this.wallets.length;
    for (let attempt = 0; attempt < poolSize; attempt += 1) {
      const wallet = this.wallets[this.cursor % poolSize];
      this.cursor += 1;
      if (!wallet.disabled && wallet.inFlight === 0) {
        wallet.lastUsedAt = Date.now();
        wallet.inFlight += 1;
        return wallet;
      }
    }
    return null;
  }

  complete(walletId: string): void {
    const wallet = this.wallets.find((item) => item.id === walletId);
    if (!wallet) {
      return;
    }
    wallet.inFlight = Math.max(0, wallet.inFlight - 1);
  }

  disable(walletId: string): void {
    const wallet = this.wallets.find((item) => item.id === walletId);
    if (wallet) {
      wallet.disabled = true;
      this.logger.warn({ walletId }, "wallet_disabled");
    }
  }

  async publishAllocation(wallet: WalletRef, mint: string, amountSol: number): Promise<void> {
    if (!this.redisPublisher) {
      return;
    }
    await this.redisPublisher.publish("sniper.wallet.allocate", JSON.stringify({
      walletId: wallet.id,
      publicKey: wallet.publicKey,
      mint,
      amountSol,
      ts: Date.now()
    }));
  }

  loadWallets(): WalletRef[] {
    const parsed = this.parseConfiguredWallets();
    if (parsed.length > 0) {
      return parsed.slice(0, this.config.rotationCount);
    }
    const count = Math.max(1, this.config.rotationCount);
    return Array.from({ length: count }, (_, index) => ({
      id: `paper-${index.toString().padStart(3, "0")}`,
      publicKey: `paper_satellite_${index.toString().padStart(3, "0")}`,
      disabled: false,
      lastUsedAt: 0,
      inFlight: 0
    }));
  }

  parseConfiguredWallets(): WalletRef[] {
    if (!this.config.satelliteWalletsJson) {
      return [];
    }
    const raw = JSON.parse(this.config.satelliteWalletsJson) as Array<{ id?: string; publicKey: string; keypairPath?: string }>;
    return raw.map((item, index) => ({
      id: item.id || `wallet-${index.toString().padStart(3, "0")}`,
      publicKey: item.publicKey,
      keypairPath: item.keypairPath,
      disabled: false,
      lastUsedAt: 0,
      inFlight: 0
    }));
  }
}
