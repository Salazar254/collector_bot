export type SocialSource = "x" | "telegram" | "manual" | "feed";

export interface SocialPost {
  source: SocialSource;
  text: string;
  timestamp: number;
  mint?: string;
  symbol?: string;
  contractAddress?: string;
  author?: string;
  authorFollowers?: number;
  authorVerified?: boolean;
}

export interface SentimentScore {
  sentimentScore: number;
  whaleAccumulationScore: number;
  retailFomoScore: number;
  botSpamScore: number;
  credibilityScore: number;
  convictionScore: number;
  reasons: string[];
}

export interface SentimentSummary extends SentimentScore {
  samples: number;
  lastSeenAt?: number;
}

interface WeightedTerm {
  term: string;
  weight: number;
}

interface ScoredPost {
  post: SocialPost;
  score: SentimentScore;
}

const whaleTerms: WeightedTerm[] = [
  { term: "smart money", weight: 0.2 },
  { term: "whale", weight: 0.16 },
  { term: "swept", weight: 0.16 },
  { term: "accumulating", weight: 0.18 },
  { term: "accumulation", weight: 0.18 },
  { term: "fresh wallet", weight: 0.16 },
  { term: "funded by", weight: 0.14 },
  { term: "same funder", weight: 0.14 },
  { term: "builder bought", weight: 0.14 },
  { term: "dev bought", weight: 0.1 },
  { term: "lp added", weight: 0.14 },
  { term: "floor bid", weight: 0.12 },
  { term: "absorbing sells", weight: 0.16 },
  { term: "wallet cluster", weight: 0.16 }
];

const fomoTerms: WeightedTerm[] = [
  { term: "ape", weight: 0.09 },
  { term: "aping", weight: 0.09 },
  { term: "send it", weight: 0.11 },
  { term: "moon", weight: 0.08 },
  { term: "100x", weight: 0.13 },
  { term: "gem", weight: 0.08 },
  { term: "don't fade", weight: 0.11 },
  { term: "dont fade", weight: 0.11 },
  { term: "based", weight: 0.06 },
  { term: "cto", weight: 0.08 },
  { term: "cabal", weight: 0.08 },
  { term: "sendor", weight: 0.07 }
];

const riskTerms: WeightedTerm[] = [
  { term: "rug", weight: 0.2 },
  { term: "honeypot", weight: 0.25 },
  { term: "freeze", weight: 0.2 },
  { term: "mint auth", weight: 0.22 },
  { term: "dumping", weight: 0.16 },
  { term: "dev sold", weight: 0.18 },
  { term: "bundle snipe", weight: 0.12 },
  { term: "paid call", weight: 0.11 },
  { term: "blacklist", weight: 0.2 }
];

const spamTerms: WeightedTerm[] = [
  { term: "guaranteed", weight: 0.18 },
  { term: "no risk", weight: 0.18 },
  { term: "last chance", weight: 0.1 },
  { term: "private group", weight: 0.1 },
  { term: "airdrop", weight: 0.08 },
  { term: "presale", weight: 0.12 }
];

const clamp01 = (value: number): number => Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

export class DegenspeakSentimentEngine {
  score(post: SocialPost): SentimentScore {
    const text = post.text.toLowerCase();
    const whale = termScore(text, whaleTerms);
    const retail = termScore(text, fomoTerms);
    const risk = termScore(text, riskTerms);
    const spam = clamp01(termScore(text, spamTerms) + structuralSpamScore(post.text));
    const credibility = credibilityScore(post);
    const conviction = clamp01((whale * 0.55 + retail * 0.25 + credibility * 0.2) * (1 - spam * 0.55));
    const sentiment = clamp01((whale * 0.68 + retail * 0.24 - risk * 0.52 - spam * 0.35 + 0.1) * credibility);
    return {
      sentimentScore: sentiment,
      whaleAccumulationScore: clamp01(whale * credibility * (1 - spam * 0.35)),
      retailFomoScore: clamp01(retail * (1 - spam * 0.2)),
      botSpamScore: spam,
      credibilityScore: credibility,
      convictionScore: conviction,
      reasons: explain(text)
    };
  }
}

export class SocialSentimentBook {
  engine: DegenspeakSentimentEngine;
  halfLifeMs: number;
  staleMs: number;
  maxPostsPerKey: number;
  posts = new Map<string, ScoredPost[]>();

  constructor(options: {
    engine?: DegenspeakSentimentEngine;
    halfLifeMs: number;
    staleMs: number;
    maxPostsPerKey?: number;
  }) {
    this.engine = options.engine || new DegenspeakSentimentEngine();
    this.halfLifeMs = options.halfLifeMs;
    this.staleMs = options.staleMs;
    this.maxPostsPerKey = options.maxPostsPerKey ?? 128;
  }

  ingest(post: SocialPost): SentimentScore | undefined {
    const key = socialKey(post);
    if (!key) {
      return undefined;
    }
    const score = this.engine.score(post);
    const bucket = this.posts.get(key) || [];
    bucket.push({ post, score });
    if (bucket.length > this.maxPostsPerKey) {
      bucket.splice(0, bucket.length - this.maxPostsPerKey);
    }
    this.posts.set(key, bucket);
    return score;
  }

  summaryFor(keys: Array<string | undefined>, now = Date.now()): SentimentSummary {
    const rows = uniqueKeys(keys).flatMap((key) => this.posts.get(key) || []);
    const fresh = rows.filter((row) => now - row.post.timestamp <= this.staleMs);
    if (fresh.length === 0) {
      return neutralSummary();
    }
    let weightSum = 0;
    let sentiment = 0;
    let whale = 0;
    let retail = 0;
    let spam = 0;
    let credibility = 0;
    let conviction = 0;
    let lastSeenAt = 0;
    const reasons = new Set<string>();
    for (const row of fresh) {
      const ageMs = Math.max(0, now - row.post.timestamp);
      const weight = Math.exp(-Math.LN2 * ageMs / Math.max(this.halfLifeMs, 1));
      weightSum += weight;
      sentiment += row.score.sentimentScore * weight;
      whale += row.score.whaleAccumulationScore * weight;
      retail += row.score.retailFomoScore * weight;
      spam += row.score.botSpamScore * weight;
      credibility += row.score.credibilityScore * weight;
      conviction += row.score.convictionScore * weight;
      lastSeenAt = Math.max(lastSeenAt, row.post.timestamp);
      row.score.reasons.slice(0, 4).forEach((reason) => reasons.add(reason));
    }
    return {
      sentimentScore: sentiment / weightSum,
      whaleAccumulationScore: whale / weightSum,
      retailFomoScore: retail / weightSum,
      botSpamScore: spam / weightSum,
      credibilityScore: credibility / weightSum,
      convictionScore: conviction / weightSum,
      samples: fresh.length,
      lastSeenAt,
      reasons: [...reasons].slice(0, 8)
    };
  }
}

export const socialKey = (post: SocialPost): string | undefined => {
  return firstKey(post.mint, post.contractAddress, post.symbol);
};

export const firstKey = (...values: Array<string | undefined>): string | undefined => {
  for (const value of values) {
    const normalized = normalizeKey(value);
    if (normalized) {
      return normalized;
    }
  }
  return undefined;
};

export const normalizeKey = (value: string | undefined): string | undefined => {
  const normalized = value?.trim().toLowerCase().replace(/^\$/, "");
  return normalized || undefined;
};

const neutralSummary = (): SentimentSummary => ({
  sentimentScore: 0,
  whaleAccumulationScore: 0,
  retailFomoScore: 0,
  botSpamScore: 0,
  credibilityScore: 0,
  convictionScore: 0,
  samples: 0,
  reasons: []
});

const uniqueKeys = (keys: Array<string | undefined>): string[] => {
  return [...new Set(keys.map(normalizeKey).filter((key): key is string => Boolean(key)))];
};

const termScore = (text: string, terms: WeightedTerm[]): number => {
  let total = 0;
  for (const term of terms) {
    if (text.includes(term.term)) {
      total += term.weight;
    }
  }
  return clamp01(total);
};

const structuralSpamScore = (text: string): number => {
  const urlCount = (text.match(/https?:\/\//g) || []).length;
  const tickerCount = (text.match(/\$[a-z0-9]{2,12}/g) || []).length;
  const capsWords = (text.match(/\b[A-Z0-9]{5,}\b/g) || []).length;
  const repeated = /(.)\1{5,}/.test(text) ? 0.16 : 0;
  return clamp01(urlCount * 0.08 + Math.max(0, tickerCount - 2) * 0.06 + capsWords * 0.03 + repeated);
};

const credibilityScore = (post: SocialPost): number => {
  const followers = Math.max(0, post.authorFollowers ?? 0);
  const followerScore = followers > 0 ? Math.min(0.28, Math.log10(followers + 1) / 18) : 0;
  const verifiedScore = post.authorVerified ? 0.08 : 0;
  const sourceScore = post.source === "telegram" ? 0.52 : post.source === "x" ? 0.58 : 0.5;
  return clamp01(sourceScore + followerScore + verifiedScore);
};

const explain = (text: string): string[] => {
  const reasons: string[] = [];
  for (const [label, terms] of [
    ["whale", whaleTerms],
    ["fomo", fomoTerms],
    ["risk", riskTerms],
    ["spam", spamTerms]
  ] as const) {
    const hits = terms.filter((term) => text.includes(term.term)).map((term) => term.term);
    if (hits.length > 0) {
      reasons.push(`${label}:${hits.slice(0, 3).join(",")}`);
    }
  }
  if (structuralSpamScore(text) > 0.15) {
    reasons.push("spam:structure");
  }
  return reasons;
};
