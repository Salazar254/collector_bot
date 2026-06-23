export interface DistributionSummary {
  mean: number;
  std: number;
  min: number;
  max: number;
  samples: number;
}

export interface MultiTaskUncertainty {
  rugProb: DistributionSummary;
  timeToRug: DistributionSummary;
  maxDrawdown: DistributionSummary;
  pump2xProb: DistributionSummary;
}

export const clamp = (value: number, low = 0, high = 1): number => {
  if (!Number.isFinite(value)) return low;
  return Math.max(low, Math.min(high, value));
};

export const summarize = (values: number[]): DistributionSummary => {
  if (values.length === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, samples: 0 };
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...values),
    max: Math.max(...values),
    samples: values.length
  };
};

export const summarizeMultiTask = (samples: Array<{
  rugProb: number;
  timeToRug: number;
  maxDrawdown: number;
  pump2xProb: number;
}>): MultiTaskUncertainty => ({
  rugProb: summarize(samples.map((item) => item.rugProb)),
  timeToRug: summarize(samples.map((item) => item.timeToRug)),
  maxDrawdown: summarize(samples.map((item) => item.maxDrawdown)),
  pump2xProb: summarize(samples.map((item) => item.pump2xProb))
});

export interface RugBlockInput {
  riskProbability: number;
  uncertaintyStd: number;
  threshold: number;
  highUncertaintyStd?: number;
  uncertainThresholdFactor?: number;
}

/**
 * Fail-closed rug-probability gate. Blocks when the model is confident the token is
 * risky (riskProbability over threshold, any uncertainty), AND when the model is
 * uncertain about a token whose risk is already non-trivial. Uncertainty must never
 * widen the path to "proceed" — it lowers the effective block threshold, it does not
 * waive the block. Only a low-risk token the model is confident about passes cleanly.
 */
export const shouldBlockRugRisk = (input: RugBlockInput): boolean => {
  const highUncertaintyStd = input.highUncertaintyStd ?? 0.05;
  const uncertainThresholdFactor = input.uncertainThresholdFactor ?? 0.5;
  if (input.riskProbability > input.threshold) {
    return true;
  }
  const uncertain = input.uncertaintyStd >= highUncertaintyStd;
  return uncertain && input.riskProbability > input.threshold * uncertainThresholdFactor;
};

export const shouldReviewUncertainty = (std: number): boolean => std > 0.10;
