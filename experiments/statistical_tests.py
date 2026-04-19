"""Statistical significance tests for OLYMPUS experiments.

- Wilcoxon signed-rank test (non-parametric, paired)
- 95% confidence intervals via bootstrap
- Effect size (Cohen's d)
"""

from __future__ import annotations

import math
import random
from typing import Optional


def wilcoxon_test(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test.
    Returns (W statistic, p-value approximation via normal approximation).
    """
    assert len(x) == len(y), "Samples must be paired and equal length"
    differences = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in differences if d != 0]
    if not nonzero:
        return 0.0, 1.0

    # Rank differences
    nonzero.sort(key=lambda t: t[0])
    n = len(nonzero)
    ranks = list(range(1, n + 1))
    W_plus = sum(r for r, (_, s) in zip(ranks, nonzero) if s > 0)
    W_minus = sum(r for r, (_, s) in zip(ranks, nonzero) if s < 0)
    W = min(W_plus, W_minus)

    # Normal approximation for p-value
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return W, 1.0
    z = (W - mu) / sigma
    p = 2 * (1 - _normal_cdf(abs(z)))  # two-tailed
    return W, round(p, 6)


def _normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF."""
    return (1 + math.erf(z / math.sqrt(2))) / 2


def confidence_interval(
    data: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval.
    Returns (mean, lower_bound, upper_bound).
    """
    rng = random.Random(seed)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return data[0], data[0], data[0]

    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = 1 - confidence
    lo = means[int(alpha / 2 * n_bootstrap)]
    hi = means[int((1 - alpha / 2) * n_bootstrap)]
    mean = sum(data) / n
    return round(mean, 6), round(lo, 6), round(hi, 6)


def cohens_d(x: list[float], y: list[float]) -> float:
    """Effect size measure."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx = sum(x) / nx
    my = sum(y) / ny
    vx = sum((xi - mx) ** 2 for xi in x) / (nx - 1)
    vy = sum((yi - my) ** 2 for yi in y) / (ny - 1)
    pooled_std = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return round((mx - my) / pooled_std, 4)


def summarize_results(
    method_results: dict[str, list[float]],
    confidence: float = 0.95,
    baseline_key: Optional[str] = None,
) -> dict[str, dict]:
    """
    Summarize results across methods with CIs and significance.
    """
    summary = {}
    baseline_vals = method_results.get(baseline_key, []) if baseline_key else None

    for method, vals in method_results.items():
        mean, lo, hi = confidence_interval(vals, confidence)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        entry = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "n": len(vals),
        }
        if baseline_vals and method != baseline_key:
            if len(baseline_vals) == len(vals):
                W, p = wilcoxon_test(baseline_vals, vals)
                d = cohens_d(baseline_vals, vals)
                entry["wilcoxon_W"] = W
                entry["p_value"] = round(p, 4)
                entry["cohens_d"] = d
                entry["significant_p05"] = p < 0.05
        summary[method] = entry

    return summary


if __name__ == "__main__":
    # Quick self-test
    rng = random.Random(42)
    a = [rng.gauss(0.85, 0.02) for _ in range(10)]
    b = [rng.gauss(0.75, 0.03) for _ in range(10)]
    W, p = wilcoxon_test(a, b)
    mean, lo, hi = confidence_interval(a)
    d = cohens_d(a, b)
    print(f"Wilcoxon: W={W:.2f}, p={p:.4f}")
    print(f"CI (a): {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"Cohen's d: {d:.4f}")
