"""Confidence intervals and tests for the analysis (numpy only)."""
from math import comb, sqrt

import numpy as np

Z = 1.959964            # two-sided 95%
B = 10_000              # bootstrap resamples
SEED = 0


def wilson(k, n):
    """95% Wilson interval of k successes in n trials; (nan, nan) if n is 0."""
    if not n:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def bootstrap(sums, stat, b=B, seed=SEED):
    """
    Cluster bootstrap. sums is an array (clusters, k) of per-cluster sums and
    stat maps an array (..., k) of totals to a statistic. Clusters (games, or
    spots) are resampled with replacement. Returns (estimate, ci_lo, ci_hi, p),
    p two-sided for the statistic being 0 (twice the smaller share of
    resampled statistics on either side of 0, at most 1).
    """
    sums = np.asarray(sums, dtype=float)
    if len(sums) == 0:
        return (float("nan"),) * 4
    with np.errstate(divide="ignore", invalid="ignore"):
        est = float(stat(sums.sum(0)))
        rng = np.random.default_rng(seed)
        boot = np.concatenate([stat(sums[rng.integers(0, len(sums), (min(1000, b - i), len(sums)))]
                                    .sum(1)) for i in range(0, b, 1000)])
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    return est, float(lo), float(hi), float(p)


def binom_cdf(k, n, p=0.5):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar(b, c):
    """Exact two-sided McNemar p for b and c discordant pairs (1 if there are none)."""
    return min(1.0, 2 * binom_cdf(min(b, c), b + c)) if b + c else 1.0


def holm(pvals):
    """Holm-adjusted p-values, in the given order; nan stays nan."""
    p = np.asarray(pvals, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    out = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def paired(a, b, seed=SEED):
    """
    Paired comparison of two 0/1 outcomes on the same units (e.g. wins on
    shared seeds): rates, difference with a bootstrap CI over units, and the
    exact McNemar p.
    """
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    only_a, only_b = int(np.sum(a & (1 - b))), int(np.sum((1 - a) & b))
    est, lo, hi, _ = bootstrap(np.c_[a, b, np.ones(len(a))], lambda s: (s[..., 0] - s[..., 1]) / s[..., 2],
                               seed=seed)
    return {"n": len(a), "rate_a": a.mean() if len(a) else float("nan"),
            "rate_b": b.mean() if len(b) else float("nan"), "only_a": only_a, "only_b": only_b,
            "diff": est, "diff_lo": lo, "diff_hi": hi, "p_mcnemar": mcnemar(only_a, only_b)}
