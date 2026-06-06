"""Statistical analysis for cross-experiment comparison.

Provides:
- Kruskal-Wallis H test across momentum fractions (non-parametric ANOVA)
- Pairwise Mann-Whitney U tests with Bonferroni correction
- Effect size (rank-biserial correlation r)
- Spearman rank correlation between momentum fraction and outcome metrics
- Summary report as a dict (suitable for JSON export or dashboard display)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


_METRICS = [
    "n_crash_steps",
    "max_avg_leverage",
    "peak_ponzi_pct",
    "max_drawdown_pct",
    "total_margin_calls",
    "total_defaults",
    "final_gini",
    "annualised_sharpe",
]


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size r for Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1 - (2 * u_stat) / (n1 * n2))


def kruskal_wallis(df: pd.DataFrame, metric: str) -> dict:
    """Kruskal-Wallis H test: does metric differ across momentum fractions?"""
    groups = [g[metric].values for _, g in df.groupby("momentum_fraction")]
    if len(set(np.concatenate(groups))) < 2:
        return {"H": 0.0, "p": 1.0, "significant": False, "note": "all_identical"}
    h_stat, p_val = stats.kruskal(*groups)
    return {"H": float(h_stat), "p": float(p_val), "significant": bool(p_val < 0.05)}


def pairwise_mannwhitney(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """All pairwise Mann-Whitney U tests with Bonferroni correction."""
    fracs = sorted(df["momentum_fraction"].unique())
    rows = []
    n_comparisons = len(fracs) * (len(fracs) - 1) // 2

    for i, f1 in enumerate(fracs):
        for f2 in fracs[i + 1:]:
            x = df.loc[df["momentum_fraction"] == f1, metric].values
            y = df.loc[df["momentum_fraction"] == f2, metric].values
            _, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_bonf = min(p_raw * n_comparisons, 1.0)
            r = _rank_biserial(x, y)
            rows.append({
                "frac_a": f1,
                "frac_b": f2,
                "p_raw": p_raw,
                "p_bonferroni": p_bonf,
                "effect_r": r,
                "significant": p_bonf < 0.05,
            })

    return pd.DataFrame(rows)


def spearman_correlation(df: pd.DataFrame, metric: str) -> dict:
    """Spearman rho between momentum fraction and metric (one value per seed)."""
    if df[metric].nunique() < 2:
        return {"rho": 0.0, "p": 1.0, "significant": False, "note": "constant"}
    rho, p = stats.spearmanr(df["momentum_fraction"], df[metric])
    rho = 0.0 if np.isnan(rho) else float(rho)
    p = 1.0 if np.isnan(p) else float(p)
    return {"rho": rho, "p": p, "significant": bool(p < 0.05)}


def full_report(df: pd.DataFrame) -> dict:
    """Run all tests for all metrics. Returns a nested dict."""
    report = {}
    for metric in _METRICS:
        if metric not in df.columns:
            continue
        report[metric] = {
            "kruskal_wallis": kruskal_wallis(df, metric),
            "spearman": spearman_correlation(df, metric),
        }
    return report


def save_report(df: pd.DataFrame, output_dir: str | Path) -> dict:
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = full_report(df)
    with open(out / "statistical_tests.json", "w") as f:
        json.dump(report, f, indent=2)

    # Pairwise for crash steps and max leverage (most informative)
    for metric in ("n_crash_steps", "max_avg_leverage", "peak_ponzi_pct"):
        if metric in df.columns:
            pw = pairwise_mannwhitney(df, metric)
            pw.to_csv(out / f"pairwise_{metric}.csv", index=False)

    print(f"Statistical tests saved to {out.resolve()}")
    return report


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/vary_momentum/raw_results.csv"
    df = pd.read_csv(csv_path)
    report = save_report(df, Path(csv_path).parent)

    print("\n--- Spearman correlations with momentum fraction ---")
    for metric, res in report.items():
        rho = res["spearman"]["rho"]
        p = res["spearman"]["p"]
        sig = "*" if res["spearman"]["significant"] else ""
        print(f"  {metric:<30} rho={rho:+.3f}  p={p:.4f} {sig}")
