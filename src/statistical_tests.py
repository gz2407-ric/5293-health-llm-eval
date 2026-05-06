import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from utils import PROJECT_ROOT


SYSTEM_ORDER = ["baseline", "rag_only", "multistep_only", "full_pipeline"]

SYSTEM_COMPARISONS = [
    ("baseline", "rag_only"),
    ("baseline", "multistep_only"),
    ("baseline", "full_pipeline"),
    ("rag_only", "full_pipeline"),
    ("multistep_only", "full_pipeline"),
]


def rank_biserial_from_wilcoxon(x, y):
    """
    Approximate rank-biserial correlation for paired Wilcoxon.
    Positive value means system B > system A.
    """
    diff = np.array(y, dtype=float) - np.array(x, dtype=float)
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    abs_diff = np.abs(diff)
    ranks = pd.Series(abs_diff).rank(method="average").to_numpy()
    positive_rank_sum = ranks[diff > 0].sum()
    negative_rank_sum = ranks[diff < 0].sum()
    total_rank_sum = ranks.sum()

    if total_rank_sum == 0:
        return 0.0

    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def apply_fdr_correction(results):
    """
    Apply Benjamini-Hochberg FDR correction while safely ignoring NaN p-values.
    This fixes the previous bug where one NaN p-value made all adjusted p-values NaN.
    """
    results = results.copy()
    results["adjusted_p_value_fdr_bh"] = np.nan

    valid_mask = results["p_value"].notna()
    if valid_mask.sum() > 0:
        _, adjusted_p, _, _ = multipletests(
            results.loc[valid_mask, "p_value"].astype(float),
            method="fdr_bh"
        )
        results.loc[valid_mask, "adjusted_p_value_fdr_bh"] = adjusted_p

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    args = parser.parse_args()

    if args.mode == "pilot":
        input_path = PROJECT_ROOT / "results" / "pilot_judge_scores_unified.csv"
        output_path = PROJECT_ROOT / "results" / "pilot_statistical_results.csv"
    else:
        input_path = PROJECT_ROOT / "results" / "judge_scores.csv"
        output_path = PROJECT_ROOT / "results" / "statistical_results.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing judge score file: {input_path}")

    scores = pd.read_csv(input_path)
    scores = scores.dropna(subset=["score"]).copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score"])

    result_rows = []

    for metric in sorted(scores["metric"].unique()):
        metric_df = scores[scores["metric"] == metric]

        wide = metric_df.pivot_table(
            index="scenario_id",
            columns="system_name",
            values="score",
            aggfunc="mean"
        )

        for sys_a, sys_b in SYSTEM_COMPARISONS:
            if sys_a not in wide.columns or sys_b not in wide.columns:
                continue

            paired = wide[[sys_a, sys_b]].dropna()
            if len(paired) < 2:
                continue

            x = paired[sys_a].to_numpy(dtype=float)
            y = paired[sys_b].to_numpy(dtype=float)
            diff = y - x

            if np.all(diff == 0):
                stat, p_value = np.nan, np.nan
            else:
                try:
                    stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    stat, p_value = np.nan, np.nan

            effect_size = rank_biserial_from_wilcoxon(x, y)

            result_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{sys_a} vs {sys_b}",
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "n_pairs": len(paired),
                    "mean_score_a": float(np.mean(x)),
                    "mean_score_b": float(np.mean(y)),
                    "mean_difference_b_minus_a": float(np.mean(diff)),
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "effect_size_rank_biserial": effect_size,
                }
            )

    results = pd.DataFrame(result_rows)
    results = apply_fdr_correction(results)

    results.to_csv(output_path, index=False)
    print(f"Saved statistical results to {output_path}")

    if "adjusted_p_value_fdr_bh" in results.columns:
        print("\nSignificant results at FDR-adjusted p < 0.05:")
        sig = results[
            results["adjusted_p_value_fdr_bh"].notna()
            & (results["adjusted_p_value_fdr_bh"] < 0.05)
        ]
        if sig.empty:
            print("None")
        else:
            print(sig[[
                "metric", "comparison", "mean_difference_b_minus_a",
                "p_value", "adjusted_p_value_fdr_bh", "effect_size_rank_biserial"
            ]].to_string(index=False))


if __name__ == "__main__":
    main()
