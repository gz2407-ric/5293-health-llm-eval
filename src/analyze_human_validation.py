import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from utils import PROJECT_ROOT


METRICS = ["relevance", "personalization", "guideline_alignment", "safety"]


def load_rater_file(path, rater_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing rater file: {path}")

    df = pd.read_csv(path).copy()
    df["rater_name"] = rater_name

    required = ["output_id"] + METRICS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df


def metric_agreement_between_two_raters(rater1, rater2):
    rows = []

    merged = rater1[["output_id"] + METRICS].merge(
        rater2[["output_id"] + METRICS],
        on="output_id",
        suffixes=("_rater1", "_rater2"),
        how="inner",
    )

    for metric in METRICS:
        a = merged[f"{metric}_rater1"]
        b = merged[f"{metric}_rater2"]
        valid = pd.DataFrame({"a": a, "b": b}).dropna()

        if len(valid) > 1:
            exact_agreement = (valid["a"] == valid["b"]).mean()
            avg_abs_diff = (valid["a"] - valid["b"]).abs().mean()

            try:
                spearman_corr = spearmanr(valid["a"], valid["b"]).correlation
            except Exception:
                spearman_corr = np.nan

            try:
                weighted_kappa = cohen_kappa_score(
                    valid["a"].astype(int),
                    valid["b"].astype(int),
                    weights="quadratic",
                )
            except Exception:
                weighted_kappa = np.nan
        else:
            exact_agreement = np.nan
            avg_abs_diff = np.nan
            spearman_corr = np.nan
            weighted_kappa = np.nan

        rows.append({
            "metric": metric,
            "n_pairs": len(valid),
            "human_human_exact_agreement": exact_agreement,
            "human_human_avg_abs_diff": avg_abs_diff,
            "human_human_spearman": spearman_corr,
            "human_human_weighted_kappa": weighted_kappa,
        })

    return pd.DataFrame(rows), merged


def compute_human_average(long_df):
    avg_rows = (
        long_df
        .groupby("output_id")[METRICS]
        .mean()
        .reset_index()
    )
    return avg_rows


def compare_human_to_llm(human_avg, mapping, judge_scores):
    mapped = human_avg.merge(mapping, on="output_id", how="left")

    long_human = mapped.melt(
        id_vars=["output_id", "scenario_id", "system_name"],
        value_vars=METRICS,
        var_name="metric",
        value_name="human_avg_score",
    )

    llm = judge_scores[["scenario_id", "system_name", "metric", "score"]].copy()
    llm = llm.rename(columns={"score": "llm_judge_score"})
    llm["llm_judge_score"] = pd.to_numeric(llm["llm_judge_score"], errors="coerce")

    merged = long_human.merge(
        llm,
        on=["scenario_id", "system_name", "metric"],
        how="left",
    )

    rows = []
    for metric in METRICS:
        sub = merged[merged["metric"] == metric].dropna(subset=["human_avg_score", "llm_judge_score"])
        if len(sub) > 1:
            exact_agreement = (sub["human_avg_score"].round() == sub["llm_judge_score"].round()).mean()
            avg_abs_diff = (sub["human_avg_score"] - sub["llm_judge_score"]).abs().mean()
            try:
                spearman_corr = spearmanr(sub["human_avg_score"], sub["llm_judge_score"]).correlation
            except Exception:
                spearman_corr = np.nan
            try:
                weighted_kappa = cohen_kappa_score(
                    sub["human_avg_score"].round().astype(int),
                    sub["llm_judge_score"].round().astype(int),
                    weights="quadratic",
                )
            except Exception:
                weighted_kappa = np.nan
        else:
            exact_agreement = np.nan
            avg_abs_diff = np.nan
            spearman_corr = np.nan
            weighted_kappa = np.nan

        rows.append({
            "metric": metric,
            "n_pairs": len(sub),
            "human_llm_exact_agreement_rounded": exact_agreement,
            "human_llm_avg_abs_diff": avg_abs_diff,
            "human_llm_spearman": spearman_corr,
            "human_llm_weighted_kappa_rounded": weighted_kappa,
        })

    return merged, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater1", default="results/human_validation_rater1.csv")
    parser.add_argument("--rater2", default="results/human_validation_rater2.csv")
    parser.add_argument("--mapping", default="results/human_validation_mapping.csv")
    parser.add_argument("--judge_scores", default="results/judge_scores.csv")
    args = parser.parse_args()

    rater1_path = PROJECT_ROOT / args.rater1
    rater2_path = PROJECT_ROOT / args.rater2
    mapping_path = PROJECT_ROOT / args.mapping
    judge_scores_path = PROJECT_ROOT / args.judge_scores

    rater1 = load_rater_file(rater1_path, "rater1")
    rater2 = load_rater_file(rater2_path, "rater2")

    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
    if not judge_scores_path.exists():
        raise FileNotFoundError(f"Missing judge scores file: {judge_scores_path}")

    mapping = pd.read_csv(mapping_path).copy()
    judge_scores = pd.read_csv(judge_scores_path).copy()

    long_df = pd.concat([rater1, rater2], ignore_index=True)

    # Save long-format human ratings
    long_path = PROJECT_ROOT / "results" / "human_validation_all_raters_long.csv"
    long_df.to_csv(long_path, index=False)

    # Merge rater ratings side by side for inspection
    hh_agreement, pairwise = metric_agreement_between_two_raters(rater1, rater2)

    pairwise_path = PROJECT_ROOT / "results" / "human_validation_pairwise_raters.csv"
    pairwise.to_csv(pairwise_path, index=False)

    # Average human scores by output_id and recover scenario/system identity
    human_avg = compute_human_average(long_df)
    human_avg_mapped = human_avg.merge(mapping, on="output_id", how="left")

    merged_path = PROJECT_ROOT / "results" / "human_validation_merged.csv"
    human_avg_mapped.to_csv(merged_path, index=False)

    # Compare human average scores to LLM judge scores
    human_llm_details, human_llm_summary = compare_human_to_llm(human_avg, mapping, judge_scores)

    human_llm_details_path = PROJECT_ROOT / "results" / "human_vs_llm_details.csv"
    human_llm_details.to_csv(human_llm_details_path, index=False)

    # Combine metric-level summaries
    summary = hh_agreement.merge(human_llm_summary, on="metric", how="outer")
    summary_path = PROJECT_ROOT / "results" / "human_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Overall summary row
    overall = {
        "n_outputs": human_avg_mapped["output_id"].nunique(),
        "n_raters": 2,
        "mean_human_human_exact_agreement": hh_agreement["human_human_exact_agreement"].mean(),
        "mean_human_human_avg_abs_diff": hh_agreement["human_human_avg_abs_diff"].mean(),
        "mean_human_human_spearman": hh_agreement["human_human_spearman"].mean(),
        "mean_human_human_weighted_kappa": hh_agreement["human_human_weighted_kappa"].mean(),
        "mean_human_llm_exact_agreement_rounded": human_llm_summary["human_llm_exact_agreement_rounded"].mean(),
        "mean_human_llm_avg_abs_diff": human_llm_summary["human_llm_avg_abs_diff"].mean(),
        "mean_human_llm_spearman": human_llm_summary["human_llm_spearman"].mean(),
        "mean_human_llm_weighted_kappa_rounded": human_llm_summary["human_llm_weighted_kappa_rounded"].mean(),
    }
    overall_path = PROJECT_ROOT / "results" / "human_validation_overall_summary.csv"
    pd.DataFrame([overall]).to_csv(overall_path, index=False)

    print(f"Saved all-rater long file to {long_path}")
    print(f"Saved merged human validation file to {merged_path}")
    print(f"Saved human-vs-LLM details to {human_llm_details_path}")
    print(f"Saved metric summary to {summary_path}")
    print(f"Saved overall summary to {overall_path}")

    print("\nOverall summary:")
    print(pd.DataFrame([overall]).to_string(index=False))

    print("\nMetric-level summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
