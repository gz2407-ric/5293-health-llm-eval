import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils import PROJECT_ROOT, ensure_dir


SYSTEM_ORDER = ["baseline", "rag_only", "multistep_only", "full_pipeline"]
SYSTEM_LABELS = {
    "baseline": "Baseline",
    "rag_only": "RAG only",
    "multistep_only": "Multi-step only",
    "full_pipeline": "Full pipeline",
}

METRIC_LABELS = {
    "relevance": "Relevance",
    "personalization": "Personalization",
    "guideline_alignment": "Guideline Alignment",
    "safety": "Safety",
}


def apply_system_order(df):
    df = df.copy()
    df["system_name"] = pd.Categorical(df["system_name"], categories=SYSTEM_ORDER, ordered=True)
    return df.sort_values("system_name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    args = parser.parse_args()

    if args.mode == "pilot":
        score_path = PROJECT_ROOT / "results" / "pilot_judge_scores_unified.csv"
        figure_prefix = "pilot"
    else:
        score_path = PROJECT_ROOT / "results" / "judge_scores.csv"
        figure_prefix = "full"

    if not score_path.exists():
        raise FileNotFoundError(f"Missing score file: {score_path}")

    fig_dir = PROJECT_ROOT / "figures"
    ensure_dir(fig_dir)

    scores = pd.read_csv(score_path).dropna(subset=["score"])
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score"])

    summary = (
        scores
        .groupby(["system_name", "metric"], observed=False)["score"]
        .mean()
        .reset_index()
    )
    summary = apply_system_order(summary)
    summary["system_label"] = summary["system_name"].map(SYSTEM_LABELS)

    # Figure 1: average score by system and metric
    pivot = summary.pivot(index="system_label", columns="metric", values="score")
    ordered_labels = [SYSTEM_LABELS[s] for s in SYSTEM_ORDER if SYSTEM_LABELS[s] in pivot.index]
    ordered_metrics = [m for m in ["relevance", "personalization", "guideline_alignment", "safety"] if m in pivot.columns]
    pivot = pivot.loc[ordered_labels, ordered_metrics]
    pivot.columns = [METRIC_LABELS.get(c, c) for c in pivot.columns]

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Average Judge Scores by System")
    ax.set_xlabel("System")
    ax.set_ylabel("Average Score")
    ax.set_ylim(2.5, 5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out1 = fig_dir / f"{figure_prefix}_average_scores_by_system.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # One chart per metric
    for metric in ordered_metrics:
        metric_summary = summary[summary["metric"] == metric].copy()
        metric_summary = apply_system_order(metric_summary)
        metric_summary["system_label"] = metric_summary["system_name"].map(SYSTEM_LABELS)

        ax = metric_summary.plot(
            x="system_label",
            y="score",
            kind="bar",
            legend=False,
            figsize=(8, 5),
        )
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} Score by System")
        ax.set_xlabel("System")
        ax.set_ylabel("Average Score")
        ax.set_ylim(2.5, 5)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out = fig_dir / f"{figure_prefix}_{metric}_by_system.png"
        plt.savefig(out, dpi=200)
        plt.close()

    # Retrieval plot if available
    retrieval_path = PROJECT_ROOT / "results" / "retrieval_summary.csv"
    if args.mode == "full" and retrieval_path.exists():
        retrieval = pd.read_csv(retrieval_path)
        retrieval["system_name"] = pd.Categorical(retrieval["system_name"], categories=SYSTEM_ORDER, ordered=True)
        retrieval = retrieval.sort_values("system_name")
        retrieval["system_label"] = retrieval["system_name"].map(SYSTEM_LABELS)

        ax = retrieval.plot(
            x="system_label",
            y=["recall_at_3", "recall_at_5"],
            kind="bar",
            figsize=(8, 5),
        )
        ax.set_title("Retrieval Recall by System")
        ax.set_xlabel("System")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out = fig_dir / "full_retrieval_recall.png"
        plt.savefig(out, dpi=200)
        plt.close()

    # Output length plot if available
    length_path = PROJECT_ROOT / "results" / "output_length_analysis.csv"
    if args.mode == "full" and length_path.exists():
        length = pd.read_csv(length_path)
        length["system_name"] = pd.Categorical(length["system_name"], categories=SYSTEM_ORDER, ordered=True)
        length = length.sort_values("system_name")
        length["system_label"] = length["system_name"].map(SYSTEM_LABELS)

        ax = length.plot(
            x="system_label",
            y="mean_word_count",
            kind="bar",
            legend=False,
            figsize=(8, 5),
        )
        ax.set_title("Average Output Length by System")
        ax.set_xlabel("System")
        ax.set_ylabel("Mean Word Count")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out = fig_dir / "full_output_length_by_system.png"
        plt.savefig(out, dpi=200)
        plt.close()

    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
