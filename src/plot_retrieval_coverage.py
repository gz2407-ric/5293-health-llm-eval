import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    input_path = PROJECT_ROOT / "results" / "retrieval_summary.csv"
    output_path = PROJECT_ROOT / "figures" / "full_retrieval_coverage.png"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_csv(input_path)

    # Make system labels readable
    label_map = {
        "rag_only": "RAG only",
        "full_pipeline": "Full pipeline"
    }

    df["system_label"] = df["system_name"].map(label_map)

    # Keep a fixed order
    order = ["RAG only", "Full pipeline"]
    df["system_label"] = pd.Categorical(df["system_label"], categories=order, ordered=True)
    df = df.sort_values("system_label")

    plot_df = df[[
        "system_label",
        "mean_coverage_rate_at_3",
        "mean_coverage_rate_at_5"
    ]].copy()

    plot_df = plot_df.rename(columns={
        "system_label": "System",
        "mean_coverage_rate_at_3": "Coverage@3",
        "mean_coverage_rate_at_5": "Coverage@5"
    })

    ax = plot_df.plot(
        x="System",
        y=["Coverage@3", "Coverage@5"],
        kind="bar",
        figsize=(8, 5)
    )

    ax.set_title("Retrieval Coverage by System")
    ax.set_xlabel("System")
    ax.set_ylabel("Average Topic Coverage")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved retrieval coverage figure to {output_path}")


if __name__ == "__main__":
    main()