import json
import pandas as pd

from utils import PROJECT_ROOT


def load_scenarios_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main():
    scenario_path = PROJECT_ROOT / "data" / "scenarios.jsonl"
    scores_path = PROJECT_ROOT / "results" / "judge_scores.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing {scenario_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")

    scenarios = load_scenarios_jsonl(scenario_path)
    scores = pd.read_csv(scores_path).dropna(subset=["score"]).copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score"])

    merged = scores.merge(
        scenarios[["scenario_id", "scenario_type"]],
        on="scenario_id",
        how="left"
    )

    breakdown = (
        merged
        .groupby(["scenario_type", "system_name", "metric"])["score"]
        .mean()
        .reset_index()
        .sort_values(["scenario_type", "metric", "system_name"])
    )

    output_path = PROJECT_ROOT / "results" / "scenario_type_breakdown.csv"
    breakdown.to_csv(output_path, index=False)

    pivot = breakdown.pivot_table(
        index=["scenario_type", "system_name"],
        columns="metric",
        values="score"
    ).reset_index()

    pivot_path = PROJECT_ROOT / "results" / "scenario_type_breakdown_wide.csv"
    pivot.to_csv(pivot_path, index=False)

    print(f"Saved scenario-type breakdown to {output_path}")
    print(f"Saved wide scenario-type breakdown to {pivot_path}")


if __name__ == "__main__":
    main()
