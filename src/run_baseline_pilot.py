from pathlib import Path

from baseline import run_baseline
from utils import PROJECT_ROOT, load_jsonl, scenario_to_profile_text, write_csv


def main():
    scenario_path = PROJECT_ROOT / "data" / "pilot_scenarios.jsonl"
    output_path = PROJECT_ROOT / "results" / "pilot_baseline_outputs.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Cannot find {scenario_path}. Please put pilot_scenarios.jsonl under the data/ folder."
        )

    scenarios = load_jsonl(scenario_path)
    rows = []

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        print(f"Running baseline for {scenario_id}...")

        model_output = run_baseline(scenario)

        rows.append(
            {
                "scenario_id": scenario_id,
                "system_name": "baseline",
                "user_profile": scenario_to_profile_text(scenario),
                "user_question": scenario.get("user_question", ""),
                "expected_safety_warning": scenario.get("expected_safety_warning", ""),
                "expected_guideline_topic": "; ".join(scenario.get("expected_guideline_topic", []))
                if isinstance(scenario.get("expected_guideline_topic", []), list)
                else scenario.get("expected_guideline_topic", ""),
                "model_output": model_output,
            }
        )

    write_csv(
        output_path,
        rows,
        fieldnames=[
            "scenario_id",
            "system_name",
            "user_profile",
            "user_question",
            "expected_safety_warning",
            "expected_guideline_topic",
            "model_output",
        ],
    )

    print(f"Done. Saved results to {output_path}")


if __name__ == "__main__":
    main()
