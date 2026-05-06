import json

from multistep_only import run_multistep_only
from safety_classifier import risk_flags_to_text
from utils import PROJECT_ROOT, load_jsonl, scenario_to_profile_text, write_csv


def main():
    scenario_path = PROJECT_ROOT / "data" / "pilot_scenarios.jsonl"
    output_path = PROJECT_ROOT / "results" / "pilot_multistep_outputs.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Cannot find {scenario_path}. Please put pilot_scenarios.jsonl under data/."
        )

    scenarios = load_jsonl(scenario_path)
    rows = []

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        print(f"Running multi-step-only for {scenario_id}...")

        result = run_multistep_only(scenario)

        expected_topic = scenario.get("expected_guideline_topic", "")
        if isinstance(expected_topic, list):
            expected_topic = "; ".join(expected_topic)

        rows.append(
            {
                "scenario_id": scenario_id,
                "system_name": "multistep_only",
                "user_profile": scenario_to_profile_text(scenario),
                "user_question": scenario.get("user_question", ""),
                "expected_safety_warning": scenario.get("expected_safety_warning", ""),
                "expected_guideline_topic": expected_topic,
                "risk_flags": json.dumps(result["risk_flags"], ensure_ascii=False),
                "active_risk_flags": risk_flags_to_text(result["risk_flags"]),
                "initial_output": result["initial_output"],
                "safety_classification": result["safety_result"].get("classification", ""),
                "safety_reason": result["safety_result"].get("reason", ""),
                "safety_required_fix": result["safety_result"].get("required_fix", ""),
                "revised": result["revised"],
                "model_output": result["final_output"],
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
            "risk_flags",
            "active_risk_flags",
            "initial_output",
            "safety_classification",
            "safety_reason",
            "safety_required_fix",
            "revised",
            "model_output",
        ],
    )

    print(f"Done. Saved results to {output_path}")


if __name__ == "__main__":
    main()
