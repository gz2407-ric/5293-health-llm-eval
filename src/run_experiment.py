import argparse
import json

from baseline import run_baseline
from rag_only import run_rag_only
from multistep_only import run_multistep_only
from full_pipeline import run_full_pipeline
from safety_classifier import risk_flags_to_text
from run_rag_pilot import summarize_retrieved_chunks
from utils import PROJECT_ROOT, load_jsonl, scenario_to_profile_text, write_csv


def get_paths(mode):
    if mode == "pilot":
        scenario_path = PROJECT_ROOT / "data" / "pilot_scenarios.jsonl"
        guideline_path = PROJECT_ROOT / "guidelines" / "pilot_guideline_chunks.jsonl"
        output_path = PROJECT_ROOT / "results" / "pilot_all_outputs_unified.csv"
    else:
        scenario_path = PROJECT_ROOT / "data" / "scenarios.jsonl"
        guideline_path = PROJECT_ROOT / "guidelines" / "guideline_chunks.jsonl"
        output_path = PROJECT_ROOT / "results" / "model_outputs.csv"

    return scenario_path, guideline_path, output_path


def expected_topic_text(scenario):
    expected_topic = scenario.get("expected_guideline_topic", "")
    if isinstance(expected_topic, list):
        expected_topic = "; ".join(expected_topic)
    return expected_topic


def base_row(scenario, system_name):
    return {
        "scenario_id": scenario.get("scenario_id"),
        "system_name": system_name,
        "user_profile": scenario_to_profile_text(scenario),
        "user_question": scenario.get("user_question", ""),
        "expected_safety_warning": scenario.get("expected_safety_warning", ""),
        "expected_guideline_topic": expected_topic_text(scenario),
        "risk_flags": "",
        "active_risk_flags": "",
        "retrieved_chunk_ids": "",
        "retrieved_topics": "",
        "retrieved_guidelines": "",
        "initial_output": "",
        "safety_classification": "",
        "safety_reason": "",
        "safety_required_fix": "",
        "revised": "",
        "model_output": "",
    }


def run_one_scenario_all_systems(scenario, guideline_path):
    rows = []
    scenario_id = scenario.get("scenario_id")

    print(f"Running Baseline for {scenario_id}...")
    baseline_output = run_baseline(scenario)
    row = base_row(scenario, "baseline")
    row["model_output"] = baseline_output
    rows.append(row)

    print(f"Running RAG-only for {scenario_id}...")
    rag_output, rag_chunks = run_rag_only(scenario, guideline_path=guideline_path)
    row = base_row(scenario, "rag_only")
    row["retrieved_chunk_ids"] = "; ".join([str(c.get("chunk_id", "")) for c in rag_chunks])
    row["retrieved_topics"] = "; ".join([str(c.get("topic", "")) for c in rag_chunks])
    row["retrieved_guidelines"] = summarize_retrieved_chunks(rag_chunks)
    row["model_output"] = rag_output
    rows.append(row)

    print(f"Running Multi-step-only for {scenario_id}...")
    multistep_result = run_multistep_only(scenario)
    row = base_row(scenario, "multistep_only")
    row["risk_flags"] = json.dumps(multistep_result["risk_flags"], ensure_ascii=False)
    row["active_risk_flags"] = risk_flags_to_text(multistep_result["risk_flags"])
    row["initial_output"] = multistep_result["initial_output"]
    row["safety_classification"] = multistep_result["safety_result"].get("classification", "")
    row["safety_reason"] = multistep_result["safety_result"].get("reason", "")
    row["safety_required_fix"] = multistep_result["safety_result"].get("required_fix", "")
    row["revised"] = multistep_result["revised"]
    row["model_output"] = multistep_result["final_output"]
    rows.append(row)

    print(f"Running Full Pipeline for {scenario_id}...")
    full_result = run_full_pipeline(scenario, guideline_path=guideline_path)
    full_chunks = full_result["retrieved_chunks"]
    row = base_row(scenario, "full_pipeline")
    row["risk_flags"] = json.dumps(full_result["risk_flags"], ensure_ascii=False)
    row["active_risk_flags"] = risk_flags_to_text(full_result["risk_flags"])
    row["retrieved_chunk_ids"] = "; ".join([str(c.get("chunk_id", "")) for c in full_chunks])
    row["retrieved_topics"] = "; ".join([str(c.get("topic", "")) for c in full_chunks])
    row["retrieved_guidelines"] = summarize_retrieved_chunks(full_chunks)
    row["initial_output"] = full_result["initial_output"]
    row["safety_classification"] = full_result["safety_result"].get("classification", "")
    row["safety_reason"] = full_result["safety_result"].get("reason", "")
    row["safety_required_fix"] = full_result["safety_result"].get("required_fix", "")
    row["revised"] = full_result["revised"]
    row["model_output"] = full_result["final_output"]
    rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of scenarios to run")
    args = parser.parse_args()

    scenario_path, guideline_path, output_path = get_paths(args.mode)

    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario file: {scenario_path}")
    if not guideline_path.exists():
        raise FileNotFoundError(f"Missing guideline file: {guideline_path}")

    scenarios = load_jsonl(scenario_path)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    all_rows = []
    for scenario in scenarios:
        all_rows.extend(run_one_scenario_all_systems(scenario, guideline_path=guideline_path))

    fieldnames = [
        "scenario_id",
        "system_name",
        "user_profile",
        "user_question",
        "expected_safety_warning",
        "expected_guideline_topic",
        "risk_flags",
        "active_risk_flags",
        "retrieved_chunk_ids",
        "retrieved_topics",
        "retrieved_guidelines",
        "initial_output",
        "safety_classification",
        "safety_reason",
        "safety_required_fix",
        "revised",
        "model_output",
    ]

    write_csv(output_path, all_rows, fieldnames=fieldnames)
    print(f"Done. Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
