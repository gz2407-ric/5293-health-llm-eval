from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_required_files_exist():
    required_files = [
        "README.md",
        "RESULTS.md",
        "model_data_card.md",
        "requirements.txt",
        ".env.example",
        "app/streamlit_app.py",
        "data/scenarios.jsonl",
        "guidelines/guideline_chunks.jsonl",
        "results/model_outputs.csv",
        "results/judge_scores.csv",
        "results/judge_summary.csv",
        "figures/full_guideline_alignment_by_system.png",
        "figures/full_retrieval_recall.png",
    ]

    missing = []
    for file_path in required_files:
        if not (PROJECT_ROOT / file_path).exists():
            missing.append(file_path)

    assert not missing, f"Missing required files: {missing}"


def test_scenario_file_has_50_records():
    scenario_path = PROJECT_ROOT / "data" / "scenarios.jsonl"
    scenarios = load_jsonl(scenario_path)

    assert len(scenarios) == 50, f"Expected 50 scenarios, found {len(scenarios)}"

    required_fields = [
        "scenario_id",
        "scenario_type",
        "age",
        "BMI",
        "lifestyle_habits",
        "health_goal",
        "risk_condition",
        "user_question",
        "expected_safety_warning",
        "expected_guideline_topic",
    ]

    for scenario in scenarios:
        for field in required_fields:
            assert field in scenario, f"Missing field {field} in scenario {scenario.get('scenario_id')}"


def test_guideline_file_has_50_chunks():
    guideline_path = PROJECT_ROOT / "guidelines" / "guideline_chunks.jsonl"
    chunks = load_jsonl(guideline_path)

    assert len(chunks) == 50, f"Expected 50 guideline chunks, found {len(chunks)}"

    required_fields = [
        "chunk_id",
        "source",
        "source_title",
        "topic",
        "retrieval_keywords",
        "text",
        "url",
    ]

    for chunk in chunks:
        for field in required_fields:
            assert field in chunk, f"Missing field {field} in chunk {chunk.get('chunk_id')}"


def test_main_results_have_expected_shape():
    outputs_path = PROJECT_ROOT / "results" / "model_outputs.csv"
    outputs = pd.read_csv(outputs_path)

    assert len(outputs) == 200, f"Expected 200 model outputs, found {len(outputs)}"

    expected_systems = {
        "baseline",
        "rag_only",
        "multistep_only",
        "full_pipeline",
    }

    actual_systems = set(outputs["system_name"].unique())
    assert expected_systems.issubset(actual_systems), f"Missing systems: {expected_systems - actual_systems}"


def test_judge_summary_exists_and_has_metrics():
    summary_path = PROJECT_ROOT / "results" / "judge_summary.csv"
    summary = pd.read_csv(summary_path)

    expected_metrics = {
        "relevance",
        "personalization",
        "guideline_alignment",
        "safety",
    }

    actual_metrics = set(summary["metric"].unique())
    assert expected_metrics.issubset(actual_metrics), f"Missing metrics: {expected_metrics - actual_metrics}"


if __name__ == "__main__":
    test_required_files_exist()
    test_scenario_file_has_50_records()
    test_guideline_file_has_50_chunks()
    test_main_results_have_expected_shape()
    test_judge_summary_exists_and_has_metrics()
    print("All lightweight project checks passed.")