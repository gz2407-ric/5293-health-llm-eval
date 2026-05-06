import json
import argparse
from pathlib import Path

from utils import PROJECT_ROOT


REQUIRED_SCENARIO_FIELDS = [
    "scenario_id",
    "age",
    "BMI",
    "lifestyle_habits",
    "health_goal",
    "risk_condition",
    "user_question",
    "expected_safety_warning",
    "expected_guideline_topic",
]

REQUIRED_GUIDELINE_FIELDS = [
    "chunk_id",
    "source",
    "source_title",
    "topic",
    "retrieval_keywords",
    "text",
    "url",
]


def load_jsonl(path):
    records = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}")
    return records


def check_required_fields(records, required_fields, file_label):
    errors = []
    for idx, record in enumerate(records, start=1):
        missing = [field for field in required_fields if field not in record]
        if missing:
            errors.append(f"{file_label} line {idx}: missing fields {missing}")
    return errors


def validate_scenarios(path):
    records = load_jsonl(path)
    errors = check_required_fields(records, REQUIRED_SCENARIO_FIELDS, "scenario")

    ids = [r.get("scenario_id") for r in records]
    duplicate_ids = sorted({x for x in ids if ids.count(x) > 1})
    if duplicate_ids:
        errors.append(f"Duplicate scenario_id values: {duplicate_ids}")

    for r in records:
        sid = r.get("scenario_id", "UNKNOWN")

        try:
            age = int(r.get("age"))
            if age < 18 or age > 80:
                errors.append(f"{sid}: age looks outside expected adult project scope: {age}")
        except Exception:
            errors.append(f"{sid}: age is not a valid integer")

        try:
            bmi = float(r.get("BMI"))
            if bmi < 15 or bmi > 50:
                errors.append(f"{sid}: BMI looks unusual: {bmi}")
        except Exception:
            errors.append(f"{sid}: BMI is not a valid number")

        q = str(r.get("user_question", "")).lower()
        clinical_terms = ["dosage", "dose", "prescribe", "diagnose", "diagnosis", "medicine", "medication"]
        if any(term in q for term in clinical_terms):
            errors.append(f"{sid}: user_question may be too clinical for project scope")

    return records, errors


def validate_guidelines(path):
    records = load_jsonl(path)
    errors = check_required_fields(records, REQUIRED_GUIDELINE_FIELDS, "guideline")

    ids = [r.get("chunk_id") for r in records]
    duplicate_ids = sorted({x for x in ids if ids.count(x) > 1})
    if duplicate_ids:
        errors.append(f"Duplicate chunk_id values: {duplicate_ids}")

    for r in records:
        cid = r.get("chunk_id", "UNKNOWN")
        text = str(r.get("text", "")).strip()
        if len(text.split()) < 20:
            errors.append(f"{cid}: text is very short; may not be useful for retrieval")
        if len(text.split()) > 250:
            errors.append(f"{cid}: text is long; consider splitting into smaller chunks")
        if not str(r.get("url", "")).startswith("http"):
            errors.append(f"{cid}: url is missing or does not look valid")

    return records, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()

    if args.mode == "pilot":
        scenario_path = PROJECT_ROOT / "data" / "pilot_scenarios.jsonl"
        guideline_path = PROJECT_ROOT / "guidelines" / "pilot_guideline_chunks.jsonl"
    else:
        scenario_path = PROJECT_ROOT / "data" / "scenarios.jsonl"
        guideline_path = PROJECT_ROOT / "guidelines" / "guideline_chunks.jsonl"

    print(f"Validating scenarios: {scenario_path}")
    scenario_records, scenario_errors = validate_scenarios(scenario_path)
    print(f"Scenario records: {len(scenario_records)}")

    print(f"Validating guidelines: {guideline_path}")
    guideline_records, guideline_errors = validate_guidelines(guideline_path)
    print(f"Guideline chunks: {len(guideline_records)}")

    errors = scenario_errors + guideline_errors

    if errors:
        print("\nValidation warnings/errors:")
        for err in errors:
            print(f"- {err}")
    else:
        print("\nValidation passed. No obvious format problems found.")


if __name__ == "__main__":
    main()
