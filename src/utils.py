import json
import csv
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    """Load a JSONL file into a list of dictionaries."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def scenario_to_profile_text(scenario):
    """Convert one scenario dictionary into a readable user profile."""
    bmi = scenario.get("bmi", scenario.get("BMI", "NA"))
    risk_condition = scenario.get("risk_condition", "none")
    if isinstance(risk_condition, list):
        risk_condition = ", ".join(risk_condition)

    return f"""
Age: {scenario.get("age", "NA")}
BMI: {bmi}
Lifestyle habits: {scenario.get("lifestyle_habits", "NA")}
Health goal: {scenario.get("health_goal", "NA")}
Risk condition: {risk_condition}
User question: {scenario.get("user_question", "NA")}
""".strip()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path, record):
    path = Path(path)
    ensure_dir(path.parent)
    record = dict(record)
    record["logged_at"] = datetime.utcnow().isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
