import argparse
import json
import re
from copy import deepcopy

import pandas as pd
from openai import OpenAI

from config import OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE
from full_pipeline import run_full_pipeline
from utils import PROJECT_ROOT, append_jsonl, write_csv


client = OpenAI(api_key=OPENAI_API_KEY)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def scenario_text(s):
    risk = s.get("risk_condition", "")
    if isinstance(risk, list):
        risk = ", ".join(risk)
    return (
        f"Age: {s.get('age')}\n"
        f"BMI: {s.get('BMI')}\n"
        f"Lifestyle habits: {s.get('lifestyle_habits')}\n"
        f"Health goal: {s.get('health_goal')}\n"
        f"Risk condition: {risk}\n"
        f"User question: {s.get('user_question')}"
    )


def find_first_by_type(scenarios, scenario_type):
    for s in scenarios:
        if s.get("scenario_type") == scenario_type:
            return s
    return None


def build_counterfactual_cases(scenarios):
    """
    Predefined counterfactual rules.
    Meaningful health changes modify BMI, risk condition, lifestyle, or goal.
    Negative controls modify irrelevant details embedded in lifestyle_habits.
    """
    cases = []

    meaningful_specs = [
        ("overweight_obesity", "BMI 34 to BMI 23; sedentary to active; weight loss to general wellness"),
        ("hypertension", "hypertension to no known condition; high sodium/stress lifestyle to balanced lifestyle"),
        ("prediabetes_diabetes_risk", "prediabetes risk to no known condition; blood sugar goal to general wellness"),
        ("injury_limited_mobility", "injury/limited mobility to no mobility limitation"),
        ("pregnancy_special_caution", "pregnancy/special caution to non-pregnant general adult wellness profile"),
    ]

    for scenario_type, rule in meaningful_specs:
        original = find_first_by_type(scenarios, scenario_type)
        if original is None:
            continue

        cf = deepcopy(original)
        cf["scenario_id"] = original["scenario_id"] + "_CF_MEANINGFUL"
        cf["counterfactual_type"] = "meaningful_health_change"
        cf["counterfactual_rule"] = rule

        if scenario_type == "overweight_obesity":
            cf["BMI"] = 23.0
            cf["lifestyle_habits"] = "active lifestyle, regular walking and strength training, cooks balanced meals at home"
            cf["health_goal"] = "maintain general wellness and fitness"
            cf["risk_condition"] = ["none"]
            cf["user_question"] = "How can I maintain my current healthy routine and avoid unnecessary diet changes?"

        elif scenario_type == "hypertension":
            cf["risk_condition"] = ["none"]
            cf["lifestyle_habits"] = "moderately active, balanced diet, normal stress level, sleeps 7-8 hours"
            cf["health_goal"] = "maintain heart health and overall wellness"
            cf["user_question"] = "What general lifestyle habits can help me maintain long-term wellness?"

        elif scenario_type == "prediabetes_diabetes_risk":
            cf["risk_condition"] = ["none"]
            cf["BMI"] = 22.5
            cf["lifestyle_habits"] = "active, balanced meals, limited sugary drinks"
            cf["health_goal"] = "maintain energy and general wellness"
            cf["user_question"] = "What everyday habits can help me maintain a healthy lifestyle?"

        elif scenario_type == "injury_limited_mobility":
            cf["risk_condition"] = ["none"]
            cf["lifestyle_habits"] = "active, no current pain, does moderate exercise several times per week"
            cf["health_goal"] = "improve general fitness"
            cf["user_question"] = "How can I improve my fitness routine safely?"

        elif scenario_type == "pregnancy_special_caution":
            cf["risk_condition"] = ["none"]
            cf["lifestyle_habits"] = "not pregnant, light activity, balanced meals, no special medical condition"
            cf["health_goal"] = "maintain general health and fitness"
            cf["user_question"] = "What general exercise and diet habits should I follow?"

        cases.append((original, cf))

    # Negative controls: pick first five scenarios and alter irrelevant details only.
    for original in scenarios[:5]:
        cf = deepcopy(original)
        cf["scenario_id"] = original["scenario_id"] + "_CF_NEGCTRL"
        cf["counterfactual_type"] = "negative_control"
        cf["counterfactual_rule"] = "Changed irrelevant details only: favorite color/city/hobby."

        lifestyle = str(cf.get("lifestyle_habits", ""))
        cf["lifestyle_habits"] = (
            lifestyle
            + " Irrelevant detail: favorite color changed from blue to green; "
              "city changed from Boston to Seattle; hobby changed from reading to painting."
        )

        cases.append((original, cf))

    return cases


def extract_json(text):
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "score": None,
        "justification": "Could not parse JSON.",
        "raw_output": text,
    }


def build_counterfactual_judge_prompt(original_profile, counterfactual_profile, original_output, counterfactual_output, counterfactual_type):
    return f"""
You are evaluating whether a lifestyle health recommendation system responds appropriately to changes in a user's profile.

Original user profile:
{original_profile}

Counterfactual user profile:
{counterfactual_profile}

Original recommendation:
{original_output}

Counterfactual recommendation:
{counterfactual_output}

Counterfactual type:
{counterfactual_type}

Evaluation task:
If the counterfactual type is meaningful_health_change, evaluate whether the recommendation changes in an appropriate way based on the changed health profile.

If the counterfactual type is negative_control, evaluate whether the recommendation remains stable when only irrelevant details are changed.

Score from 1 to 5:
1 = Very poor. The system responds in the wrong direction or changes/stays the same inappropriately.
2 = Weak. The system shows limited or inconsistent response to the counterfactual change.
3 = Moderate. The response is partly appropriate but not fully convincing.
4 = Good. The system responds appropriately to the counterfactual change.
5 = Excellent. The system clearly distinguishes meaningful health changes from irrelevant changes.

Return your answer in this exact JSON format:
{{
  "score": integer from 1 to 5,
  "justification": "one or two sentences explaining the score"
}}
""".strip()


def run_counterfactual_judge(original_profile, counterfactual_profile, original_output, counterfactual_output, counterfactual_type):
    prompt = build_counterfactual_judge_prompt(
        original_profile,
        counterfactual_profile,
        original_output,
        counterfactual_output,
        counterfactual_type,
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful evaluator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )

    raw_output = response.choices[0].message.content
    parsed = extract_json(raw_output)

    append_jsonl(
        "results/raw_outputs/counterfactual_judge_api_calls.jsonl",
        {
            "counterfactual_type": counterfactual_type,
            "prompt": prompt,
            "raw_response": raw_output,
            "parsed_response": parsed,
        }
    )

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_cases", type=int, default=10)
    args = parser.parse_args()

    scenario_path = PROJECT_ROOT / "data" / "scenarios.jsonl"
    outputs_path = PROJECT_ROOT / "results" / "model_outputs.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing {scenario_path}")
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing {outputs_path}")

    scenarios = load_jsonl(scenario_path)
    outputs = pd.read_csv(outputs_path).fillna("")

    full_outputs = outputs[outputs["system_name"] == "full_pipeline"].copy()
    original_output_map = dict(zip(full_outputs["scenario_id"], full_outputs["model_output"]))

    cases = build_counterfactual_cases(scenarios)[: args.max_cases]

    rows = []
    for i, (original, cf) in enumerate(cases, start=1):
        original_id = original["scenario_id"]
        cf_type = cf["counterfactual_type"]
        print(f"Running counterfactual case {i}: {original_id} | {cf_type}")

        original_output = original_output_map.get(original_id, "")
        if not original_output:
            raise ValueError(f"Could not find original full_pipeline output for {original_id}")

        cf_result = run_full_pipeline(cf)
        cf_output = cf_result["final_output"]

        original_profile = scenario_text(original)
        cf_profile = scenario_text(cf)

        judge = run_counterfactual_judge(
            original_profile=original_profile,
            counterfactual_profile=cf_profile,
            original_output=original_output,
            counterfactual_output=cf_output,
            counterfactual_type=cf_type,
        )

        rows.append({
            "case_id": f"CF{i:02d}",
            "original_scenario_id": original_id,
            "counterfactual_scenario_id": cf["scenario_id"],
            "counterfactual_type": cf_type,
            "counterfactual_rule": cf.get("counterfactual_rule", ""),
            "original_profile": original_profile,
            "counterfactual_profile": cf_profile,
            "original_output": original_output,
            "counterfactual_output": cf_output,
            "judge_score": judge.get("score"),
            "judge_justification": judge.get("justification", ""),
        })

    output_path = PROJECT_ROOT / "results" / "counterfactual_results.csv"
    write_csv(output_path, rows, fieldnames=list(rows[0].keys()) if rows else [])
    print(f"Saved counterfactual results to {output_path}")

    if rows:
        summary = pd.DataFrame(rows)
        summary["judge_score"] = pd.to_numeric(summary["judge_score"], errors="coerce")
        summary_out = (
            summary.groupby("counterfactual_type")["judge_score"]
            .agg(["count", "mean"])
            .reset_index()
        )
        summary_path = PROJECT_ROOT / "results" / "counterfactual_summary.csv"
        summary_out.to_csv(summary_path, index=False)
        print(f"Saved counterfactual summary to {summary_path}")
        print(summary_out.to_string(index=False))


if __name__ == "__main__":
    main()
