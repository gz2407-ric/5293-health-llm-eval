import argparse
import json
import re

import pandas as pd
from openai import OpenAI

from config import OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE
from judge_prompts import JUDGE_PROMPT_BUILDERS
from utils import PROJECT_ROOT, append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


def get_paths(mode):
    if mode == "pilot":
        input_path = PROJECT_ROOT / "results" / "pilot_all_outputs_unified.csv"
        output_path = PROJECT_ROOT / "results" / "pilot_judge_scores_unified.csv"
        summary_path = PROJECT_ROOT / "results" / "pilot_judge_summary_unified.csv"
    else:
        input_path = PROJECT_ROOT / "results" / "model_outputs.csv"
        output_path = PROJECT_ROOT / "results" / "judge_scores.csv"
        summary_path = PROJECT_ROOT / "results" / "judge_summary.csv"

    return input_path, output_path, summary_path


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
        "justification": "Could not parse judge JSON output.",
        "raw_output": text,
    }


def run_judge(row, metric):
    row_dict = row.to_dict()
    prompt_builder = JUDGE_PROMPT_BUILDERS[metric]
    prompt = prompt_builder(row_dict)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful evaluator. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
    )

    raw_output = response.choices[0].message.content
    parsed = extract_json(raw_output)

    append_jsonl(
        "results/raw_outputs/judge_api_calls.jsonl",
        {
            "scenario_id": row_dict.get("scenario_id", ""),
            "system_name": row_dict.get("system_name", ""),
            "metric": metric,
            "judge_model": JUDGE_MODEL,
            "prompt": prompt,
            "raw_response": raw_output,
            "parsed_response": parsed,
        }
    )

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to score")
    args = parser.parse_args()

    input_path, output_path, summary_path = get_paths(args.mode)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input outputs file: {input_path}")

    outputs = pd.read_csv(input_path).fillna("")
    if args.limit is not None:
        outputs = outputs.iloc[: args.limit].copy()

    score_rows = []

    for _, row in outputs.iterrows():
        scenario_id = row.get("scenario_id", "")
        system_name = row.get("system_name", "")

        for metric in JUDGE_PROMPT_BUILDERS.keys():
            print(f"Judging {scenario_id} | {system_name} | {metric}...")
            parsed = run_judge(row, metric)

            score_rows.append(
                {
                    "scenario_id": scenario_id,
                    "system_name": system_name,
                    "metric": metric,
                    "score": parsed.get("score"),
                    "unsafe_flag": parsed.get("unsafe_flag", ""),
                    "justification": parsed.get("justification", parsed.get("reason", "")),
                    "judge_model": JUDGE_MODEL,
                }
            )

    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(output_path, index=False)

    summary = (
        score_df
        .dropna(subset=["score"])
        .groupby(["system_name", "metric"])["score"]
        .mean()
        .reset_index()
        .sort_values(["metric", "system_name"])
    )
    summary.to_csv(summary_path, index=False)

    print(f"Done. Saved judge scores to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
