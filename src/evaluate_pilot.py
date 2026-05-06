import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

from config import OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE
from judge_prompts import JUDGE_PROMPT_BUILDERS
from utils import PROJECT_ROOT, append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


PILOT_OUTPUT_FILES = [
    "pilot_baseline_outputs.csv",
    "pilot_rag_outputs.csv",
    "pilot_multistep_outputs.csv",
    "pilot_full_outputs.csv",
]


def load_all_pilot_outputs():
    frames = []
    for filename in PILOT_OUTPUT_FILES:
        path = PROJECT_ROOT / "results" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Please run all four pilot systems first.")
        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.fillna("")

    output_path = PROJECT_ROOT / "results" / "pilot_all_outputs.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved combined pilot outputs to {output_path}")

    return combined


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
    outputs = load_all_pilot_outputs()
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
    output_path = PROJECT_ROOT / "results" / "pilot_judge_scores.csv"
    score_df.to_csv(output_path, index=False)

    summary = (
        score_df
        .dropna(subset=["score"])
        .groupby(["system_name", "metric"])["score"]
        .mean()
        .reset_index()
        .sort_values(["metric", "system_name"])
    )
    summary_path = PROJECT_ROOT / "results" / "pilot_judge_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Done. Saved judge scores to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
