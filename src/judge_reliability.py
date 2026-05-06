import argparse
import json
import re
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from config import OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE
from judge_prompts import JUDGE_PROMPT_BUILDERS
from utils import PROJECT_ROOT, append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


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
    prompt = JUDGE_PROMPT_BUILDERS[metric](row_dict)

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
        "results/raw_outputs/judge_reliability_api_calls.jsonl",
        {
            "scenario_id": row_dict.get("scenario_id", ""),
            "system_name": row_dict.get("system_name", ""),
            "metric": metric,
            "judge_model": JUDGE_MODEL,
            "raw_response": raw_output,
            "parsed_response": parsed,
        }
    )

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_outputs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5293)
    args = parser.parse_args()

    outputs_path = PROJECT_ROOT / "results" / "model_outputs.csv"
    original_scores_path = PROJECT_ROOT / "results" / "judge_scores.csv"

    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing {outputs_path}")
    if not original_scores_path.exists():
        raise FileNotFoundError(f"Missing {original_scores_path}")

    outputs = pd.read_csv(outputs_path).fillna("")
    original_scores = pd.read_csv(original_scores_path).copy()

    sample = outputs.sample(n=min(args.n_outputs, len(outputs)), random_state=args.seed).copy()

    rows = []
    for _, row in sample.iterrows():
        for metric in JUDGE_PROMPT_BUILDERS.keys():
            print(f"Reliability re-judge {row.get('scenario_id')} | {row.get('system_name')} | {metric}...")
            parsed = run_judge(row, metric)

            rows.append({
                "scenario_id": row.get("scenario_id"),
                "system_name": row.get("system_name"),
                "metric": metric,
                "repeat_score": parsed.get("score"),
                "repeat_justification": parsed.get("justification", parsed.get("reason", "")),
                "judge_model": JUDGE_MODEL,
            })

    repeat = pd.DataFrame(rows)
    merged = repeat.merge(
        original_scores[["scenario_id", "system_name", "metric", "score"]],
        on=["scenario_id", "system_name", "metric"],
        how="left"
    ).rename(columns={"score": "original_score"})

    merged["original_score"] = pd.to_numeric(merged["original_score"], errors="coerce")
    merged["repeat_score"] = pd.to_numeric(merged["repeat_score"], errors="coerce")
    valid = merged.dropna(subset=["original_score", "repeat_score"]).copy()

    if len(valid) > 1:
        spearman_corr = spearmanr(valid["original_score"], valid["repeat_score"]).correlation
        exact_agreement = (valid["original_score"] == valid["repeat_score"]).mean()
        avg_abs_diff = (valid["original_score"] - valid["repeat_score"]).abs().mean()
        weighted_kappa = cohen_kappa_score(
            valid["original_score"].astype(int),
            valid["repeat_score"].astype(int),
            weights="quadratic"
        )
    else:
        spearman_corr = np.nan
        exact_agreement = np.nan
        avg_abs_diff = np.nan
        weighted_kappa = np.nan

    merged_path = PROJECT_ROOT / "results" / "judge_reliability.csv"
    summary_path = PROJECT_ROOT / "results" / "judge_reliability_summary.csv"

    merged.to_csv(merged_path, index=False)

    summary = pd.DataFrame([{
        "n_scored_pairs": len(valid),
        "exact_agreement_rate": exact_agreement,
        "average_absolute_difference": avg_abs_diff,
        "spearman_correlation": spearman_corr,
        "weighted_cohens_kappa": weighted_kappa,
        "judge_model": JUDGE_MODEL,
        "note": "This is a self-consistency check using the same judge model."
    }])
    summary.to_csv(summary_path, index=False)

    print(f"Saved judge reliability details to {merged_path}")
    print(f"Saved judge reliability summary to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
