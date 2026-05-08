import argparse
import json
import random
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from openai import OpenAI

from config import OPENAI_API_KEY, GENERATION_MODEL, JUDGE_MODEL, TEMPERATURE, TOP_K
from utils import PROJECT_ROOT, load_jsonl, scenario_to_profile_text, write_csv, append_jsonl
from retrieval import load_guideline_chunks
from rag_only import build_rag_prompt
from judge_prompts import JUDGE_PROMPT_BUILDERS


client = OpenAI(api_key=OPENAI_API_KEY)


def sample_random_chunks(chunks, top_k=TOP_K, seed=5293, scenario_id=""):
    """Sample random guideline chunks with a deterministic seed per scenario."""
    rng = random.Random(f"{seed}_{scenario_id}")
    if len(chunks) <= top_k:
        selected = list(chunks)
    else:
        selected = rng.sample(chunks, top_k)

    selected = [dict(c) for c in selected]
    for c in selected:
        c["similarity_score"] = None
        c["retrieval_text"] = c.get("text", "")
    return selected


def summarize_retrieved_chunks(chunks):
    parts = []
    for chunk in chunks:
        parts.append(
            f"{chunk.get('chunk_id', '')} | "
            f"{chunk.get('topic', '')} | "
            f"{chunk.get('text', '')[:250].replace(chr(10), ' ')}"
        )
    return "\n---\n".join(parts)


def generate_random_rag_outputs(seed=5293):
    scenario_path = PROJECT_ROOT / "data" / "scenarios.jsonl"
    guideline_path = PROJECT_ROOT / "guidelines" / "guideline_chunks.jsonl"
    output_path = PROJECT_ROOT / "results" / "random_rag_outputs.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing {scenario_path}")
    if not guideline_path.exists():
        raise FileNotFoundError(f"Missing {guideline_path}")

    scenarios = load_jsonl(scenario_path)
    chunks = load_guideline_chunks(guideline_path)

    rows = []
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", "")
        print(f"Running Random RAG for {scenario_id}...")

        random_chunks = sample_random_chunks(chunks, top_k=TOP_K, seed=seed, scenario_id=scenario_id)
        prompt = build_rag_prompt(scenario, random_chunks)

        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You generate safe, guideline-grounded, non-clinical lifestyle health recommendations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
        )

        output = response.choices[0].message.content

        expected_topic = scenario.get("expected_guideline_topic", "")
        if isinstance(expected_topic, list):
            expected_topic = "; ".join(expected_topic)

        rows.append({
            "scenario_id": scenario_id,
            "system_name": "random_rag",
            "user_profile": scenario_to_profile_text(scenario),
            "user_question": scenario.get("user_question", ""),
            "expected_safety_warning": scenario.get("expected_safety_warning", ""),
            "expected_guideline_topic": expected_topic,
            "retrieved_chunk_ids": "; ".join([str(c.get("chunk_id", "")) for c in random_chunks]),
            "retrieved_topics": "; ".join([str(c.get("topic", "")) for c in random_chunks]),
            "retrieved_guidelines": summarize_retrieved_chunks(random_chunks),
            "model_output": output,
        })

        append_jsonl(
            "results/raw_outputs/random_rag_api_calls.jsonl",
            {
                "scenario_id": scenario_id,
                "system_name": "random_rag",
                "model": GENERATION_MODEL,
                "random_chunk_ids": [c.get("chunk_id") for c in random_chunks],
                "random_topics": [c.get("topic") for c in random_chunks],
                "prompt": prompt,
                "response": output,
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
            "retrieved_chunk_ids",
            "retrieved_topics",
            "retrieved_guidelines",
            "model_output",
        ],
    )

    print(f"Saved Random RAG outputs to {output_path}")


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
        "results/raw_outputs/random_rag_judge_api_calls.jsonl",
        {
            "scenario_id": row_dict.get("scenario_id", ""),
            "system_name": "random_rag",
            "metric": metric,
            "judge_model": JUDGE_MODEL,
            "raw_response": raw_output,
            "parsed_response": parsed,
        }
    )

    return parsed


def evaluate_random_rag(all_metrics=True):
    input_path = PROJECT_ROOT / "results" / "random_rag_outputs.csv"
    output_path = PROJECT_ROOT / "results" / "random_rag_judge_scores.csv"
    summary_path = PROJECT_ROOT / "results" / "random_rag_judge_summary.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run --step generate first.")

    outputs = pd.read_csv(input_path).fillna("")
    metrics = list(JUDGE_PROMPT_BUILDERS.keys()) if all_metrics else ["guideline_alignment"]

    rows = []
    for _, row in outputs.iterrows():
        scenario_id = row.get("scenario_id", "")
        for metric in metrics:
            print(f"Judging Random RAG {scenario_id} | {metric}...")
            parsed = run_judge(row, metric)
            rows.append({
                "scenario_id": scenario_id,
                "system_name": "random_rag",
                "metric": metric,
                "score": parsed.get("score"),
                "unsafe_flag": parsed.get("unsafe_flag", ""),
                "justification": parsed.get("justification", parsed.get("reason", "")),
                "judge_model": JUDGE_MODEL,
            })

    scores = pd.DataFrame(rows)
    scores.to_csv(output_path, index=False)

    summary = (
        scores
        .dropna(subset=["score"])
        .groupby(["system_name", "metric"])["score"]
        .mean()
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved Random RAG judge scores to {output_path}")
    print(f"Saved Random RAG judge summary to {summary_path}")
    print(summary.to_string(index=False))


def paired_compare_random_rag():
    random_scores_path = PROJECT_ROOT / "results" / "random_rag_judge_scores.csv"
    main_scores_path = PROJECT_ROOT / "results" / "judge_scores.csv"
    output_path = PROJECT_ROOT / "results" / "random_rag_ablation_summary.csv"

    if not random_scores_path.exists():
        raise FileNotFoundError(f"Missing {random_scores_path}. Run --step evaluate first.")
    if not main_scores_path.exists():
        raise FileNotFoundError(f"Missing {main_scores_path}.")

    random_scores = pd.read_csv(random_scores_path).dropna(subset=["score"]).copy()
    main_scores = pd.read_csv(main_scores_path).dropna(subset=["score"]).copy()
    random_scores["score"] = pd.to_numeric(random_scores["score"], errors="coerce")
    main_scores["score"] = pd.to_numeric(main_scores["score"], errors="coerce")

    combined = pd.concat([main_scores, random_scores], ignore_index=True)
    metrics = sorted(random_scores["metric"].unique())

    comparisons = [
        ("baseline", "random_rag"),
        ("random_rag", "rag_only"),
        ("random_rag", "full_pipeline"),
    ]

    rows = []
    for metric in metrics:
        metric_df = combined[combined["metric"] == metric]
        wide = metric_df.pivot_table(index="scenario_id", columns="system_name", values="score", aggfunc="mean")

        for a, b in comparisons:
            if a not in wide.columns or b not in wide.columns:
                continue

            paired = wide[[a, b]].dropna()
            if len(paired) < 2:
                continue

            x = paired[a].to_numpy(dtype=float)
            y = paired[b].to_numpy(dtype=float)
            diff = y - x

            if np.all(diff == 0):
                stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    stat, p = np.nan, np.nan

            rows.append({
                "metric": metric,
                "comparison": f"{a} vs {b}",
                "n_pairs": len(paired),
                "mean_score_a": float(np.mean(x)),
                "mean_score_b": float(np.mean(y)),
                "mean_difference_b_minus_a": float(np.mean(diff)),
                "p_value": p,
            })

    results = pd.DataFrame(rows)
    results["adjusted_p_value_fdr_bh"] = np.nan
    mask = results["p_value"].notna()
    if mask.any():
        _, adj, _, _ = multipletests(results.loc[mask, "p_value"].astype(float), method="fdr_bh")
        results.loc[mask, "adjusted_p_value_fdr_bh"] = adj

    results.to_csv(output_path, index=False)
    print(f"Saved Random RAG ablation summary to {output_path}")
    print(results.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["generate", "evaluate", "analyze", "all"], default="all")
    parser.add_argument("--seed", type=int, default=5293)
    parser.add_argument("--guideline_only", action="store_true", help="Only judge guideline_alignment to save API calls")
    args = parser.parse_args()

    if args.step in ["generate", "all"]:
        generate_random_rag_outputs(seed=args.seed)

    if args.step in ["evaluate", "all"]:
        evaluate_random_rag(all_metrics=not args.guideline_only)

    if args.step in ["analyze", "all"]:
        paired_compare_random_rag()


if __name__ == "__main__":
    main()
