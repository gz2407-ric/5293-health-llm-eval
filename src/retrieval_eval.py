import argparse
import re
import pandas as pd

from utils import PROJECT_ROOT


STOPWORDS = {
    "and", "or", "for", "the", "a", "an", "of", "to", "in", "with", "on", "at",
    "adults", "adult", "lifestyle", "guidelines", "guideline", "management",
    "risk", "safe", "safety", "general"
}


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s,;/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_expected_topics(expected):
    text = normalize_text(expected)
    parts = re.split(r"[,;/]|\band\b", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def keyword_tokens(phrase):
    toks = normalize_text(phrase).split()
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def topic_is_covered(expected_topic, retrieved_text):
    expected_norm = normalize_text(expected_topic)
    retrieved_norm = normalize_text(retrieved_text)

    if expected_norm and expected_norm in retrieved_norm:
        return True

    tokens = keyword_tokens(expected_topic)
    if not tokens:
        return False

    matched = sum(1 for t in tokens if t in retrieved_norm)
    threshold = max(1, min(len(tokens), int(round(len(tokens) * 0.5))))
    return matched >= threshold


def coverage_at_k(expected_topics, retrieved_topics, retrieved_guidelines, k):
    top_topics = retrieved_topics[:k]
    retrieved_text = " ".join(top_topics) + " " + str(retrieved_guidelines)

    if not expected_topics:
        return {
            "covered_count": 0,
            "expected_count": 0,
            "coverage_rate": None,
            "covered_topics": "",
            "hit": False,
        }

    covered = [topic for topic in expected_topics if topic_is_covered(topic, retrieved_text)]
    return {
        "covered_count": len(covered),
        "expected_count": len(expected_topics),
        "coverage_rate": len(covered) / len(expected_topics),
        "covered_topics": "; ".join(covered),
        "hit": len(covered) > 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    args = parser.parse_args()

    if args.mode == "pilot":
        input_path = PROJECT_ROOT / "results" / "pilot_all_outputs_unified.csv"
        output_path = PROJECT_ROOT / "results" / "pilot_retrieval_results.csv"
        summary_path = PROJECT_ROOT / "results" / "pilot_retrieval_summary.csv"
    else:
        input_path = PROJECT_ROOT / "results" / "model_outputs.csv"
        output_path = PROJECT_ROOT / "results" / "retrieval_results.csv"
        summary_path = PROJECT_ROOT / "results" / "retrieval_summary.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing model outputs file: {input_path}")

    outputs = pd.read_csv(input_path).fillna("")
    rag_outputs = outputs[outputs["system_name"].isin(["rag_only", "full_pipeline"])].copy()

    rows = []
    for _, row in rag_outputs.iterrows():
        retrieved_topics = [
            x.strip() for x in str(row.get("retrieved_topics", "")).split(";") if x.strip()
        ]
        expected_topics = split_expected_topics(row.get("expected_guideline_topic", ""))
        retrieved_guidelines = row.get("retrieved_guidelines", "")

        cov3 = coverage_at_k(expected_topics, retrieved_topics, retrieved_guidelines, 3)
        cov5 = coverage_at_k(expected_topics, retrieved_topics, retrieved_guidelines, 5)

        rows.append({
            "scenario_id": row.get("scenario_id"),
            "system_name": row.get("system_name"),
            "expected_guideline_topic": row.get("expected_guideline_topic", ""),
            "retrieved_topics": row.get("retrieved_topics", ""),
            "hit_at_3": cov3["hit"],
            "hit_at_5": cov5["hit"],
            "coverage_rate_at_3": cov3["coverage_rate"],
            "coverage_rate_at_5": cov5["coverage_rate"],
            "covered_topics_at_3": cov3["covered_topics"],
            "covered_topics_at_5": cov5["covered_topics"],
            "expected_topic_count": cov5["expected_count"],
        })

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby("system_name")
        .agg(
            recall_at_3=("hit_at_3", "mean"),
            recall_at_5=("hit_at_5", "mean"),
            mean_coverage_rate_at_3=("coverage_rate_at_3", "mean"),
            mean_coverage_rate_at_5=("coverage_rate_at_5", "mean"),
            n=("scenario_id", "count"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved retrieval results to {output_path}")
    print(f"Saved retrieval summary to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
