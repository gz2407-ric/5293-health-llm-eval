from rag_only import run_rag_only
from utils import PROJECT_ROOT, load_jsonl, scenario_to_profile_text, write_csv


def summarize_retrieved_chunks(chunks):
    parts = []
    for chunk in chunks:
        parts.append(
            f"{chunk.get('chunk_id', '')} | "
            f"{chunk.get('topic', '')} | "
            f"score={chunk.get('similarity_score', 0):.4f} | "
            f"{chunk.get('text', '')[:250].replace(chr(10), ' ')}"
        )
    return "\n---\n".join(parts)


def main():
    scenario_path = PROJECT_ROOT / "data" / "pilot_scenarios.jsonl"
    output_path = PROJECT_ROOT / "results" / "pilot_rag_outputs.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Cannot find {scenario_path}. Please put pilot_scenarios.jsonl under data/."
        )

    scenarios = load_jsonl(scenario_path)
    rows = []

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        print(f"Running RAG-only for {scenario_id}...")

        model_output, retrieved_chunks = run_rag_only(scenario)

        expected_topic = scenario.get("expected_guideline_topic", "")
        if isinstance(expected_topic, list):
            expected_topic = "; ".join(expected_topic)

        rows.append(
            {
                "scenario_id": scenario_id,
                "system_name": "rag_only",
                "user_profile": scenario_to_profile_text(scenario),
                "user_question": scenario.get("user_question", ""),
                "expected_safety_warning": scenario.get("expected_safety_warning", ""),
                "expected_guideline_topic": expected_topic,
                "retrieved_chunk_ids": "; ".join([str(c.get("chunk_id", "")) for c in retrieved_chunks]),
                "retrieved_topics": "; ".join([str(c.get("topic", "")) for c in retrieved_chunks]),
                "retrieved_guidelines": summarize_retrieved_chunks(retrieved_chunks),
                "model_output": model_output,
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

    print(f"Done. Saved results to {output_path}")


if __name__ == "__main__":
    main()
