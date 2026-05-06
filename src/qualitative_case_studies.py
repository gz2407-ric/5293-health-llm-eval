import pandas as pd

from utils import PROJECT_ROOT, write_csv


def get_score(pivot, scenario_id, system_name, metric):
    try:
        return pivot.loc[(scenario_id, system_name), metric]
    except Exception:
        return None


def get_output(outputs, scenario_id, system_name):
    row = outputs[(outputs["scenario_id"] == scenario_id) & (outputs["system_name"] == system_name)]
    if row.empty:
        return ""
    return row.iloc[0]["model_output"]


def select_cases(outputs, scores):
    pivot = scores.pivot_table(
        index=["scenario_id", "system_name"],
        columns="metric",
        values="score",
        aggfunc="mean"
    )

    scenario_ids = sorted(outputs["scenario_id"].unique())
    candidates = []

    for sid in scenario_ids:
        b_align = get_score(pivot, sid, "baseline", "guideline_alignment")
        r_align = get_score(pivot, sid, "rag_only", "guideline_alignment")
        f_align = get_score(pivot, sid, "full_pipeline", "guideline_alignment")
        m_align = get_score(pivot, sid, "multistep_only", "guideline_alignment")

        b_safe = get_score(pivot, sid, "baseline", "safety")
        m_safe = get_score(pivot, sid, "multistep_only", "safety")
        f_safe = get_score(pivot, sid, "full_pipeline", "safety")

        b_pers = get_score(pivot, sid, "baseline", "personalization")
        r_pers = get_score(pivot, sid, "rag_only", "personalization")
        m_pers = get_score(pivot, sid, "multistep_only", "personalization")
        f_pers = get_score(pivot, sid, "full_pipeline", "personalization")

        if b_align is not None and r_align is not None:
            candidates.append({
                "scenario_id": sid,
                "case_type": "RAG improves guideline alignment over baseline",
                "score_gap": r_align - b_align,
                "comparison": "baseline vs rag_only",
            })

        if m_safe is not None and b_safe is not None:
            candidates.append({
                "scenario_id": sid,
                "case_type": "Multi-step improves safety over baseline",
                "score_gap": m_safe - b_safe,
                "comparison": "baseline vs multistep_only",
            })

        if f_align is not None and m_align is not None and f_safe is not None and m_safe is not None:
            candidates.append({
                "scenario_id": sid,
                "case_type": "Full pipeline trades safety for guideline specificity",
                "score_gap": (f_align - m_align) + (m_safe - f_safe),
                "comparison": "multistep_only vs full_pipeline",
            })

        if all(v is not None for v in [b_pers, r_pers, m_pers, f_pers]):
            pers_range = max(b_pers, r_pers, m_pers, f_pers) - min(b_pers, r_pers, m_pers, f_pers)
            candidates.append({
                "scenario_id": sid,
                "case_type": "Personalization differences are small",
                "score_gap": -pers_range,
                "comparison": "all systems",
            })

    cand = pd.DataFrame(candidates)
    selected = []

    for case_type in [
        "RAG improves guideline alignment over baseline",
        "Multi-step improves safety over baseline",
        "Full pipeline trades safety for guideline specificity",
        "Personalization differences are small",
    ]:
        subset = cand[cand["case_type"] == case_type].sort_values("score_gap", ascending=False)
        if not subset.empty:
            selected.append(subset.iloc[0].to_dict())

    # Add one full pipeline strong alignment case
    align_rows = []
    for sid in scenario_ids:
        f_align = get_score(pivot, sid, "full_pipeline", "guideline_alignment")
        if f_align is not None:
            align_rows.append({
                "scenario_id": sid,
                "case_type": "Full pipeline strong guideline grounding",
                "score_gap": f_align,
                "comparison": "full_pipeline",
            })
    if align_rows:
        selected.append(pd.DataFrame(align_rows).sort_values("score_gap", ascending=False).iloc[0].to_dict())

    return selected, pivot


def main():
    outputs_path = PROJECT_ROOT / "results" / "model_outputs.csv"
    scores_path = PROJECT_ROOT / "results" / "judge_scores.csv"

    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing {outputs_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")

    outputs = pd.read_csv(outputs_path).fillna("")
    scores = pd.read_csv(scores_path).dropna(subset=["score"]).copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score"])

    selected, pivot = select_cases(outputs, scores)

    rows = []
    for idx, case in enumerate(selected, start=1):
        sid = case["scenario_id"]

        scenario_row = outputs[outputs["scenario_id"] == sid].iloc[0]

        rows.append({
            "case_id": f"CASE{idx:02d}",
            "scenario_id": sid,
            "case_type": case["case_type"],
            "comparison": case["comparison"],
            "why_this_case_matters": (
                "Selected automatically based on judge-score patterns. "
                "Review manually before using in the report."
            ),
            "user_profile": scenario_row.get("user_profile", ""),
            "user_question": scenario_row.get("user_question", ""),
            "baseline_output": get_output(outputs, sid, "baseline"),
            "rag_only_output": get_output(outputs, sid, "rag_only"),
            "multistep_only_output": get_output(outputs, sid, "multistep_only"),
            "full_pipeline_output": get_output(outputs, sid, "full_pipeline"),
            "baseline_guideline_alignment": get_score(pivot, sid, "baseline", "guideline_alignment"),
            "rag_guideline_alignment": get_score(pivot, sid, "rag_only", "guideline_alignment"),
            "multistep_safety": get_score(pivot, sid, "multistep_only", "safety"),
            "full_pipeline_safety": get_score(pivot, sid, "full_pipeline", "safety"),
            "baseline_personalization": get_score(pivot, sid, "baseline", "personalization"),
            "rag_personalization": get_score(pivot, sid, "rag_only", "personalization"),
            "multistep_personalization": get_score(pivot, sid, "multistep_only", "personalization"),
            "full_pipeline_personalization": get_score(pivot, sid, "full_pipeline", "personalization"),
        })

    output_path = PROJECT_ROOT / "results" / "case_studies_notes.csv"
    write_csv(output_path, rows, fieldnames=list(rows[0].keys()) if rows else [])
    print(f"Saved qualitative case-study candidates to {output_path}")
    print("Review these cases manually before using them in the report.")


if __name__ == "__main__":
    main()
