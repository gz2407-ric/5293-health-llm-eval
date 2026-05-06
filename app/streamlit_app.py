import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

st.set_page_config(
    page_title="Health LLM Architecture Evaluation",
    layout="wide",
)

SYSTEM_LABELS = {
    "baseline": "Baseline",
    "rag_only": "RAG only",
    "multistep_only": "Multi-step only",
    "full_pipeline": "Full pipeline",
}
SYSTEM_ORDER = ["baseline", "rag_only", "multistep_only", "full_pipeline"]

METRIC_LABELS = {
    "relevance": "Relevance",
    "personalization": "Personalization",
    "guideline_alignment": "Guideline Alignment",
    "safety": "Safety",
}


@st.cache_data
def load_csv(path):
    path = PROJECT_ROOT / path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def system_label(name):
    return SYSTEM_LABELS.get(name, name)


def main():
    st.title("Evaluating LLM Architectures for Personalized Lifestyle Health Recommendations")

    tab1, tab2 = st.tabs(["Explore Results", "Try a Custom Profile"])

    with tab1:
        st.subheader("Explore Precomputed Results")
        st.caption("This tab reads precomputed CSV files and does not call the live API.")

        outputs = load_csv("results/model_outputs.csv")
        scores = load_csv("results/judge_scores.csv")
        judge_summary = load_csv("results/judge_summary.csv")
        retrieval_summary = load_csv("results/retrieval_summary.csv")

        if outputs.empty:
            st.error("Missing results/model_outputs.csv. Run the experiment first.")
            return

        col1, col2, col3 = st.columns(3)

        scenario_ids = sorted(outputs["scenario_id"].unique())
        systems = [s for s in SYSTEM_ORDER if s in outputs["system_name"].unique()]

        with col1:
            selected_scenario = st.selectbox("Scenario", scenario_ids)
        with col2:
            selected_system = st.selectbox(
                "System",
                systems,
                format_func=system_label,
                index=systems.index("full_pipeline") if "full_pipeline" in systems else 0,
            )
        with col3:
            st.metric("Total outputs", len(outputs))

        selected = outputs[
            (outputs["scenario_id"] == selected_scenario)
            & (outputs["system_name"] == selected_system)
        ]

        if selected.empty:
            st.warning("No output found for the selected scenario/system.")
        else:
            row = selected.iloc[0]

            st.markdown("### User Profile")
            st.text(row.get("user_profile", ""))

            st.markdown("### User Question")
            st.write(row.get("user_question", ""))

            st.markdown("### Model Output")
            st.write(row.get("model_output", ""))

            if row.get("retrieved_guidelines", ""):
                with st.expander("Retrieved Guideline Chunks"):
                    st.text(row.get("retrieved_guidelines", ""))

            if row.get("active_risk_flags", ""):
                with st.expander("Risk Flags and Safety Result"):
                    st.write("**Risk flags:**", row.get("active_risk_flags", ""))
                    st.write("**Safety classification:**", row.get("safety_classification", ""))
                    st.write("**Safety reason:**", row.get("safety_reason", ""))

        st.markdown("---")
        st.markdown("### Average Judge Scores")

        if not judge_summary.empty:
            pivot = judge_summary.pivot(index="system_name", columns="metric", values="score")
            ordered = [s for s in SYSTEM_ORDER if s in pivot.index]
            pivot = pivot.loc[ordered]
            pivot.index = [system_label(s) for s in pivot.index]
            pivot = pivot.rename(columns=METRIC_LABELS)
            st.dataframe(pivot, use_container_width=True)
            st.bar_chart(pivot)

        st.markdown("### Retrieval Summary")
        if not retrieval_summary.empty:
            temp = retrieval_summary.copy()
            temp["system_name"] = temp["system_name"].map(system_label)
            st.dataframe(temp, use_container_width=True)

        st.markdown("### Key Figures")
        fig_paths = [
            ("Guideline Alignment", "figures/full_guideline_alignment_by_system.png"),
            ("Retrieval Recall", "figures/full_retrieval_recall.png"),
            ("Average Scores", "figures/full_average_scores_by_system.png"),
            ("Output Length", "figures/full_output_length_by_system.png"),
        ]
        for title, rel_path in fig_paths:
            path = PROJECT_ROOT / rel_path
            if path.exists():
                st.markdown(f"#### {title}")
                st.image(str(path), use_container_width=True)

    with tab2:
        st.subheader("Try a Custom Profile")
        st.warning("This mode calls the live API and may take 30 seconds or longer. Use Explore Results for the stable presentation demo.")

        age = st.number_input("Age", min_value=18, max_value=80, value=45)
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=28.0, step=0.1)
        lifestyle = st.text_area("Lifestyle habits", "Sedentary office job, high sodium diet, poor sleep.")
        goal = st.text_input("Health goal", "Improve fitness and lower health risk.")
        risk_condition_text = st.text_input("Risk condition(s)", "hypertension")
        question = st.text_area("User question", "What lifestyle changes should I make safely?")

        if st.button("Run Full Pipeline"):
            scenario = {
                "scenario_id": "custom_demo",
                "age": int(age),
                "BMI": float(bmi),
                "lifestyle_habits": lifestyle,
                "health_goal": goal,
                "risk_condition": [x.strip() for x in risk_condition_text.split(",") if x.strip()],
                "user_question": question,
                "expected_safety_warning": "",
                "expected_guideline_topic": "",
            }

            try:
                from full_pipeline import run_full_pipeline

                with st.spinner("Running Full Pipeline..."):
                    result = run_full_pipeline(scenario)

                st.markdown("### Full Pipeline Output")
                st.write(result["final_output"])

                with st.expander("Risk Flags"):
                    st.json(result["risk_flags"])

                with st.expander("Safety Result"):
                    st.json(result["safety_result"])

                with st.expander("Retrieved Guideline Chunks"):
                    for chunk in result["retrieved_chunks"]:
                        st.write(f"**{chunk.get('chunk_id')} — {chunk.get('topic')}**")
                        st.write(chunk.get("text", ""))

            except Exception as e:
                st.error(f"Live API demo failed: {e}")


if __name__ == "__main__":
    main()
