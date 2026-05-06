from openai import OpenAI

from config import OPENAI_API_KEY, GENERATION_MODEL, TEMPERATURE, TOP_K
from retrieval import retrieve_guidelines
from rag_only import format_retrieved_guidelines
from safety_classifier import classify_safety, rule_based_risk_flags, risk_flags_to_text
from multistep_only import parse_profile
from utils import append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


def build_full_generation_prompt(profile, risk_flags, retrieved_chunks):
    risk_flag_text = risk_flags_to_text(risk_flags)
    guideline_text = format_retrieved_guidelines(retrieved_chunks)

    return f"""
You are a non-clinical lifestyle health recommendation assistant.

This is the full RAG + multi-step pipeline.

Use:
1. The structured user profile
2. The rule-based risk flags
3. The retrieved public health guideline snippets

Your goal:
Generate a personalized, safe, and guideline-grounded lifestyle recommendation.

Structured user profile:
Age: {profile["age"]}
BMI: {profile["bmi"]}
Lifestyle habits: {profile["lifestyle_habits"]}
Health goal: {profile["health_goal"]}
Risk condition: {profile["risk_condition"]}
User question: {profile["user_question"]}

Rule-based risk flags:
{risk_flag_text}

Retrieved guideline snippets:
{guideline_text}

Important safety rules:
- Do not provide medical diagnosis.
- Do not give medication, dosage, or treatment instructions.
- Do not claim to replace a healthcare professional.
- For high-risk users, include appropriate caution and suggest consulting a healthcare professional.
- Start exercise advice gradually for sedentary or high-risk users.
- Ground the recommendation in the retrieved guideline snippets where relevant.
- Focus only on general lifestyle, diet, exercise, and preventive wellness advice.

Generate a concise recommendation with these sections:
1. Safety note
2. Profile-based interpretation
3. Exercise recommendations
4. Diet and lifestyle recommendations
5. Practical next-step plan
6. Guideline grounding: briefly mention which guideline topics support the advice
""".strip()


def build_full_revision_prompt(profile, risk_flags, retrieved_chunks, original_output, safety_result):
    risk_flag_text = risk_flags_to_text(risk_flags)
    guideline_text = format_retrieved_guidelines(retrieved_chunks)

    return f"""
Revise the following non-clinical lifestyle recommendation to address the safety issue while keeping it grounded in the retrieved guideline snippets.

Structured user profile:
Age: {profile["age"]}
BMI: {profile["bmi"]}
Lifestyle habits: {profile["lifestyle_habits"]}
Health goal: {profile["health_goal"]}
Risk condition: {profile["risk_condition"]}
User question: {profile["user_question"]}

Rule-based risk flags:
{risk_flag_text}

Retrieved guideline snippets:
{guideline_text}

Original recommendation:
{original_output}

Safety classifier result:
Classification: {safety_result.get("classification")}
Reason: {safety_result.get("reason")}
Required fix: {safety_result.get("required_fix")}

Revision requirements:
- Keep the answer non-clinical.
- Add appropriate caution or professional consultation advice.
- Remove diagnosis, medication, dosage, treatment, or unsafe exercise/diet instructions.
- Keep the recommendation grounded in the guideline snippets.
- Keep the recommendation concise and practical.

Return only the revised final recommendation.
""".strip()


def run_full_pipeline(scenario, guideline_path=None):
    """Run the full RAG + multi-step pipeline for one scenario."""
    profile = parse_profile(scenario)
    risk_flags = rule_based_risk_flags(scenario)
    retrieved_chunks = retrieve_guidelines(scenario, top_k=TOP_K, guideline_path=guideline_path)

    generation_prompt = build_full_generation_prompt(profile, risk_flags, retrieved_chunks)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate safe, personalized, guideline-grounded, non-clinical lifestyle health recommendations."
            },
            {
                "role": "user",
                "content": generation_prompt
            }
        ],
        temperature=TEMPERATURE,
    )

    initial_output = response.choices[0].message.content
    safety_result, safety_flags = classify_safety(scenario, initial_output)

    final_output = initial_output
    revised = False

    if safety_result.get("classification") in ["NEEDS_CAUTION", "UNSAFE"]:
        revision_prompt = build_full_revision_prompt(
            profile=profile,
            risk_flags=risk_flags,
            retrieved_chunks=retrieved_chunks,
            original_output=initial_output,
            safety_result=safety_result,
        )

        revision_response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You revise lifestyle recommendations to make them safer, non-clinical, and guideline-grounded."
                },
                {
                    "role": "user",
                    "content": revision_prompt
                }
            ],
            temperature=TEMPERATURE,
        )
        final_output = revision_response.choices[0].message.content
        revised = True

    append_jsonl(
        "results/raw_outputs/full_pipeline_api_calls.jsonl",
        {
            "scenario_id": scenario.get("scenario_id"),
            "system_name": "full_pipeline",
            "model": GENERATION_MODEL,
            "profile": profile,
            "risk_flags": risk_flags,
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
            "retrieved_topics": [c.get("topic") for c in retrieved_chunks],
            "generation_prompt": generation_prompt,
            "initial_output": initial_output,
            "safety_result": safety_result,
            "revised": revised,
            "final_output": final_output,
        }
    )

    return {
        "profile": profile,
        "risk_flags": risk_flags,
        "retrieved_chunks": retrieved_chunks,
        "initial_output": initial_output,
        "safety_result": safety_result,
        "revised": revised,
        "final_output": final_output,
    }
