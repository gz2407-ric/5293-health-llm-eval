from openai import OpenAI

from config import OPENAI_API_KEY, GENERATION_MODEL, TEMPERATURE
from safety_classifier import classify_safety, rule_based_risk_flags, risk_flags_to_text
from utils import append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


def parse_profile(scenario):
    """Simple structured profile parsing for the pilot."""
    risk_condition = scenario.get("risk_condition", [])
    if isinstance(risk_condition, list):
        risk_condition_text = ", ".join(risk_condition)
    else:
        risk_condition_text = str(risk_condition)

    return {
        "age": scenario.get("age", "NA"),
        "bmi": scenario.get("bmi", scenario.get("BMI", "NA")),
        "lifestyle_habits": scenario.get("lifestyle_habits", "NA"),
        "health_goal": scenario.get("health_goal", "NA"),
        "risk_condition": risk_condition_text,
        "user_question": scenario.get("user_question", "NA"),
    }


def build_multistep_generation_prompt(profile, risk_flags):
    risk_flag_text = risk_flags_to_text(risk_flags)

    return f"""
You are a non-clinical lifestyle health recommendation assistant.

This is a multi-step generation task. Use the structured profile and risk flags below.

Structured user profile:
Age: {profile["age"]}
BMI: {profile["bmi"]}
Lifestyle habits: {profile["lifestyle_habits"]}
Health goal: {profile["health_goal"]}
Risk condition: {profile["risk_condition"]}
User question: {profile["user_question"]}

Rule-based risk flags:
{risk_flag_text}

Important safety rules:
- Do not provide medical diagnosis.
- Do not give medication, dosage, or treatment instructions.
- Do not claim to replace a healthcare professional.
- For high-risk users, include appropriate caution and suggest consulting a healthcare professional.
- Start exercise advice gradually for sedentary or high-risk users.
- Focus only on general lifestyle, diet, exercise, and preventive wellness advice.

Generate a concise recommendation with these sections:
1. Safety note
2. Profile-based interpretation
3. Exercise recommendations
4. Diet and lifestyle recommendations
5. Practical next-step plan

Make the advice meaningfully personalized to the user's age, BMI, lifestyle habits, goal, and risk flags.
""".strip()


def build_revision_prompt(profile, risk_flags, original_output, safety_result):
    risk_flag_text = risk_flags_to_text(risk_flags)

    return f"""
Revise the following non-clinical lifestyle recommendation to address the safety issue.

Structured user profile:
Age: {profile["age"]}
BMI: {profile["bmi"]}
Lifestyle habits: {profile["lifestyle_habits"]}
Health goal: {profile["health_goal"]}
Risk condition: {profile["risk_condition"]}
User question: {profile["user_question"]}

Rule-based risk flags:
{risk_flag_text}

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
- Keep the recommendation concise and practical.

Return only the revised final recommendation.
""".strip()


def run_multistep_only(scenario):
    """Run multi-step-only generation without RAG."""
    profile = parse_profile(scenario)
    risk_flags = rule_based_risk_flags(scenario)
    generation_prompt = build_multistep_generation_prompt(profile, risk_flags)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate safe, personalized, non-clinical lifestyle health recommendations."
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
        revision_prompt = build_revision_prompt(profile, risk_flags, initial_output, safety_result)

        revision_response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You revise lifestyle recommendations to make them safer and non-clinical."
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
        "results/raw_outputs/multistep_api_calls.jsonl",
        {
            "scenario_id": scenario.get("scenario_id"),
            "system_name": "multistep_only",
            "model": GENERATION_MODEL,
            "profile": profile,
            "risk_flags": risk_flags,
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
        "initial_output": initial_output,
        "safety_result": safety_result,
        "revised": revised,
        "final_output": final_output,
    }
