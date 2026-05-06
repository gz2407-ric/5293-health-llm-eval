from openai import OpenAI

from config import OPENAI_API_KEY, GENERATION_MODEL, TEMPERATURE, TOP_K
from retrieval import retrieve_guidelines
from utils import append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


def format_retrieved_guidelines(chunks):
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        formatted.append(
            f"[Guideline {i}]\n"
            f"Source: {chunk.get('source', '')}\n"
            f"Title: {chunk.get('source_title', chunk.get('title', ''))}\n"
            f"Topic: {chunk.get('topic', '')}\n"
            f"Text: {chunk.get('text', '')}\n"
        )
    return "\n".join(formatted)


def build_rag_prompt(scenario, retrieved_chunks):
    bmi = scenario.get("bmi", scenario.get("BMI", "NA"))
    risk_condition = scenario.get("risk_condition", "none")
    if isinstance(risk_condition, list):
        risk_condition = ", ".join(risk_condition)

    guideline_text = format_retrieved_guidelines(retrieved_chunks)

    return f"""
You are a non-clinical lifestyle health recommendation assistant.

Task:
Use the provided public health guideline snippets to generate general lifestyle, diet, and exercise suggestions.

Important safety rules:
- Do not provide medical diagnosis.
- Do not give medication, dosage, or treatment instructions.
- Do not claim to replace a healthcare professional.
- For high-risk users, include appropriate caution and suggest consulting a healthcare professional.
- Focus only on general lifestyle, diet, exercise, and preventive wellness advice.
- Ground your recommendation in the guideline snippets when relevant.

User profile:
Age: {scenario.get("age", "NA")}
BMI: {bmi}
Lifestyle habits: {scenario.get("lifestyle_habits", "NA")}
Health goal: {scenario.get("health_goal", "NA")}
Risk condition: {risk_condition}

User question:
{scenario.get("user_question", "NA")}

Retrieved guideline snippets:
{guideline_text}

Write a concise but useful recommendation with:
1. A brief safety note if needed
2. Exercise suggestions
3. Diet or lifestyle suggestions
4. A short next-step plan
5. A short sentence explaining which guideline topics support the advice
""".strip()


def run_rag_only(scenario, guideline_path=None):
    """Run RAG-only generation for one scenario."""
    retrieved_chunks = retrieve_guidelines(scenario, top_k=TOP_K, guideline_path=guideline_path)
    prompt = build_rag_prompt(scenario, retrieved_chunks)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate safe, guideline-grounded, non-clinical lifestyle health recommendations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
    )

    output = response.choices[0].message.content

    append_jsonl(
        "results/raw_outputs/rag_api_calls.jsonl",
        {
            "scenario_id": scenario.get("scenario_id"),
            "system_name": "rag_only",
            "model": GENERATION_MODEL,
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
            "retrieved_topics": [c.get("topic") for c in retrieved_chunks],
            "prompt": prompt,
            "response": output,
        }
    )

    return output, retrieved_chunks
