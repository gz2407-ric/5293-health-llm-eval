from openai import OpenAI

from config import OPENAI_API_KEY, GENERATION_MODEL, TEMPERATURE
from prompts import build_baseline_prompt
from utils import append_jsonl


client = OpenAI(api_key=OPENAI_API_KEY)


def run_baseline(scenario):
    """Run baseline single-prompt generation for one scenario."""
    prompt = build_baseline_prompt(scenario)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate safe, non-clinical lifestyle health recommendations."
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
        "results/raw_outputs/baseline_api_calls.jsonl",
        {
            "scenario_id": scenario.get("scenario_id"),
            "system_name": "baseline",
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "response": output,
        }
    )

    return output
