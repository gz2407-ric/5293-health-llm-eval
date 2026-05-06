from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model=GENERATION_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say API test successful in one sentence."}
    ],
    temperature=0
)

print(response.choices[0].message.content)
