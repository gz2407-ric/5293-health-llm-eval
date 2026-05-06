import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

TEMPERATURE = 0
TOP_K = 5

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")
