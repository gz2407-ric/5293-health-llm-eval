import json
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, EMBEDDING_MODEL, TOP_K
from utils import PROJECT_ROOT


client = OpenAI(api_key=OPENAI_API_KEY)


_EMBEDDING_CACHE = {}


def load_guideline_chunks(path=None):
    """Load guideline chunks from JSONL."""
    if path is None:
        path = PROJECT_ROOT / "guidelines" / "pilot_guideline_chunks.jsonl"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find guideline chunks file: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


def chunk_to_text(chunk):
    """Convert one guideline chunk into searchable text."""
    keywords = chunk.get("retrieval_keywords", "")
    if isinstance(keywords, list):
        keywords = ", ".join(keywords)

    return f"""
Source: {chunk.get("source", "")}
Title: {chunk.get("source_title", chunk.get("title", ""))}
Topic: {chunk.get("topic", "")}
Keywords: {keywords}
Text: {chunk.get("text", "")}
""".strip()


def scenario_to_query(scenario):
    """Build a retrieval query from one scenario."""
    risk_condition = scenario.get("risk_condition", "none")
    if isinstance(risk_condition, list):
        risk_condition = ", ".join(risk_condition)

    return f"""
User question: {scenario.get("user_question", "")}
Health goal: {scenario.get("health_goal", "")}
Lifestyle habits: {scenario.get("lifestyle_habits", "")}
Risk condition: {risk_condition}
Expected guideline topic: {scenario.get("expected_guideline_topic", "")}
""".strip()


def get_embedding(text):
    """Get OpenAI embedding for a text string, with simple in-memory cache."""
    key = text.strip()
    if key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[key]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=key
    )
    embedding = response.data[0].embedding
    _EMBEDDING_CACHE[key] = embedding
    return embedding


def retrieve_guidelines(scenario, top_k=TOP_K, guideline_path=None):
    """
    Retrieve top-k guideline chunks for one scenario.
    guideline_path can be pilot_guideline_chunks.jsonl or guideline_chunks.jsonl.
    """
    chunks = load_guideline_chunks(guideline_path)
    query = scenario_to_query(scenario)

    query_embedding = np.array(get_embedding(query)).reshape(1, -1)

    chunk_texts = [chunk_to_text(chunk) for chunk in chunks]
    chunk_embeddings = np.array([get_embedding(text) for text in chunk_texts])

    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []
    for idx in ranked_indices:
        chunk = dict(chunks[idx])
        chunk["similarity_score"] = float(scores[idx])
        chunk["retrieval_text"] = chunk_texts[idx]
        retrieved.append(chunk)

    return retrieved
