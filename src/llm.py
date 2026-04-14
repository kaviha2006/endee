import os
from typing import List, Dict
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise OSError("GROQ_API_KEY is not set in your .env file.")
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are DocMind, an expert AI assistant that answers questions
strictly based on the provided context excerpts from the user's documents.
Answer concisely and accurately. If the context does not contain enough
information, say so clearly. Do not make up information."""


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    client = _get_client()

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] Source: {source}\n{text}")
    context_str = "\n\n".join(context_parts)

    user_message = (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()