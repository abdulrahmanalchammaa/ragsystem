import ollama

DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.

Rules:
- Answer only from the provided context. If the answer isn't in the context, say so clearly.
- Cite the source document(s) using their [number] reference when possible.
- Be concise and accurate. Do not invent facts.
- If the context is insufficient, acknowledge the limitation."""


def build_prompt(query: str, context: str) -> str:
    return f"""Context documents:

{context}

---

Question: {query}

Answer based on the context above:"""


def generate_answer(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, context)},
            ],
            options={"num_predict": max_tokens},
        )
        return response.message.content
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running.\n"
                "Download from: https://ollama.com\n"
                "Then run: ollama pull llama3.2"
            )
        raise
