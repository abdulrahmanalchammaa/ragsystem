from typing import List
from sentence_transformers import SentenceTransformer

_model = None
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> List[List[float]]:
    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    return embed_texts([query], model_name)[0]
