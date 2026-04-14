from typing import List
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> List[float]:
    model = _get_model()
    return model.encode(text.replace("\n", " ").strip(), convert_to_numpy=True).tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    model = _get_model()
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    embeddings = model.encode(cleaned, batch_size=batch_size, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]