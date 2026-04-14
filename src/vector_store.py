import os
from typing import List, Dict, Any
from endee import Endee, Precision

INDEX_NAME = "docmind_index"
DIMENSION = 384
SPACE_TYPE = "cosine"


class VectorStore:
    def __init__(self) -> None:
        host = os.getenv("ENDEE_HOST", "http://localhost:8080")
        self._client = Endee()
        self._client.set_base_url(f"{host}/api/v1")
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._client.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                space_type=SPACE_TYPE,
                precision=Precision.INT8,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "already exists" in error_msg or "conflict" in error_msg:
                pass
            else:
                raise RuntimeError(
                    f"Could not connect to Endee at http://localhost:8080/api/v1. "
                    f"Make sure the Docker container is running.\nError: {exc}"
                ) from exc

    def delete_index(self) -> None:
        try:
            idx = self._client.get_index(name=INDEX_NAME)
            idx.delete()
        except Exception:
            pass
        self._client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type=SPACE_TYPE,
            precision=Precision.INT8,
        )

    def upsert(self, vectors: List[Dict[str, Any]], batch_size: int = 500) -> None:
        index = self._client.get_index(name=INDEX_NAME)
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors[i: i + batch_size])

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_filter: str = None,
    ) -> List[Dict[str, Any]]:
        index = self._client.get_index(name=INDEX_NAME)
        kwargs: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": top_k,
            "ef": 128,
            "include_vectors": False,
        }
        if source_filter:
            kwargs["filter"] = [{"source": {"$eq": source_filter}}]
        results = index.query(**kwargs)
        return results