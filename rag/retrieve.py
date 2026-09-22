"""Dense+sparse retrieval with Qdrant-native reciprocal-rank fusion."""

from dataclasses import dataclass
from time import perf_counter

from qdrant_client import QdrantClient, models

from rag.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    FINAL_TOP_K,
    PREFETCH_LIMIT,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)
from rag.models import dense_model, sparse_model


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    heading: str
    text: str


class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        if not self.client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Run: python -m rag.index"
            )
        self.dense = dense_model()
        self.sparse = sparse_model()

    def search(self, query: str, top_k: int = FINAL_TOP_K):
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        started = perf_counter()
        dense_vector = next(self.dense.query_embed(query))
        sparse_vector = next(self.sparse.query_embed(query))

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=PREFETCH_LIMIT,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=PREFETCH_LIMIT,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        latency_ms = (perf_counter() - started) * 1000

        results = [
            RetrievalResult(
                rank=rank,
                score=float(point.score),
                heading=str((point.payload or {}).get("heading", "")),
                text=str((point.payload or {}).get("text", "")),
            )
            for rank, point in enumerate(response.points, start=1)
        ]
        return results, latency_ms

