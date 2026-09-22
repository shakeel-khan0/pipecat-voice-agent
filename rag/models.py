from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from rag.config import DENSE_MODEL, SPARSE_MODEL


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL)


@lru_cache(maxsize=1)
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL)

