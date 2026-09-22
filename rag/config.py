from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge" / "agentix_rag_knowledge_base.md"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "agentix_rag_knowledge"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 384

FINAL_TOP_K = 3
PREFETCH_LIMIT = 12

