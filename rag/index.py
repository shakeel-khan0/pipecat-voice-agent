"""Explicitly build or rebuild the standalone Qdrant knowledge index."""

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient, models

from rag.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    KNOWLEDGE_PATH,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)
from rag.models import dense_model, sparse_model


def load_chunks():
    if not KNOWLEDGE_PATH.is_file():
        raise FileNotFoundError(f"Knowledge source not found: {KNOWLEDGE_PATH}")

    converter = DocumentConverter(allowed_formats=[InputFormat.MD])
    document = converter.convert(KNOWLEDGE_PATH).document
    chunker = HybridChunker(always_emit_headings=True)
    chunks = list(chunker.chunk(document))
    if not chunks:
        raise RuntimeError("Docling produced no chunks from the knowledge source.")
    return chunker, chunks


def heading_for(chunk) -> str:
    headings = [heading.strip() for heading in (chunk.meta.headings or []) if heading.strip()]
    return headings[-1] if headings else KNOWLEDGE_PATH.stem


def build_index() -> int:
    chunker, chunks = load_chunks()
    contextualized = [chunker.contextualize(chunk) for chunk in chunks]

    dense_vectors = list(dense_model().passage_embed(contextualized))
    sparse_vectors = list(sparse_model().passage_embed(contextualized))

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    points = []
    for index, (chunk, dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ):
        headings = [heading.strip() for heading in (chunk.meta.headings or []) if heading.strip()]
        points.append(
            models.PointStruct(
                id=index,
                vector={
                    DENSE_VECTOR_NAME: dense.tolist(),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk.text,
                    "heading": heading_for(chunk),
                    "headings": headings,
                    "source": KNOWLEDGE_PATH.name,
                    "chunk_index": index,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    indexed = client.count(collection_name=COLLECTION_NAME, exact=True).count
    if indexed != len(points):
        raise RuntimeError(f"Qdrant confirmed {indexed} points; expected {len(points)}.")
    return indexed


def main():
    indexed = build_index()
    print(f"Indexed {indexed} chunks into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()

