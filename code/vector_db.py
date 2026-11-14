"""
Step 2 of the pipeline: store chunks in Qdrant (our vector database).

For HYBRID search every chunk is stored with two vectors:
- Dense vector  (Gemini):  captures MEANING   -> "heart attack" matches "myocardial infarction"
- Sparse vector (BM25):    captures KEYWORDS  -> exact drug names, lab codes, abbreviations
"""
import time
import logging
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from config import QDRANT_URL, COLLECTION_NAME, embeddings

log = logging.getLogger("vector_db")

# BM25 keyword model. Runs on your computer (small download the first time).
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Settings shared by both functions below
QDRANT_SETTINGS = dict(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    retrieval_mode=RetrievalMode.HYBRID,
    timeout=120,  # seconds. The default (5s) is too short when saving many chunks.
)


def has_documents():
    """True if at least one PDF has been uploaded (the collection exists)."""
    return QdrantClient(url=QDRANT_URL).collection_exists(COLLECTION_NAME)


def save_chunks(chunks):
    """Create both vectors for each chunk and save them in Qdrant."""
    log.info(f"Embedding and saving {len(chunks)} chunks to Qdrant...")
    start = time.time()
    if has_documents():
        get_vector_db().add_documents(chunks)
    else:
        # First upload: this also creates the collection
        log.info(f"Creating new collection '{COLLECTION_NAME}'")
        QdrantVectorStore.from_documents(chunks, **QDRANT_SETTINGS)
    log.info(f"Saved {len(chunks)} chunks in {time.time() - start:.1f}s")


def get_vector_db():
    """Connect to the existing collection so we can search it."""
    return QdrantVectorStore.from_existing_collection(**QDRANT_SETTINGS)
