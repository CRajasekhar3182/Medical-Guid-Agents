"""
Step 3 of the pipeline: find the chunks that best answer a question.

1. Hybrid search: Qdrant searches by meaning AND by keywords, then merges
   both result lists. We take the top 10 candidates.
2. Reranking: a small model reads the question and each chunk TOGETHER
   and scores how well the chunk answers it. We keep the best 4.
"""
import logging
from flashrank import Ranker, RerankRequest
from vector_db import get_vector_db
from config import SEARCH_CANDIDATES, TOP_RESULTS

log = logging.getLogger("search")

# Tiny reranking model that runs on CPU (~4 MB). ~0.1s per search.
# (ms-marco-MiniLM-L-12-v2 ranks a bit better but takes ~3.5s on a laptop CPU)
reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")


def find_relevant_chunks(question):
    # 1. Hybrid search -> 10 candidates
    candidates = get_vector_db().similarity_search(question, k=SEARCH_CANDIDATES)
    log.info(f"Hybrid search for '{question}' found {len(candidates)} candidates")
    if not candidates:
        return []

    # 2. Rerank. Flashrank wants a list of {"id", "text"}; the id lets us find the chunk again.
    passages = [{"id": i, "text": chunk.page_content} for i, chunk in enumerate(candidates)]
    ranked = reranker.rerank(RerankRequest(query=question, passages=passages))  # best first

    # 3. Keep the top 4 and remember their score (shown in the UI)
    best_chunks = []
    for result in ranked[:TOP_RESULTS]:
        chunk = candidates[result["id"]]
        chunk.metadata["score"] = round(float(result["score"]), 3)
        best_chunks.append(chunk)

    kept = ", ".join(f"p.{c.metadata['page']} ({c.metadata['score']})" for c in best_chunks)
    log.info(f"Reranked, kept top {len(best_chunks)}: {kept}")
    return best_chunks


# ---------- Helpers used by the agents to show chunks ----------

def chunks_to_text(chunks):
    """Turn chunks into numbered blocks for prompts: [1] (Source: file.pdf, page 3) text..."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        blocks.append(f"[{number}] (Source: {meta['source']}, page {meta['page']})\n{chunk.page_content}")
    return "\n\n".join(blocks)


def chunks_to_sources(chunks):
    """The source list sent to the chat UI (same numbering as the citations)."""
    return [
        {
            "id": number,
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "score": chunk.metadata.get("score"),
            "text": chunk.page_content,
        }
        for number, chunk in enumerate(chunks, start=1)
    ]
