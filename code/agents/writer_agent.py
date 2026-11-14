"""
Agent 3: WRITER. Writes the answer using ONLY the chunks the Researcher found.

The chunks are numbered [1], [2], ... in the prompt so Gemini can cite them.
If the Reviewer sent the answer back, the writer also gets the reviewer's notes and fixes them.
"""
import logging
from config import llm
from search import chunks_to_text, chunks_to_sources

log = logging.getLogger("writer_agent")

WRITER_PROMPT = """You are MediGuide AI, a careful clinical knowledge assistant.
Answer the question using ONLY the context below.

Rules:
- If the context does not contain the answer, say "I don't know based on the provided documents."
- After each fact, cite the source number like [1] or [2].
- Do not invent drug doses, numbers or facts that are not in the context.
- Do not give a personal diagnosis. Keep the answer clear and short.

Context:
{context}

Question: {question}
{fix_notes}
Answer:"""

# Added to the prompt only when the Reviewer found problems in the first draft
FIX_NOTES = """
Your previous answer was:
{draft}

A fact-checker found these problems: {feedback}
Write a corrected answer that fixes them.
"""


def writer_agent(state):
    fix_notes = ""
    if state["feedback"]:
        fix_notes = FIX_NOTES.format(draft=state["answer"], feedback=state["feedback"])

    prompt = WRITER_PROMPT.format(context=chunks_to_text(state["chunks"]),
                                  question=state["question"], fix_notes=fix_notes)
    answer = llm.invoke(prompt).text
    step = "writer: rewrite draft" if fix_notes else "writer: draft"
    log.info(f"{step} ({len(answer)} characters)")
    return {"answer": answer, "sources": chunks_to_sources(state["chunks"]), "steps": state["steps"] + [step]}
