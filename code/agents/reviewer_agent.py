"""
Agent 4: REVIEWER. A fact-checker that reads the Writer's answer next to the sources.

- PASS -> approve the answer and add the medical disclaimer
- FAIL -> send it back to the Writer with notes (only once, so we never loop forever)
- FAIL again after the fix -> still return it, but with a clear warning on top
"""
import logging
from config import llm
from search import chunks_to_text
from safety import DISCLAIMER, UNVERIFIED_NOTE

log = logging.getLogger("reviewer_agent")

REVIEW_PROMPT = """You are a strict medical fact-checker.
Check the ANSWER against the SOURCES:
- Every fact in the answer must be supported by the sources.
- Each citation like [1] must point to a source that supports that fact.
- There must be no drug doses or numbers that are not in the sources.

Reply with exactly PASS if the answer is fine.
Otherwise reply FAIL: followed by a short list of the problems.

SOURCES:
{context}

QUESTION: {question}

ANSWER:
{answer}"""


def reviewer_agent(state):
    prompt = REVIEW_PROMPT.format(context=chunks_to_text(state["chunks"]),
                                  question=state["question"], answer=state["answer"])
    verdict = llm.invoke(prompt).text.strip()
    passed = verdict.upper().startswith("PASS")
    log.info(f"Review verdict: {verdict[:200]}")

    if passed:
        return {"needs_fix": False, "answer": state["answer"] + DISCLAIMER,
                "steps": state["steps"] + ["reviewer: approved"]}

    if state["revisions"] == 0:
        # First failure -> back to the Writer with the reviewer's notes
        return {"needs_fix": True, "feedback": verdict, "revisions": 1,
                "steps": state["steps"] + ["reviewer: sent back"]}

    # Failed again -> stop, but warn the user
    log.warning("Answer failed review twice -> returning it with a warning")
    return {"needs_fix": False, "answer": UNVERIFIED_NOTE + state["answer"] + DISCLAIMER,
            "steps": state["steps"] + ["reviewer: not fully verified"]}
