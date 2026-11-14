"""
Agent 2: RESEARCHER. Finds the chunks that answer the question.

This agent is its own small LangGraph graph (a "subgraph"), because it has its own loop:

    search -> grade -> relevant?  yes -> done (found = True)
                         | no, first time -> rewrite the query -> search again
                         | no, already rewrote -> give up (found = False, "not found" message)
"""
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from config import llm
from search import find_relevant_chunks, chunks_to_text
from safety import NOT_FOUND_MESSAGE

log = logging.getLogger("research_agent")


class ResearchState(TypedDict):
    question: str        # what the user asked
    search_query: str    # what we search for (starts as the question, may be rewritten)
    rewrites: int        # how many times we rewrote the query
    chunks: list         # chunks found by search
    found: bool          # did the grader say the chunks are useful?
    answer: str          # set only when we give up
    sources: list
    steps: list


def search(state):
    chunks = find_relevant_chunks(state["search_query"])
    return {"chunks": chunks, "steps": state["steps"] + ["research: search"]}


def grade(state):
    """Ask Gemini a yes/no question: do these chunks help answer the question?"""
    if not state["chunks"]:
        return {"found": False, "steps": state["steps"] + ["research: grade (no)"]}

    prompt = (
        f"Question: {state['question']}\n\n"
        f"Documents:\n{chunks_to_text(state['chunks'])}\n\n"
        "Do these documents contain information that helps answer the question? "
        "Reply with only one word: yes or no."
    )
    found = llm.invoke(prompt).text.strip().lower().startswith("yes")
    log.info(f"Grader says chunks are relevant: {found}")
    return {"found": found, "steps": state["steps"] + [f"research: grade ({'yes' if found else 'no'})"]}


def rewrite(state):
    """Turn the question into a better search query (proper medical terms)."""
    prompt = (
        "Rewrite this question as a short search query for medical documents. "
        "Use proper medical terms. Return only the query.\n\n"
        f"Question: {state['question']}"
    )
    new_query = llm.invoke(prompt).text.strip()
    log.info(f"Rewrote query: '{state['question']}' -> '{new_query}'")
    return {"search_query": new_query, "rewrites": state["rewrites"] + 1,
            "steps": state["steps"] + [f"research: rewrite -> {new_query}"]}


def give_up(state):
    log.info("No relevant chunks even after rewriting -> 'not found'")
    return {"answer": NOT_FOUND_MESSAGE, "sources": [], "steps": state["steps"] + ["research: not found"]}


def after_grade(state):
    if state["found"]:
        return "done"
    if state["rewrites"] == 0:
        return "rewrite"     # try once more with a better query
    return "give_up"         # still nothing useful -> don't guess


graph = StateGraph(ResearchState)
graph.add_node("search", search)
graph.add_node("grade", grade)
graph.add_node("rewrite", rewrite)
graph.add_node("give_up", give_up)

graph.add_edge(START, "search")
graph.add_edge("search", "grade")
graph.add_conditional_edges("grade", after_grade, {"done": END, "rewrite": "rewrite", "give_up": "give_up"})
graph.add_edge("rewrite", "search")   # loop back and search again
graph.add_edge("give_up", END)

# Compiled subgraph. The supervisor uses it like a normal node.
research_agent = graph.compile()
