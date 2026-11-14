"""
The SUPERVISOR: connects the 4 agents into one multi-agent workflow (LangGraph).

    question
       |
    ROUTER ---- emergency / chitchat ----> reply right away
       | medical
    RESEARCHER (search -> grade -> rewrite & retry) ---- nothing found ----> "not found"
       | found chunks
    WRITER  <------------------+
       |                       | FAIL (only once)
    REVIEWER ------------------+
       | PASS
    final answer + disclaimer

How the agents talk: they share one "state" dictionary (TeamState).
Each agent reads what it needs and writes its results back. The supervisor
reads those results (intent, found, needs_fix) to decide which agent runs next.

We use fixed rules for this, not an AI "boss" that picks the next agent.
Fixed rules are faster, cheaper and always behave the same way,
which matters for medical answers.
"""
import time
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.router_agent import router_agent
from agents.research_agent import research_agent
from agents.writer_agent import writer_agent
from agents.reviewer_agent import reviewer_agent

log = logging.getLogger("supervisor")


class TeamState(TypedDict):
    question: str       # what the user asked
    intent: str         # router: "emergency", "chitchat" or "medical"
    search_query: str   # researcher: query used for search (may be rewritten)
    rewrites: int       # researcher: how many times the query was rewritten
    chunks: list        # researcher: chunks found
    found: bool         # researcher: are the chunks relevant?
    answer: str         # writer: the answer (final answer at the end)
    sources: list       # writer: citations shown in the UI
    feedback: str       # reviewer: problems found in the draft
    needs_fix: bool     # reviewer: should the writer try again?
    revisions: int      # reviewer: how many times the answer was sent back
    steps: list         # every agent adds what it did (shown in the UI)


# ---------- Supervisor decisions (which agent runs next) ----------

def after_router(state):
    return "research" if state["intent"] == "medical" else END


def after_research(state):
    return "writer" if state["found"] else END


def after_review(state):
    return "writer" if state["needs_fix"] else END


# ---------- Build the team ----------

team = StateGraph(TeamState)
team.add_node("router", router_agent)
team.add_node("research", research_agent)   # a whole subgraph used as one node
team.add_node("writer", writer_agent)
team.add_node("reviewer", reviewer_agent)

team.add_edge(START, "router")
team.add_conditional_edges("router", after_router, ["research", END])
team.add_conditional_edges("research", after_research, ["writer", END])
team.add_edge("writer", "reviewer")
team.add_conditional_edges("reviewer", after_review, ["writer", END])

supervisor = team.compile()


def ask(question):
    """Run the whole team on one question. Used by the API and the evaluation."""
    log.info(f"New question: '{question}'")
    start = time.time()
    start_state = {
        "question": question, "search_query": question, "rewrites": 0, "revisions": 0,
        "chunks": [], "answer": "", "sources": [], "feedback": "", "steps": [],
    }
    result = supervisor.invoke(start_state)
    log.info(f"Finished in {time.time() - start:.1f}s. Steps: {' -> '.join(result['steps'])}")
    return {"answer": result["answer"], "sources": result["sources"], "steps": result["steps"]}
