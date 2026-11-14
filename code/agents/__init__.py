"""
The team of agents. Each file is ONE agent with ONE job:

    router_agent.py    decides what kind of message this is (emergency / small talk / medical)
    research_agent.py  finds relevant chunks (search -> grade -> rewrite and retry)
    writer_agent.py    writes the answer with [n] citations
    reviewer_agent.py  fact-checks the answer against the sources

supervisor.py (one folder up) connects them into one LangGraph workflow.
"""
