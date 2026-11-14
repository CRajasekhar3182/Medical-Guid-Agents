"""
Agent 1: ROUTER. The front desk.

It looks at the message and decides where it should go:
- emergency -> reply with helpline numbers right away (keyword check, no AI needed, instant)
- chitchat  -> "hi", "thanks"... reply with a friendly message, no search (saves time)
- medical   -> send it to the Research agent
"""
import logging
from config import llm
from safety import is_emergency, EMERGENCY_MESSAGE, CHITCHAT_MESSAGE

log = logging.getLogger("router_agent")

ROUTER_PROMPT = """You are the router of a medical document assistant.
Classify the user's message with one word:
- medical: any question about health, diseases, symptoms, drugs, treatment, guidelines or the documents
- chitchat: greetings, thanks, small talk, or questions about the assistant itself
Reply with only one word: medical or chitchat.

Message: {question}"""


def router_agent(state):
    question = state["question"]

    # 1. Emergencies are checked with keywords first: fast and never misses a listed word
    if is_emergency(question):
        log.warning("Emergency words detected -> emergency message")
        return {"intent": "emergency", "answer": EMERGENCY_MESSAGE, "sources": [],
                "steps": state["steps"] + ["router: emergency"]}

    # 2. Ask Gemini: medical question or small talk? (if unsure, treat it as medical)
    reply = llm.invoke(ROUTER_PROMPT.format(question=question)).text.strip().lower()
    intent = "chitchat" if reply.startswith("chitchat") else "medical"
    log.info(f"Message classified as: {intent}")

    if intent == "chitchat":
        return {"intent": intent, "answer": CHITCHAT_MESSAGE, "sources": [],
                "steps": state["steps"] + ["router: chitchat"]}
    return {"intent": intent, "steps": state["steps"] + ["router: medical"]}
