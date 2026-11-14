"""
Basic medical safety rules.

This is a simple keyword check, enough for a learning project.
A real hospital system would use a trained classifier and human review.
"""

EMERGENCY_WORDS = [
    "chest pain", "can't breathe", "cannot breathe", "not breathing", "unconscious",
    "heart attack", "stroke", "severe bleeding", "overdose", "seizure",
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
]

EMERGENCY_MESSAGE = (
    "🚨 **This sounds like it could be a medical emergency.**\n\n"
    "Please call your local emergency number now (**112 / 108** in India, **911** in the US) "
    "or go to the nearest emergency department.\n\n"
    "If you are having thoughts of suicide or self-harm, please contact a crisis helpline "
    "(**Tele-MANAS 14416** in India, **988** in the US) or someone you trust right now."
)

# Added to the end of every normal answer
DISCLAIMER = (
    "\n\n---\n⚠️ *For educational purposes only. Not a substitute for professional "
    "medical advice. Please consult a qualified healthcare provider.*"
)

# Used when the documents don't contain the answer (better than guessing)
NOT_FOUND_MESSAGE = (
    "I couldn't find information about this in the uploaded medical documents. "
    "Try rephrasing your question, upload a relevant document, or consult a healthcare professional."
)


# Reply to greetings and small talk (no search needed)
CHITCHAT_MESSAGE = (
    "Hi! 👋 I'm **MediGuide AI**. Ask me a question about your uploaded medical documents, "
    "for example: *What lifestyle changes help lower blood pressure?*"
)

# Shown on top of an answer that the Reviewer agent could not fully verify
UNVERIFIED_NOTE = (
    "⚠️ *Our fact-checker could not fully verify this answer against the sources. "
    "Please check the cited pages.*\n\n"
)


def is_emergency(question):
    """True if the question contains any emergency word."""
    question = question.lower()
    return any(word in question for word in EMERGENCY_WORDS)
