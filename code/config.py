"""
All settings in one place.

Every other file imports what it needs from here, so you only change
model names, paths or the Qdrant address in this file (or in .env).
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Project folder = the folder above "code/"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"          # uploaded PDFs are saved here

# Read secrets and settings from the .env file
load_dotenv(PROJECT_DIR / ".env")

# ---------- Logging ----------
# Every step prints a line like:  14:02:31 | INFO | search | Hybrid search found 10 candidates
# Lines go to the terminal AND to mediguide.log (so you can read them later).
LOG_FILE = PROJECT_DIR / "mediguide.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
# Hide the noisy "HTTP Request: POST ..." lines from libraries
for noisy in ["httpx", "httpcore", "google_genai", "flashrank"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_docs")

# Chunking: how big each piece of text is, and how much neighbours overlap
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Search: fetch 10 candidates, keep the best 4 after reranking
SEARCH_CANDIDATES = 10
TOP_RESULTS = 4

# Gemini LLM (temperature=0 -> same question gives a stable, factual answer)
# thinking_level="minimal": skip Gemini's slow "thinking" step (about 2x faster; our tasks are simple)
# timeout + max_retries: if Gemini is slow or busy, retry twice, then give up instead of hanging forever
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    temperature=0, thinking_level="minimal", timeout=60, max_retries=2,
)

# Gemini embedding model (turns text into a list of numbers that captures meaning)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"))
