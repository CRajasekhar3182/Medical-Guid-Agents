"""
The web app (FastAPI). It does two jobs:
1. Serves the chat page from the ui/ folder   -> http://127.0.0.1:8001
2. Provides the API that the page calls        -> /upload and /ask

Start it from the mediguide-ai folder:
    uvicorn main:app --reload --app-dir code --port 8001
"""
import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATA_DIR
from pdf_loader import load_and_split_pdf
from vector_db import save_chunks, has_documents
from supervisor import ask

UI_DIR = Path(__file__).parent / "ui"
log = logging.getLogger("main")

app = FastAPI(title="MediGuide AI")

# Makes ui/style.css and ui/script.js available at /ui/style.css and /ui/script.js
app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@app.get("/")
def chat_page():
    """Open http://127.0.0.1:8001 in the browser to see the chat page."""
    return FileResponse(UI_DIR / "index.html")


class Question(BaseModel):
    question: str


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Save the PDF in data/, split it into chunks and store them in Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    log.info(f"Upload received: {file.filename}")
    DATA_DIR.mkdir(exist_ok=True)
    pdf_path = DATA_DIR / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_split_pdf(pdf_path)
    save_chunks(chunks)
    return {"file": file.filename, "chunks": len(chunks)}


@app.post("/ask")
def ask_question(body: Question):
    """Run the multi-agent team and return the answer, sources and steps."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not has_documents():
        raise HTTPException(status_code=400, detail="No documents yet. Upload a PDF first.")
    return ask(body.question)


@app.exception_handler(Exception)
def show_error(request: Request, error: Exception):
    """If anything crashes (e.g. Gemini is busy, Qdrant is off), log it and send the reason to the page."""
    log.exception(f"Error while handling {request.url.path}")  # full error details go to the log
    return JSONResponse(status_code=500, content={"detail": f"{type(error).__name__}: {error}"})
