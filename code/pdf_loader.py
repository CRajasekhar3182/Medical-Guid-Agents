"""
Step 1 of the pipeline: read a PDF and cut it into small chunks.

Why chunks? A whole PDF is too big to search or to put in a prompt.
Small chunks let us find and send only the parts that matter.
"""
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

log = logging.getLogger("pdf_loader")


def load_and_split_pdf(file_path):
    """Return a list of chunks. Each chunk remembers its file name and page number."""
    # 1. Read the PDF -> one Document per page
    pages = PyPDFLoader(str(file_path)).load()

    # 2. Keep only the metadata we need for citations.
    #    PyPDFLoader counts pages from 0, people count from 1.
    file_name = Path(file_path).name
    for page in pages:
        page.metadata = {"source": file_name, "page": page.metadata.get("page", 0) + 1}

    # 3. Split pages into chunks. It tries to cut at paragraphs first,
    #    then lines, then sentences, so chunks stay readable.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)

    # 4. Skip near-empty chunks (blank pages, page numbers only)
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
    log.info(f"{file_name}: {len(pages)} pages -> {len(chunks)} chunks")
    return chunks
