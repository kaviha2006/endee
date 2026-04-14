"""
document_processor.py
Load uploaded files and split them into overlapping text chunks
suitable for embedding and vector storage.
"""
from __future__ import annotations

import io
from typing import IO, List, Dict

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks


def _split_text(text: str) -> List[str]:
    """Split a long string into overlapping chunks."""
    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF using PyPDF2."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def load_and_chunk_document(uploaded_file: IO) -> List[Dict]:
    """
    Accept a Streamlit UploadedFile (or any file-like object with .name and .read()),
    extract text, split into chunks, and return a list of dicts:
      { "text": str }
    """
    raw_bytes = uploaded_file.read()
    name: str = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = _extract_text_from_pdf(raw_bytes)
    elif name.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    raw_chunks = _split_text(text)
    return [{"text": chunk} for chunk in raw_chunks if len(chunk) > 30]
