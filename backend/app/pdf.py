"""PDF validation, page-level text extraction and page-aware chunking."""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


class PDFError(ValueError):
    """Raised for invalid, corrupted, encrypted, empty or unsupported PDFs."""


@dataclass
class Chunk:
    page_number: int
    chunk_index: int
    content: str


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    if not data:
        raise PDFError("The file is empty.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise PDFError("The file is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PDFError("The PDF is password-protected.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = re.sub(r"[ \t]+", " ", text).strip()
            pages.append((i, text))
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError(f"The PDF is corrupted or unreadable ({exc.__class__.__name__}).") from exc
    if not any(t for _, t in pages):
        raise PDFError("No extractable text found (the PDF may be a scanned image without OCR).")
    return pages


def chunk_pages(pages: list[tuple[int, str]], size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping chunks on sentence boundaries.

    Chunks never span pages, so every chunk maps to exactly one page citation.
    """
    chunks: list[Chunk] = []
    for page_number, text in pages:
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            while len(sentence) > size:  # very long run-on text
                head, sentence = sentence[:size], sentence[size - overlap :]
                if current:
                    chunks.append(Chunk(page_number, len(chunks), current))
                    current = ""
                chunks.append(Chunk(page_number, len(chunks), head))
            if current and len(current) + len(sentence) + 1 > size:
                chunks.append(Chunk(page_number, len(chunks), current))
                current = current[-overlap:].split(" ", 1)[-1] + " " + sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(Chunk(page_number, len(chunks), current))
    return chunks
