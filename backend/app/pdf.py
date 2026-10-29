"""PDF validation, page-level text extraction and page-aware chunking.

Extraction uses PDFium (pypdfium2, the engine behind Chrome's PDF viewer),
which is several times faster than pure-Python parsers on large books, and
falls back to pypdf for files PDFium cannot open.
"""

import io
import re
from dataclasses import dataclass

import pypdfium2 as pdfium
from pypdf import PdfReader


class PDFError(ValueError):
    """Raised for invalid, corrupted, encrypted, empty or unsupported PDFs."""


@dataclass
class Chunk:
    page_number: int
    chunk_index: int
    content: str


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single line breaks are layout, not structure
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def _extract_pdfium(data: bytes) -> list[tuple[int, str]]:
    pdf = pdfium.PdfDocument(data)
    try:
        pages = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded()
                finally:
                    textpage.close()
            except Exception:
                text = ""
            finally:
                page.close()
            pages.append((i + 1, clean_text(text)))
        return pages
    finally:
        pdf.close()


def _extract_pypdf(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise PDFError("The PDF is password-protected.")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, clean_text(text)))
    return pages


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    if not data:
        raise PDFError("The file is empty.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise PDFError("The file is not a valid PDF.")
    try:
        pages = _extract_pdfium(data)
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise PDFError("The PDF is password-protected.") from exc
        try:
            pages = _extract_pypdf(data)
        except PDFError:
            raise
        except Exception as exc2:
            raise PDFError("The PDF is corrupted or unreadable.") from exc2
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
