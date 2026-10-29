import io

import pytest
from reportlab.pdfgen import canvas

from app import llm
from app.pdf import PDFError, chunk_pages, extract_pages


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


def test_extracts_text_with_page_numbers():
    pages = extract_pages(make_pdf(["Metformin is first-line therapy.", "Lisinopril lowers blood pressure."]))
    assert [p for p, _ in pages] == [1, 2]
    assert "Metformin" in pages[0][1] and "Lisinopril" in pages[1][1]


@pytest.mark.parametrize("data,msg", [(b"", "empty"), (b"hello world", "not a valid PDF"), (b"%PDF-1.4 garbage", "corrupted")])
def test_rejects_invalid_files(data, msg):
    with pytest.raises(PDFError, match=msg):
        extract_pages(data)


def test_rejects_pdf_without_text():
    with pytest.raises(PDFError, match="No extractable text"):
        extract_pages(make_pdf(["", ""]))


def test_chunks_never_span_pages_and_respect_size():
    long = " ".join(f"Sentence number {i} about hypertension." for i in range(200))
    chunks = chunk_pages([(1, long), (2, "Short page two.")], size=300, overlap=50)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(len(c.content) <= 300 for c in chunks)
    assert chunks[-1].page_number == 2 and chunks[-1].content == "Short page two."


CHUNKS = [
    {"document_id": "d1", "filename": "a.pdf", "page_number": 3, "content": "Metformin dose is 500 mg."},
    {"document_id": "d2", "filename": "b.pdf", "page_number": 7, "content": "Aspirin reduces fever."},
]


def test_citations_come_only_from_retrieved_chunks():
    answer = "Start at 500 mg [S1]. Aspirin helps [S2][S2]. Made up [S9]."
    cleaned = llm.strip_invalid_labels(answer, len(CHUNKS))
    assert "[S9]" not in cleaned
    cites = llm.extract_citations(cleaned, CHUNKS)
    assert [(c["filename"], c["page_number"]) for c in cites] == [("a.pdf", 3), ("b.pdf", 7)]


def test_prompt_labels_sources_with_document_and_page():
    msgs = llm.build_messages("dose?", CHUNKS, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert msgs[-1]["role"] == "user"
    assert 'label="S1" document="a.pdf" page="3"' in msgs[-1]["content"]
    assert len(msgs) == 3
