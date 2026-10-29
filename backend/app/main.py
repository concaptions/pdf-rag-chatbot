"""FastAPI backend for the Medical PDF RAG Chatbot."""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import db, embeddings, llm
from app.config import get_settings
from app.pdf import PDFError, chunk_pages, extract_pages

settings = get_settings()
logger = logging.getLogger("uvicorn.error")
_ingest_slots = asyncio.Semaphore(2)
_tasks: set[asyncio.Task] = set()
_active_docs: set[str] = set()  # documents queued or being indexed by this process
EMBED_BATCH = 128
DOC_FIELDS = "id,filename,status,error,page_count,chunk_count,progress,created_at,updated_at"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(embeddings.warmup)  # load the model once, not on the first request
    yield
    await db.close()


app = FastAPI(title="Medical PDF RAG Chatbot", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.origins, allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"]
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: str | None) -> datetime:
    """Parse a Postgres timestamptz; unparseable values count as 'now' (never stale)."""
    try:
        # Normalise fractional seconds to exactly 6 digits for fromisoformat.
        value = re.sub(r"\.(\d+)", lambda m: "." + m.group(1)[:6].ljust(6, "0"), value or "", count=1)
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


def workspace(x_workspace_id: str | None) -> str:
    try:
        return str(uuid.UUID(x_workspace_id or ""))
    except ValueError:
        raise HTTPException(400, "Missing or invalid X-Workspace-Id header.")


def db_guard(exc: db.DatabaseError):
    return HTTPException(503, "Storage is temporarily unavailable. Please try again shortly.")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    return llm.catalog()


# ---------------------------------------------------------------- documents


async def load_documents(ws: str, fields: str = DOC_FIELDS) -> list[dict]:
    docs = await db.select("documents", {"workspace_id": f"eq.{ws}", "select": fields, "order": "created_at.desc"})
    # A job whose worker died (restart or redeploy) never finishes; fail it so the user can retry.
    # Jobs queued or running in this process are alive even if they have not reported progress yet.
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.stale_processing_s)
    for d in docs:
        if d["status"] == "processing" and d["id"] not in _active_docs and parse_ts(d.get("updated_at")) < cutoff:
            d.update(status="failed", error="Processing stopped unexpectedly. Please upload the file again.")
            await db.update("documents", {"id": f"eq.{d['id']}"}, {"status": "failed", "error": d["error"]})
    return docs


@app.get("/documents")
async def list_documents(x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        return await load_documents(ws)
    except db.DatabaseError as exc:
        raise db_guard(exc)


async def set_progress(doc_id: str, **values) -> None:
    await db.update("documents", {"id": f"eq.{doc_id}"}, {**values, "updated_at": now_iso()})


async def process_document(doc_id: str, ws: str, data: bytes) -> None:
    async with _ingest_slots:
        try:
            await set_progress(doc_id, progress=2)
            pages = await asyncio.to_thread(extract_pages, data)
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            preview = " ".join(t for _, t in pages if t)[:700]
            await set_progress(doc_id, progress=10, page_count=len(pages), preview=preview)

            # Embed batch N+1 while batch N is being written, and report progress as we go.
            pending: asyncio.Task | None = None
            last_reported = 10
            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start : start + EMBED_BATCH]
                vectors = await embeddings.embed_documents([c.content for c in batch])
                rows = [
                    {"document_id": doc_id, "workspace_id": ws, "page_number": c.page_number,
                     "chunk_index": c.chunk_index, "content": c.content, "embedding": str(v)}
                    for c, v in zip(batch, vectors)
                ]
                if pending:
                    await pending
                pending = asyncio.create_task(db.insert("chunks", rows, returning=False))
                done = min(start + EMBED_BATCH, len(chunks))
                progress = 10 + int(88 * done / len(chunks))
                if progress - last_reported >= 4:
                    await set_progress(doc_id, progress=progress)
                    last_reported = progress
            if pending:
                await pending
            await set_progress(doc_id, status="ready", progress=100, chunk_count=len(chunks))
        except Exception as exc:
            msg = str(exc) if isinstance(exc, PDFError) else (
                "Storage was unavailable while indexing. Please upload the file again." if isinstance(exc, db.DatabaseError)
                else "Processing failed unexpectedly. Please upload the file again.")
            try:
                await db.delete("chunks", {"document_id": f"eq.{doc_id}"})
                await set_progress(doc_id, status="failed", error=msg[:500])
            except db.DatabaseError:
                pass


@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(...), x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    if len(files) > settings.max_files_per_request:
        raise HTTPException(400, f"Upload at most {settings.max_files_per_request} files at a time.")
    results = []
    for f in files:
        name = (f.filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1][:200]
        data = await f.read(settings.max_upload_mb * 1024 * 1024 + 1)
        error = None
        if not name.lower().endswith(".pdf") and f.content_type != "application/pdf":
            error = "Unsupported file type. Only PDF files are accepted."
        elif len(data) > settings.max_upload_mb * 1024 * 1024:
            error = f"File exceeds {settings.max_upload_mb} MB."
        elif not data:
            error = "The file is empty."
        elif data.lstrip()[:5] != b"%PDF-":
            error = "The file is not a valid PDF."
        if error:
            results.append({"filename": name, "status": "rejected", "error": error})
            continue
        sha = hashlib.sha256(data).hexdigest()
        try:
            existing = await db.select("documents", {"workspace_id": f"eq.{ws}", "sha256": f"eq.{sha}", "select": "id,filename,status,error"})
            if existing and existing[0]["status"] != "failed":
                results.append({**existing[0], "duplicate": True})  # no re-processing, no re-embedding
                continue
            if existing:
                await db.delete("documents", {"id": f"eq.{existing[0]['id']}"})
            doc = (await db.insert("documents", {"workspace_id": ws, "filename": name, "sha256": sha}))[0]
        except db.DatabaseError as exc:
            raise db_guard(exc)
        _active_docs.add(doc["id"])
        task = asyncio.create_task(process_document(doc["id"], ws, data))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        task.add_done_callback(lambda _t, i=doc["id"]: _active_docs.discard(i))
        results.append({"id": doc["id"], "filename": name, "status": "processing"})
    return results


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: uuid.UUID, x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        await db.delete("documents", {"id": f"eq.{doc_id}", "workspace_id": f"eq.{ws}"})
    except db.DatabaseError as exc:
        raise db_guard(exc)
    return {"deleted": str(doc_id)}


# ------------------------------------------------------------ conversations


@app.get("/conversations")
async def list_conversations(x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        return await db.select("conversations", {"workspace_id": f"eq.{ws}", "select": "id,title,created_at,updated_at",
                                                 "order": "updated_at.desc", "limit": "200"})
    except db.DatabaseError as exc:
        raise db_guard(exc)


async def owned_conversation(conv_id: uuid.UUID, ws: str) -> dict:
    rows = await db.select("conversations", {"id": f"eq.{conv_id}", "workspace_id": f"eq.{ws}", "select": "id,title"})
    if not rows:
        raise HTTPException(404, "Conversation not found.")
    return rows[0]


@app.get("/conversations/{conv_id}/messages")
async def conversation_messages(conv_id: uuid.UUID, x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        await owned_conversation(conv_id, ws)
        return await db.select("messages", {"conversation_id": f"eq.{conv_id}", "workspace_id": f"eq.{ws}",
                                            "select": "role,content,citations,provider,model,created_at",
                                            "order": "created_at.asc,id.asc", "limit": "500"})
    except db.DatabaseError as exc:
        raise db_guard(exc)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@app.patch("/conversations/{conv_id}")
async def rename_conversation(conv_id: uuid.UUID, req: RenameRequest, x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        await owned_conversation(conv_id, ws)
        await db.update("conversations", {"id": f"eq.{conv_id}"}, {"title": req.title.strip()})
    except db.DatabaseError as exc:
        raise db_guard(exc)
    return {"id": str(conv_id), "title": req.title.strip()}


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: uuid.UUID, x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        await db.delete("conversations", {"id": f"eq.{conv_id}", "workspace_id": f"eq.{ws}"})
    except db.DatabaseError as exc:
        raise db_guard(exc)
    return {"deleted": str(conv_id)}


# --------------------------------------------------------------------- chat


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None
    provider: str | None = None
    model: str | None = None


async def retrieve(question: str, ws: str, history: list[dict]) -> list[dict]:
    # Follow-up questions ("what about its side effects?") borrow the previous user turn for retrieval.
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    query = f"{prev} {question}" if prev and len(question.split()) < 8 else question
    vector = await embeddings.embed_query(query)
    rows = await db.rpc("match_chunks", {"query_embedding": str(vector), "query_text": query,
                                         "p_workspace_id": ws, "match_count": settings.top_k})
    return [r for r in rows if r["similarity"] >= settings.min_similarity]


def names(docs: list[dict]) -> str:
    quoted = [f"**{d['filename']}**" for d in docs]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def library_reply(docs: list[dict]) -> str | None:
    """A fixed, friendly reply when there is nothing indexed to answer from (no LLM call needed).

    Documents still indexing become searchable as soon as their first batch is written
    (progress above 10%), so those go to the model, which is told how far indexing has got.
    """
    if any(d["status"] == "ready" or (d["status"] == "processing" and (d.get("progress") or 0) > 10) for d in docs):
        return None
    if not docs:
        return ("I don't have any documents to work with yet. Upload one or more medical PDFs (clinical guidelines, "
                "drug labels, research papers or patient leaflets) and I'll answer your questions about them, "
                "with the document and page for every fact.")
    processing = [d for d in docs if d["status"] == "processing"]
    if processing:
        progress = ", ".join(f"{d['filename']} {d.get('progress') or 0}%" for d in processing)
        return (f"I'm still indexing {names(processing)} ({progress}). I'll be able to answer questions about "
                f"{'it' if len(processing) == 1 else 'them'} as soon as indexing finishes, usually within a minute or two "
                "for a long book. Feel free to ask again then.")
    return (f"None of your documents could be read: {names(docs)}. This usually means the PDF is scanned images "
            "without text, password-protected or damaged. Please upload a text-based PDF and I'll be ready to help.")


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)

    async def events():
        t0 = time.perf_counter()
        try:
            if req.conversation_id:
                conv = await owned_conversation(req.conversation_id, ws)
                history = await db.select("messages", {"conversation_id": f"eq.{conv['id']}", "select": "role,content",
                                                       "order": "created_at.desc,id.desc", "limit": "6"})
                history.reverse()
            else:
                title = " ".join(req.question.split())[:60] + ("..." if len(req.question) > 60 else "")
                conv = (await db.insert("conversations", {"workspace_id": ws, "title": title}))[0]
                history = []
                yield sse({"type": "conversation", "id": conv["id"], "title": conv["title"]})
            docs = await load_documents(ws, "id,filename,status,page_count,progress,preview,updated_at")
            fixed = library_reply(docs)
            chunks = [] if fixed else await retrieve(req.question, ws, history)
        except HTTPException as exc:
            yield sse({"type": "error", "message": exc.detail})
            return
        except db.DatabaseError:
            yield sse({"type": "error", "message": "Your documents couldn't be searched right now. Please try again in a moment."})
            return
        t_retrieval = time.perf_counter() - t0

        used: dict = {"provider": None, "model": None}
        if fixed:
            answer, citations = fixed, []
            yield sse({"type": "token", "text": answer})
        else:
            answer = ""
            try:
                async for kind, value in llm.stream_answer(req.question, chunks, history, docs, req.provider, req.model):
                    if kind == "start":
                        used = value
                        yield sse({"type": "model", **used, "model_name": llm.model_name(used["model"])})
                    elif kind == "reset":  # model failed mid-answer; the fallback starts over
                        answer = ""
                        yield sse({"type": "reset"})
                    else:
                        answer += value
                        yield sse({"type": "token", "text": value})
            except llm.AllProvidersFailed as exc:
                yield sse({"type": "error", "message": exc.user_message()})
                return
            answer = llm.strip_invalid_labels(answer, len(chunks)).strip()
            citations = llm.extract_citations(answer, chunks)
        try:
            # PostgREST bulk inserts require every row to have exactly the same keys.
            await db.insert("messages", [
                {"workspace_id": ws, "conversation_id": conv["id"], "role": "user", "content": req.question,
                 "citations": [], "provider": None, "model": None},
                {"workspace_id": ws, "conversation_id": conv["id"], "role": "assistant", "content": answer,
                 "citations": citations, "provider": used["provider"], "model": used["model"]},
            ], returning=False)
            await db.update("conversations", {"id": f"eq.{conv['id']}"}, {"updated_at": now_iso()})
        except db.DatabaseError as exc:
            logger.error("history_save_failed conversation=%s error=%s", conv["id"], exc)  # answer still delivered
        yield sse({"type": "done", "answer": answer, "citations": citations, "conversation_id": conv["id"],
                   "provider": used["provider"], "model": used["model"],
                   "model_name": llm.model_name(used["model"]) if used["model"] else None,
                   "timing": {"retrieval_ms": round(t_retrieval * 1000), "total_ms": round((time.perf_counter() - t0) * 1000)}})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
