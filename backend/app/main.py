"""FastAPI backend for the Medical PDF RAG Chatbot."""

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import db, embeddings, llm
from app.config import get_settings
from app.pdf import PDFError, chunk_pages, extract_pages

settings = get_settings()
_ingest_slots = asyncio.Semaphore(2)
_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(embeddings.warmup)  # load the model once, not on the first request
    yield
    await db.close()


app = FastAPI(title="Medical PDF RAG Chatbot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"]
)


def workspace(x_workspace_id: str | None) -> str:
    try:
        return str(uuid.UUID(x_workspace_id or ""))
    except ValueError:
        raise HTTPException(400, "Missing or invalid X-Workspace-Id header.")


def db_guard(exc: db.DatabaseError):
    return HTTPException(503, f"Storage is temporarily unavailable. {exc}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/documents")
async def list_documents(x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        return await db.select(
            "documents",
            {"workspace_id": f"eq.{ws}", "select": "id,filename,status,error,page_count,chunk_count,created_at",
             "order": "created_at.desc"},
        )
    except db.DatabaseError as exc:
        raise db_guard(exc)


async def process_document(doc_id: str, ws: str, data: bytes) -> None:
    async with _ingest_slots:
        try:
            pages = await asyncio.to_thread(extract_pages, data)
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            vectors = await embeddings.embed_documents([c.content for c in chunks])
            rows = [
                {"document_id": doc_id, "workspace_id": ws, "page_number": c.page_number,
                 "chunk_index": c.chunk_index, "content": c.content, "embedding": str(v)}
                for c, v in zip(chunks, vectors)
            ]
            for i in range(0, len(rows), 250):
                await db.insert("chunks", rows[i : i + 250], returning=False)
            await db.update("documents", {"id": f"eq.{doc_id}"},
                            {"status": "ready", "page_count": len(pages), "chunk_count": len(rows)})
        except Exception as exc:
            msg = str(exc) if isinstance(exc, (PDFError, db.DatabaseError)) else "Processing failed unexpectedly."
            try:
                await db.delete("chunks", {"document_id": f"eq.{doc_id}"})
                await db.update("documents", {"id": f"eq.{doc_id}"}, {"status": "failed", "error": msg[:500]})
            except db.DatabaseError:
                pass


@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(...), x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    if len(files) > settings.max_files_per_request:
        raise HTTPException(400, f"Upload at most {settings.max_files_per_request} files at a time.")
    results = []
    for f in files:
        name = (f.filename or "document.pdf").rsplit("/", 1)[-1][:200]
        data = await f.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if not name.lower().endswith(".pdf") and f.content_type != "application/pdf":
            results.append({"filename": name, "status": "rejected", "error": "Unsupported file type. Only PDF files are accepted."})
            continue
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            results.append({"filename": name, "status": "rejected", "error": f"File exceeds {settings.max_upload_mb} MB."})
            continue
        try:
            extract_pages_check = data.lstrip()[:5] == b"%PDF-"
            if not data or not extract_pages_check:
                raise PDFError("The file is empty." if not data else "The file is not a valid PDF.")
        except PDFError as exc:
            results.append({"filename": name, "status": "rejected", "error": str(exc)})
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
        task = asyncio.create_task(process_document(doc["id"], ws, data))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
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


@app.get("/messages")
async def list_messages(x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)
    try:
        return await db.select("messages", {"workspace_id": f"eq.{ws}", "select": "role,content,citations,created_at", "order": "created_at.asc", "limit": "200"})
    except db.DatabaseError as exc:
        raise db_guard(exc)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


async def retrieve(question: str, ws: str, history: list[dict]) -> list[dict]:
    # Follow-up questions ("what about its side effects?") borrow the previous user turn for retrieval.
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    query = f"{prev} {question}" if prev and len(question.split()) < 8 else question
    vector = await embeddings.embed_query(query)
    rows = await db.rpc("match_chunks", {"query_embedding": str(vector), "query_text": query,
                                         "p_workspace_id": ws, "match_count": settings.top_k})
    return [r for r in rows if r["similarity"] >= settings.min_similarity]


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, x_workspace_id: str | None = Header(None)):
    ws = workspace(x_workspace_id)

    async def events():
        t0 = time.perf_counter()
        try:
            history = await db.select("messages", {"workspace_id": f"eq.{ws}", "select": "role,content", "order": "created_at.desc", "limit": "6"})
            history.reverse()
            chunks = await retrieve(req.question, ws, history)
        except db.DatabaseError:
            yield sse({"type": "error", "message": "Document retrieval failed. Please try again."})
            return
        t_retrieval = time.perf_counter() - t0
        answer, provider, streamed = "", None, 0
        if chunks:
            try:
                async for kind, text in llm.stream_answer(req.question, chunks, history):
                    if kind == "provider":
                        provider = text
                        continue
                    if kind == "reset":  # provider failed mid-answer; the fallback starts over
                        answer, streamed = "", 0
                        yield sse({"type": "reset"})
                        continue
                    answer += text
                    # Hold text back while it could still be the INSUFFICIENT_EVIDENCE sentinel,
                    # then flush everything not yet sent.
                    if llm.SENTINEL.startswith(answer.strip()):
                        continue
                    yield sse({"type": "token", "text": answer[streamed:]})
                    streamed = len(answer)
            except llm.AllProvidersFailed as exc:
                yield sse({"type": "error", "message": exc.user_message()})
                return
        if not chunks or llm.SENTINEL in answer:
            answer, citations = llm.NO_EVIDENCE, []
        else:
            answer = llm.strip_invalid_labels(answer, len(chunks))
            citations = llm.extract_citations(answer, chunks)
        try:
            await db.insert("messages", [{"workspace_id": ws, "role": "user", "content": req.question},
                                         {"workspace_id": ws, "role": "assistant", "content": answer, "citations": citations}], returning=False)
        except db.DatabaseError:
            pass  # the answer is still delivered; only history persistence failed
        yield sse({"type": "done", "answer": answer, "citations": citations, "provider": provider,
                   "timing": {"retrieval_ms": round(t_retrieval * 1000), "total_ms": round((time.perf_counter() - t0) * 1000)}})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
