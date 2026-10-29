# Medical PDF RAG Chatbot

Upload multiple medical PDFs and ask questions about them. Anthropic Claude generates answers **only** from the uploaded documents, with verifiable citations (document name and page). When the documents do not contain enough evidence, the assistant says so instead of guessing.

> This tool is not a substitute for professional medical advice, diagnosis or treatment.

Author: **Usama Navid**

## Stack

- **Frontend:** React 18 + Vite + TypeScript (`frontend/`), deployed on Vercel
- **Backend:** Python 3.12 FastAPI (`backend/`), Docker on Railway
- **Database and vectors:** Supabase Postgres + pgvector, with hybrid vector and full-text retrieval (`supabase/migrations/`)
- **LLM:** Anthropic Messages API, streaming, with a server-side key
- **Embeddings:** local ONNX `BAAI/bge-small-en-v1.5` via fastembed (no extra API key)

See [docs/architecture.md](docs/architecture.md) for the full design, data flow, citation flow and failure handling.

## Features

- Multi-file PDF upload with per-file status: processing, ready, or failed with a reason
- Page-level text extraction, with chunks that never span pages, so every citation points to one exact page
- Handling of invalid, empty, corrupted, password-protected, image-only and non-PDF files
- SHA-256 dedup, so re-uploading a file never re-embeds it
- Hybrid retrieval (pgvector HNSW + Postgres full-text search, fused with Reciprocal Rank Fusion) in a single SQL call
- Streaming answers over Server-Sent Events (SSE), with clickable citations that show the exact source excerpt
- Citations derived only from retrieved chunks; unknown labels are stripped
- A "not enough evidence" response, with no LLM call, when retrieval finds nothing relevant
- Persistent conversation history, and follow-up questions that use the earlier context
- Graceful handling of Anthropic rate limits and timeouts, database outages and invalid uploads

## Prerequisites

Python 3.12, Node 20+, a Supabase project and an Anthropic API key.

## Setup

### 1. Supabase
In the Supabase dashboard, open the SQL Editor, paste [supabase/migrations/0001_init.sql](supabase/migrations/0001_init.sql) and run it. It creates the tables, the pgvector HNSW and full-text GIN indexes, Row Level Security (RLS) and the `match_chunks` function.

### 2. Backend
```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # fill in ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
uvicorn app.main:app --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
cp .env.example .env.local    # VITE_API_URL=http://localhost:8000
npm run dev                   # http://localhost:5173
```

## Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | backend | Claude API key (server-side only) |
| `ANTHROPIC_MODEL` | backend | Default `claude-opus-4-8` |
| `SUPABASE_URL` | backend | `https://<ref>.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | backend | Server-side only; never ship it to the browser |
| `ALLOWED_ORIGINS` | backend | Comma-separated CORS origins, e.g. your Vercel URL |
| `TOP_K`, `MIN_SIMILARITY`, `MAX_UPLOAD_MB` | backend | Retrieval and upload tuning |
| `VITE_API_URL` | frontend | Public backend URL |

## Testing

```bash
cd backend && python -m pytest -q
cd frontend && npm run build    # type-check + production build
```
The tests cover:
- PDF extraction with page numbers
- Invalid, corrupted and empty PDFs
- Page-bounded chunking
- Citation mapping and stripping of fabricated labels
- Prompt construction
- File-type rejection
- The database-outage 503 path
- The no-evidence path, which asserts that the LLM is not called

## Deployment

**Backend (Railway):** create a service from this repo with root directory `backend`. It uses `backend/Dockerfile` and `backend/railway.json`, with health check `/health`. Set the backend environment variables, then generate a public domain.

**Frontend (Vercel):** import the repo with root directory `frontend` (framework Vite, see `frontend/vercel.json`). Set `VITE_API_URL` to the Railway URL, then add the Vercel URL to the backend's `ALLOWED_ORIGINS`.

## Production hardening (not yet implemented)

- Replace the anonymous workspace ID with Supabase Auth, enforced by RLS policies on `auth.uid()`
- Store the original PDFs in Supabase Storage, so citations can open at the exact page
- Add an evaluation set for retrieval and answer quality (hit rate, citation precision, abstention accuracy)

## License

MIT, see [LICENSE](LICENSE). The project was originally derived from an MIT-licensed PDF RAG scaffold by Sohag Gain, and the application has since been rewritten.
