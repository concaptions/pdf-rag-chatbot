# Medical PDF RAG Chatbot

**MedDocs AI**: upload multiple medical PDFs and ask questions about them. Claude (or OpenAI as a fallback) answers **only** from the uploaded documents, with verifiable citations (document name and page). When the documents do not cover a question, the assistant says so politely instead of guessing.

> This tool is not a substitute for professional medical advice, diagnosis or treatment.

Author: **Usama Navid**

## Stack

- **Frontend:** React 18 + Vite + TypeScript (`frontend/`), deployed on Vercel
- **Backend:** Python 3.12 FastAPI (`backend/`), Docker on Railway
- **Database and vectors:** Supabase Postgres + pgvector, with hybrid vector and full-text retrieval (`supabase/migrations/`)
- **LLM:** Anthropic Messages API (Claude Sonnet 5.5, Opus 5.5, Haiku 5.5) and OpenAI Chat Completions (GPT-4.1, GPT-5, GPT-5 mini), streaming, server-side keys only
- **Embeddings:** local ONNX `BAAI/bge-small-en-v1.5` via fastembed (no extra API key)
- **PDF extraction:** PDFium (`pypdfium2`), with `pypdf` as a fallback

See [docs/architecture.md](docs/architecture.md) for the full design, data flow, citation flow and failure handling.

## Features

- Multi-file PDF upload (button or drag and drop) with a live indexing progress bar per document
- Documents become searchable while they are still indexing, so a 1,000-page book can be asked about within seconds
- Page-level text extraction, with chunks that never span pages, so every citation points to one exact page
- Handling of invalid, empty, corrupted, password-protected, image-only and non-PDF files, and automatic failure of jobs interrupted by a server restart
- SHA-256 dedup, so re-uploading a file never re-embeds it
- Hybrid retrieval (pgvector HNSW + Postgres full-text search, fused with Reciprocal Rank Fusion) in a single SQL call
- Streaming Markdown answers over Server-Sent Events (SSE), with numbered citations and source chips that open the exact excerpt used
- Citations derived only from retrieved chunks; unknown labels are stripped
- Conversational replies: greetings, "what documents do you have?", out-of-scope questions and still-indexing documents all get gentle, accurate answers instead of a canned refusal
- Chat history: conversations are saved, grouped by date, and can be renamed or deleted
- Long-conversation memory: the model sees at least the last 30 messages word for word, and older messages are automatically compacted into a running summary in the background, so a chat can continue for hundreds of messages without slowing down or forgetting
- Topic switches and follow-ups both work: each question is searched on its own and together with the previous question, in parallel
- Collapsible sidebars: hide or show chat history and the library from the top bar (remembered per browser)
- Model picker with 3 models per provider and a live status badge per provider (Active, Ready, Not set up, or the reason it last failed)
- Automatic AI fallback: if the chosen model fails (no credits, bad key, rate limit, outage), the other provider answers, and a failure partway through an answer restarts cleanly. If both fail, the user sees a plain-language reason for each
- Stop generation, copy answer with citations, and a responsive layout (history and library become slide-over panels on small screens)

## Prerequisites

Python 3.12, Node 20+, a Supabase project and an Anthropic API key.

## Setup

### 1. Supabase
In the Supabase dashboard, open the SQL Editor and run the migrations in order:
1. [supabase/migrations/0001_init.sql](supabase/migrations/0001_init.sql): tables, the pgvector HNSW and full-text GIN indexes, Row Level Security (RLS) and the `match_chunks` function.
2. [supabase/migrations/0002_conversations_progress.sql](supabase/migrations/0002_conversations_progress.sql): chat history, indexing progress, document previews, and search while indexing.
3. [supabase/migrations/0003_conversation_memory.sql](supabase/migrations/0003_conversation_memory.sql): the rolling conversation summary. Until it is applied, chats still work with the 30+ message window, without summaries.

Every migration is safe to run more than once.

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
| `ANTHROPIC_MODEL` | backend | Default Claude model: `claude-sonnet-5-5` (or `claude-opus-5-5`, `claude-haiku-5-5`) |
| `OPENAI_API_KEY` | backend | OpenAI key (server-side only); enables the OpenAI models and the fallback |
| `OPENAI_MODEL` | backend | Default OpenAI model: `gpt-4.1` (or `gpt-5`, `gpt-5-mini`) |
| `LLM_PRIMARY` | backend | Provider selected by default: `anthropic` (default) or `openai` |
| `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` | backend | Per-request AI timeout (default 60 s) and retries (default 2) |
| `HISTORY_WINDOW`, `HISTORY_COMPACT_AT` | backend | Messages always sent verbatim (default 30), and the unsummarised count that triggers compaction (default 50) |
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
- The database-outage 503 path (without leaking internal details)
- Gentle replies with no LLM call when there are no documents or indexing has only just started
- Partly indexed documents being answerable, and stuck jobs being marked failed
- Meta questions ("what context do you have?") reaching the model with the document library
- Chat history: saving, follow-ups with context, listing, renaming, deleting, and isolation between workspaces
- Model selection, unknown-model fallback, and the 3-per-provider model catalog with status
- AI fallback in both directions, mid-answer failure reset, and plain-language messages when both providers fail
- Conversation memory: 30+ message window, history always starting on a user turn, old citations rendered as document and page, compaction folding exactly what is older than the window, summariser failure losing nothing, and a topic switch after a long thread still retrieving the new topic's passages

## Deployment

**Backend (Railway):** create a service from this repo with root directory `backend`. It uses `backend/Dockerfile` and `backend/railway.json`, with health check `/health`. Set the backend environment variables, then generate a public domain.

**Frontend (Vercel):** import the repo with root directory `frontend` (framework Vite, see `frontend/vercel.json`). Set `VITE_API_URL` to the Railway URL, then add the Vercel URL to the backend's `ALLOWED_ORIGINS`.

## Production hardening (not yet implemented)

- Replace the anonymous workspace ID with Supabase Auth, enforced by RLS policies on `auth.uid()`
- Store the original PDFs in Supabase Storage, so citations can open at the exact page
- Add an evaluation set for retrieval and answer quality (hit rate, citation precision, abstention accuracy)
- Faster indexing for very large books: more vCPUs on Railway, or a hosted embedding API (embedding runs at roughly 9 passages per second on the current instance)

## License

MIT, see [LICENSE](LICENSE). The project was originally derived from an MIT-licensed PDF RAG scaffold by Sohag Gain, and the application has since been rewritten.
