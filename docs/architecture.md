# Architecture

## Components

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React 18 + Vite + TypeScript, on Vercel | Uploads, per-document status, streaming chat, clickable citations |
| Backend | Python FastAPI, Docker on Railway | Validation, PDF extraction, chunking, embeddings, retrieval, Anthropic calls |
| Database | Supabase Postgres | `documents`, `chunks`, `messages` tables |
| Vector store | Supabase pgvector (HNSW, cosine) + Postgres full-text search (GIN) | Hybrid retrieval |
| LLM | Anthropic Messages API (streaming), with OpenAI Chat Completions as automatic fallback; server-side keys only | Grounded answer generation |
| Embeddings | fastembed ONNX `BAAI/bge-small-en-v1.5` (384 dims), in-process | Document and query vectors |

## Why the backend is not on Vercel

Vercel functions cap request bodies at 4.5 MB, have execution time limits and cold starts, and cannot keep a background ingestion task alive after the response is sent. Medical PDFs are often larger than 4.5 MB, and embedding takes seconds. The FastAPI backend therefore runs as a persistent container (Railway). The embedding model is loaded once at startup and baked into the image, uploads are not limited to 4.5 MB, and ingestion runs as a background task. Vercel serves only the static React build.

## Data flow

**Upload** (`POST /documents`, multipart, up to 10 files):
1. Each file is checked for extension or MIME type, size (25 MB) and the `%PDF-` signature. Rejected files come back with a reason and are never stored.
2. A SHA-256 hash dedupes within each workspace. Re-uploading the same PDF returns the existing document with no re-extraction and no re-embedding.
3. A `documents` row is created with `status = processing`, and the response returns immediately.
4. A background task (at most 2 run at once) extracts text page by page with pypdf. Corrupted, encrypted, empty or image-only PDFs raise a clear error.
5. Pages are split into sentence-aligned chunks of about 900 characters with 150 characters of overlap. **Chunks never span pages**, so each chunk has exactly one page number.
6. Chunks are embedded in batches and bulk-inserted, 250 rows per request. The status then becomes `ready` with page and chunk counts, or `failed` with the reason (partial chunks are deleted).
7. The frontend polls `GET /documents` every 1.5 s while any document is still processing.

**Question** (`POST /chat`, Server-Sent Events):
1. The last 6 messages are loaded. Short follow-up questions are expanded with the previous user question before retrieval.
2. The query is embedded (about 10 ms once warm), and `match_chunks` runs in one SQL round trip. It takes the top 40 pgvector cosine matches and the top 40 full-text matches over the workspace's `ready` documents, fuses them with Reciprocal Rank Fusion, and returns the top 6.
3. Chunks below the similarity floor (0.45) are discarded. If none remain, the app answers "not enough evidence" **without calling the LLM**, which saves cost and prevents hallucination.
4. Otherwise the chunks are sent to Claude labelled `[S1]` to `[S6]`, each with its document name and page. The system prompt is prompt-cached. It forbids outside knowledge, requires a label after every factual sentence, and requires the sentinel `INSUFFICIENT_EVIDENCE` when the sources do not answer the question.
5. Tokens stream to the browser. When the answer is complete, the backend removes any label that does not match a retrieved chunk and maps the remaining labels to `{filename, page_number, excerpt}`. **Citations are therefore always derived from retrieved content and can never be fabricated.**
6. The question and the answer (with its citations) are saved to `messages`.

## Citation flow

Each `[Sn]` in the answer points to the nth retrieved chunk, which carries its `document_id`, `filename`, `page_number` and text. In the UI each citation is a button labelled `file.pdf p.N`. Clicking it opens the exact excerpt that was used.

## Isolation and security

- The browser generates a random UUID workspace ID, stores it in localStorage and sends it as `X-Workspace-Id`. Every query filters by it. Workspace IDs are unguessable but not authenticated, so add Supabase Auth for multi-user production use (see README).
- Row Level Security (RLS) is enabled on all tables with no anon or authenticated policies, and only `service_role` can execute `match_chunks`. The anon key cannot read anything. Only the backend, which holds the service-role key server-side, can access data.
- The Anthropic key and the service-role key exist only in backend environment variables. The frontend holds only `VITE_API_URL`.

## Failure handling

| Failure | Behaviour |
|---|---|
| Invalid, empty, corrupted, encrypted, scanned or non-PDF file | Rejected or marked `failed`, with a readable reason |
| Supabase unreachable or returning errors | `503` with a message, and chat streams an `error` event |
| AI provider failure (no credits, invalid key, rate limit, timeout, outage, missing model) | The SDK retries twice with backoff (60 s timeout). If the provider still fails, the other provider (Claude or OpenAI, order set by `LLM_PRIMARY`) answers instead. A failure partway through an answer sends a `reset` event, so the partial text is discarded before the fallback streams. If both fail, the user sees a plain-language reason per provider (for example "the account has run out of credits"), while the server logs the technical detail. The server never crashes |
| No relevant evidence | A fixed "not enough evidence, consult a professional" response, with no LLM call |
| History save fails | The answer is still delivered |
