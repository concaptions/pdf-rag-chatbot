# Architecture

## Components

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React 18 + Vite + TypeScript, on Vercel | Chat history, document library with indexing progress, model picker, streaming Markdown answers, clickable citations |
| Backend | Python FastAPI, Docker on Railway | Validation, PDF extraction, chunking, embeddings, retrieval, model calls with fallback, chat history |
| Database | Supabase Postgres | `documents`, `chunks`, `conversations`, `messages` tables |
| Vector store | Supabase pgvector (HNSW, cosine) + Postgres full-text search (GIN) | Hybrid retrieval |
| LLM | Anthropic Messages API and OpenAI Chat Completions, streaming; server-side keys only | Grounded answer generation, 3 selectable models per provider, automatic cross-provider fallback |
| Embeddings | fastembed ONNX `BAAI/bge-small-en-v1.5` (384 dims), in-process | Document and query vectors |
| PDF extraction | PDFium (`pypdfium2`), `pypdf` fallback | Page-level text |

## Why the backend is not on Vercel

Vercel functions cap request bodies at 4.5 MB, have execution time limits and cold starts, and cannot keep a background ingestion task alive after the response is sent. Medical PDFs are often larger than 4.5 MB, and embedding a large book takes minutes. The FastAPI backend therefore runs as a persistent container (Railway). The embedding model is loaded once at startup and baked into the image, uploads are not limited to 4.5 MB, and ingestion runs as a background task. Vercel serves only the static React build.

## Data flow

**Upload** (`POST /documents`, multipart, up to 10 files):
1. Each file is checked for extension or MIME type, size (25 MB) and the `%PDF-` signature. Rejected files come back with a reason and are never stored.
2. A SHA-256 hash dedupes within each workspace. Re-uploading the same PDF returns the existing document with no re-extraction and no re-embedding.
3. A `documents` row is created with `status = processing`, and the response returns immediately.
4. A background task (at most 2 run at once) extracts text page by page with PDFium, which is several times faster than pure-Python parsers on large books. Words hyphenated across lines are re-joined and layout line breaks removed. Corrupted, encrypted, empty or image-only PDFs raise a clear error.
5. Pages are split into sentence-aligned chunks of about 900 characters with 150 characters of overlap. **Chunks never span pages**, so each chunk has exactly one page number. The first 700 characters are stored as the document's preview.
6. Chunks are embedded in batches of 128. While one batch is written to the database, the next is embedded. Progress (0 to 100) and `updated_at` are written after each batch, and the status becomes `ready`, or `failed` with the reason (partial chunks are deleted).
7. **Chunks are searchable as soon as they are written**: retrieval includes `processing` documents, so a long book can be asked about while it is still indexing.
8. The frontend polls `GET /documents` every 1.5 s while anything is processing and shows a progress bar. A `processing` document with no progress for 3 minutes that is not running in this server process (its worker died in a restart) is marked `failed` with "please upload again".

**Question** (`POST /chat`, Server-Sent Events):
1. Without a `conversation_id`, a conversation is created and titled from the question, and a `conversation` event is sent first. With one, the backend checks it belongs to the workspace and loads its last 6 messages. Short follow-up questions are expanded with the previous user question before retrieval.
2. The document library (names, status, page counts, progress, previews) is loaded. If nothing is searchable yet, a fixed friendly reply is sent with no model call: "upload a PDF", "still indexing book.pdf (45%)", or "none of your documents could be read".
3. The query is embedded (about 10 ms once warm), and `match_chunks` runs in one SQL round trip. It takes the top 40 pgvector cosine matches and the top 40 full-text matches, fuses them with Reciprocal Rank Fusion, and returns the top 8. Passages below a 0.3 similarity floor are dropped as noise.
4. The model receives `<library>` plus the passages labelled `[S1]` to `[S8]`, each with its document name and page, even when no passage matched. So it can answer greetings, "what documents do you have?" and out-of-scope questions accurately and politely. The system prompt is prompt-cached. It forbids outside knowledge, requires a citation label after every factual sentence, and covers partial answers, still-indexing documents and medical safety (no diagnosis or personal dosing, emergency guidance).
5. The user's chosen provider and model are validated against the catalog. If that model fails, the other provider's default model answers (see Failure handling). A `model` event tells the UI which model is writing.
6. Tokens stream to the browser. When the answer is complete, the backend removes any label that does not match a retrieved chunk and maps the remaining labels to `{filename, page_number, excerpt}`. **Citations are therefore always derived from retrieved content and can never be fabricated.**
7. The question and the answer (with citations, provider and model) are saved to `messages`, and the conversation's `updated_at` moves it to the top of the history.

## Citation flow

Each `[Sn]` in the answer points to the nth retrieved chunk, which carries its `document_id`, `filename`, `page_number` and text. The UI numbers citations 1, 2, 3 in order of first appearance (the same order the server uses), renders each as a small inline button, and lists them as source chips (`file.pdf p. N`) under the answer. Clicking either opens the exact excerpt that was used.

## Models

`GET /models` returns 3 models per provider: Claude Sonnet 5.5 (default), Opus 5.5 and Haiku 5.5; GPT-4.1 (default), GPT-5 and GPT-5 mini. Each provider has a status: `not_configured` (no API key on the server), `unknown` (configured, not used yet), `ok` (last call succeeded) or `error` with a plain-language reason. GPT-5 models get `reasoning_effort: low` and a larger completion budget, because their reasoning uses completion tokens.

## Isolation and security

- The browser generates a random UUID workspace ID, stores it in localStorage and sends it as `X-Workspace-Id`. Every query, including conversations and messages, filters by it. Workspace IDs are unguessable but not authenticated, so add Supabase Auth for multi-user production use (see README).
- Row Level Security (RLS) is enabled on all tables with no anon or authenticated policies, and only `service_role` can execute `match_chunks`. The anon key cannot read anything. Only the backend, which holds the service-role key server-side, can access data.
- The AI keys and the service-role key exist only in backend environment variables. The frontend holds only `VITE_API_URL`.

## Failure handling

| Failure | Behaviour |
|---|---|
| Invalid, empty, corrupted, encrypted, scanned or non-PDF file | Rejected or marked `failed`, with a readable reason |
| Indexing interrupted by a restart or redeploy | Marked `failed` after 3 minutes without progress, with "please upload again" |
| Supabase unreachable or returning errors | `503` with a plain message (no internal details), and chat streams an `error` event |
| AI provider failure (no credits, invalid key, rate limit, timeout, outage, missing model) | The SDK retries twice with backoff (60 s timeout). If the model still fails, the other provider answers instead. A failure partway through an answer sends a `reset` event, so the partial text is discarded before the fallback streams. If both fail, the user sees a plain-language reason per provider (for example "the account has run out of credits"), while the server logs the technical detail. The server never crashes |
| Nothing searchable yet (no documents, indexing just started, all failed) | A fixed, friendly explanation, with no model call |
| Question not covered by the documents | The model says so politely, names what the documents do cover, and does not answer from general knowledge |
| History save fails | The answer is still delivered, and the error is logged |
