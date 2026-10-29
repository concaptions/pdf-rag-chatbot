-- Medical PDF RAG Chatbot schema. Run once in the Supabase SQL editor.
create extension if not exists vector;

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null,
  filename text not null,
  sha256 text not null,
  status text not null default 'processing' check (status in ('processing', 'ready', 'failed')),
  error text,
  page_count int,
  chunk_count int,
  created_at timestamptz not null default now(),
  unique (workspace_id, sha256)
);
create index if not exists documents_workspace_idx on public.documents (workspace_id, created_at desc);

create table if not exists public.chunks (
  id bigint generated always as identity primary key,
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null,
  page_number int not null,
  chunk_index int not null,
  content text not null,
  embedding vector(384) not null,
  fts tsvector generated always as (to_tsvector('english', content)) stored
);
create index if not exists chunks_workspace_idx on public.chunks (workspace_id);
create index if not exists chunks_document_idx on public.chunks (document_id);
create index if not exists chunks_embedding_idx on public.chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx on public.chunks using gin (fts);

create table if not exists public.messages (
  id bigint generated always as identity primary key,
  workspace_id uuid not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  citations jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists messages_workspace_idx on public.messages (workspace_id, created_at);

-- Access control: RLS on, no policies for anon/authenticated. Only the backend
-- (service_role key, server-side) can read or write. The frontend never talks
-- to the database directly.
alter table public.documents enable row level security;
alter table public.chunks enable row level security;
alter table public.messages enable row level security;

-- Hybrid retrieval: pgvector cosine similarity + Postgres full-text search,
-- fused with Reciprocal Rank Fusion. Always scoped to one workspace and to
-- ready documents only.
create or replace function public.match_chunks(
  query_embedding vector(384),
  query_text text,
  p_workspace_id uuid,
  match_count int default 6
)
returns table (id bigint, document_id uuid, filename text, page_number int, content text, similarity float, score float)
language sql stable
set search_path = public, extensions
as $$
  with scoped as (
    select c.* from chunks c
    join documents d on d.id = c.document_id
    where c.workspace_id = p_workspace_id and d.status = 'ready'
  ),
  semantic as (
    select s.id, row_number() over (order by s.embedding <=> query_embedding) as rnk,
           1 - (s.embedding <=> query_embedding) as sim
    from scoped s order by s.embedding <=> query_embedding limit 40
  ),
  keyword as (
    select s.id, row_number() over (order by ts_rank_cd(s.fts, q) desc) as rnk
    from scoped s, websearch_to_tsquery('english', query_text) q
    where s.fts @@ q order by ts_rank_cd(s.fts, q) desc limit 40
  ),
  fused as (
    select coalesce(se.id, k.id) as id,
           coalesce(1.0 / (60 + se.rnk), 0) + coalesce(1.0 / (60 + k.rnk), 0) as score,
           se.sim
    from semantic se full outer join keyword k on se.id = k.id
  )
  select c.id, c.document_id, d.filename, c.page_number, c.content,
         coalesce(f.sim, 1 - (c.embedding <=> query_embedding)) as similarity, f.score
  from fused f join chunks c on c.id = f.id join documents d on d.id = c.document_id
  order by f.score desc limit match_count;
$$;

revoke execute on function public.match_chunks(vector, text, uuid, int) from public, anon, authenticated;
grant execute on function public.match_chunks(vector, text, uuid, int) to service_role;
