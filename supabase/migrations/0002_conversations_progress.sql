-- Adds chat history (conversations), indexing progress and document previews.
-- Safe to run more than once. Run in the Supabase SQL editor after 0001_init.sql.

-- Documents: indexing progress, last activity (for stuck-job detection) and a short preview
alter table public.documents add column if not exists progress int not null default 0;
alter table public.documents add column if not exists updated_at timestamptz not null default now();
alter table public.documents add column if not exists preview text;

update public.documents set progress = 100 where status = 'ready' and progress < 100;

update public.documents d
set preview = left(p.text, 700)
from (
  select document_id, string_agg(content, ' ' order by chunk_index) as text
  from (
    select document_id, content, chunk_index,
           row_number() over (partition by document_id order by chunk_index) as rn
    from public.chunks
  ) c
  where rn <= 2
  group by document_id
) p
where p.document_id = d.id and d.preview is null;

-- Conversations (chat history)
create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null,
  title text not null default 'New chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists conversations_workspace_idx on public.conversations (workspace_id, updated_at desc);
alter table public.conversations enable row level security;

alter table public.messages add column if not exists conversation_id uuid references public.conversations (id) on delete cascade;
alter table public.messages add column if not exists provider text;
alter table public.messages add column if not exists model text;
create index if not exists messages_conversation_idx on public.messages (conversation_id, created_at);

-- Move messages saved before chat history existed into one "Earlier chat" per workspace
insert into public.conversations (workspace_id, title, created_at, updated_at)
select m.workspace_id, 'Earlier chat', min(m.created_at), max(m.created_at)
from public.messages m
where m.conversation_id is null
group by m.workspace_id;

update public.messages m
set conversation_id = c.id
from public.conversations c
where m.conversation_id is null and c.workspace_id = m.workspace_id and c.title = 'Earlier chat';

-- Search documents while they are still indexing (chunks are written batch by batch),
-- so a long book can be asked about within seconds instead of after the whole job.
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
    where c.workspace_id = p_workspace_id and d.status in ('ready', 'processing')
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

notify pgrst, 'reload schema';
