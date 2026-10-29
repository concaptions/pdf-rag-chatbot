-- Conversation memory: a rolling summary of messages older than the verbatim window.
-- Safe to run more than once. Run in the Supabase SQL editor after 0002.

alter table public.conversations add column if not exists summary text;
-- id of the newest message folded into the summary (messages.id is an identity column)
alter table public.conversations add column if not exists summary_upto bigint not null default 0;

create index if not exists messages_conversation_id_idx on public.messages (conversation_id, id);

notify pgrst, 'reload schema';
