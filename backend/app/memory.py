"""Conversation memory: a verbatim window of recent messages plus a rolling summary.

The model always sees at least the last `history_window` messages word for word.
Once a conversation has `history_compact_at` messages that are not yet summarised,
everything older than the window is folded into `conversations.summary` by a fast
model. This runs in the background after the reply has been streamed, so it never
slows an answer down, and a chat can continue indefinitely without losing context.
"""

import asyncio
import logging
import re

from app import db, llm
from app.config import get_settings

logger = logging.getLogger("uvicorn.error")
CITE_RE = re.compile(r"\s*\[(S\d+)\]")  # includes the space before a label, so "daily [S1]." -> "daily (file, p. 4)."

_locks: dict[str, asyncio.Lock] = {}
_tasks: set[asyncio.Task] = set()


def render(content: str, citations: list[dict] | None) -> str:
    """Turn [Sn] labels from an earlier turn into '(file, p. N)'.

    Old labels must not reach the model as labels: they would be confused with the
    current turn's sources. Keeping the document and page preserves the provenance.
    """
    by_label = {c.get("label"): c for c in citations or []}

    def sub(m: re.Match) -> str:
        c = by_label.get(m.group(1))
        return f" ({c['filename']}, p. {c['page_number']})" if c else ""

    return CITE_RE.sub(sub, content or "").strip()


def as_history(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": render(m["content"], m.get("citations"))} for m in messages]


async def _load_summary(conv_id: str) -> tuple[str, int]:
    try:
        rows = await db.select("conversations", {"id": f"eq.{conv_id}", "select": "summary,summary_upto"})
    except db.DatabaseError:  # migration 0003 not applied yet: fall back to the verbatim window only
        return "", 0
    if not rows:
        return "", 0
    return rows[0].get("summary") or "", int(rows[0].get("summary_upto") or 0)


async def load(conv_id: str) -> tuple[str, list[dict]]:
    """Return (summary of older messages, unsummarised recent messages oldest first)."""
    s = get_settings()
    summary, upto = await _load_summary(conv_id)
    rows = await db.select("messages", {"conversation_id": f"eq.{conv_id}", "id": f"gt.{upto}",
                                        "select": "id,role,content,citations", "order": "id.desc",
                                        "limit": str(s.history_compact_at)})
    rows.reverse()
    while rows and rows[0]["role"] != "user":  # the model's history must start with a user turn
        rows.pop(0)
    return summary, rows


def maybe_compact(conv_id: str, unsummarised: int) -> None:
    """Schedule background compaction once the unsummarised history reaches the threshold."""
    if unsummarised < get_settings().history_compact_at:
        return
    task = asyncio.create_task(compact(conv_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def compact(conv_id: str) -> bool:
    """Fold everything older than the verbatim window into the conversation summary."""
    s = get_settings()
    lock = _locks.setdefault(conv_id, asyncio.Lock())
    if lock.locked():  # a compaction for this conversation is already running
        return False
    async with lock:
        try:
            summary, upto = await _load_summary(conv_id)
            rows = await db.select("messages", {"conversation_id": f"eq.{conv_id}", "id": f"gt.{upto}",
                                                "select": "id,role,content,citations", "order": "id.asc", "limit": "1000"})
            if len(rows) < s.history_compact_at:
                return False
            cut = len(rows) - s.history_window
            while cut < len(rows) and rows[cut]["role"] != "user":  # keep the window starting on a user turn
                cut += 1
            fold = rows[:cut]
            if not fold:
                return False
            new_summary = await llm.summarize(summary, as_history(fold))
            if not new_summary:
                return False  # both summarisers failed; the window still works and the next turn retries
            await db.update("conversations", {"id": f"eq.{conv_id}"},
                            {"summary": new_summary, "summary_upto": fold[-1]["id"]})
            logger.info("conversation_compacted conversation=%s folded=%d kept=%d", conv_id, len(fold), len(rows) - cut)
            return True
        except db.DatabaseError as exc:
            logger.warning("conversation_compaction_failed conversation=%s error=%s", conv_id, exc)
            return False
