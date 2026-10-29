import { Component, StrictMode, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { Answer, Icon, LogoMark, ModelPicker, numberCitations, type Catalog, type Choice, type Citation } from "./ui";

const API = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

type Doc = { id?: string; filename: string; status: string; error?: string | null; page_count?: number; chunk_count?: number; progress?: number };
type Conversation = { id: string; title: string; updated_at: string };
type Msg = {
  role: "user" | "assistant"; content: string; citations?: Citation[]; error?: boolean;
  ms?: number; provider?: string | null; model?: string | null; modelName?: string | null; streaming?: boolean; stopped?: boolean;
};

function storage<T>(key: string, fallback: T): T {
  try { const v = localStorage.getItem(key); return v ? (JSON.parse(v) as T) : fallback; } catch { return fallback; }
}
function save(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
function workspaceId(): string {
  let id = storage<string | null>("workspace_id_v2", null);
  if (!id) {
    try { id = localStorage.getItem("workspace_id"); } catch { id = null; }  // keep workspaces created by v1
    id = id || crypto.randomUUID();
    save("workspace_id_v2", id);
  }
  return id;
}
const WS = workspaceId();
const headers = { "X-Workspace-Id": WS };

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, { ...init, headers: { ...headers, ...(init?.headers || {}) } });
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error(data?.detail || `The server responded with an error (${r.status}).`);
  return data as T;
}

function groupLabel(iso: string): string {
  const d = new Date(iso), now = new Date();
  const days = Math.floor((new Date(now.toDateString()).getTime() - new Date(d.toDateString()).getTime()) / 86400000);
  return days <= 0 ? "Today" : days === 1 ? "Yesterday" : days < 7 ? "Previous 7 days" : "Older";
}

function plainText(m: Msg): string {
  const order = numberCitations(m.content);
  return m.content.replace(/\[(S\d+)\]/g, (_, label: string) => {
    const c = m.citations?.find((x) => x.label === label);
    return c ? ` [${order.get(label)}: ${c.filename}, p. ${c.page_number}]` : "";
  });
}

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="fatal">
        <LogoMark size={48} />
        <h2>Something went wrong</h2>
        <p>The page hit an unexpected problem. Reloading usually fixes it; your chats and documents are saved.</p>
        <button className="btn primary" onClick={() => location.reload()}>Reload</button>
      </div>
    );
  }
}

function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [rejected, setRejected] = useState<Doc[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [choice, setChoice] = useState<Choice | null>(storage<Choice | null>("model_choice", null));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [source, setSource] = useState<Citation | null>(null);
  const [notice, setNotice] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [libOpen, setLibOpen] = useState(false);
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [copied, setCopied] = useState<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadDocs = useCallback(async () => {
    try { setDocs(await api<Doc[]>("/documents")); setNotice(""); }
    catch (e) { setNotice(`We couldn't reach the server. ${(e as Error).message}`); }
    finally { setLoaded(true); }
  }, []);
  const loadConversations = useCallback(async () => {
    try { setConversations(await api<Conversation[]>("/conversations")); } catch { /* shown via docs notice */ }
  }, []);
  const loadCatalog = useCallback(async () => {
    try {
      const c = await api<Catalog>("/models");
      setCatalog(c);
      setChoice((cur) => {
        const p = c.providers.find((x) => x.id === cur?.provider);
        return cur && p?.configured && p.models.some((m) => m.id === cur.model) ? cur : c.default;
      });
    } catch { /* picker shows loading */ }
  }, []);

  useEffect(() => { loadDocs(); loadConversations(); loadCatalog(); }, [loadDocs, loadConversations, loadCatalog]);
  useEffect(() => { if (choice) save("model_choice", choice); }, [choice]);
  useEffect(() => {
    if (!docs.some((d) => d.status === "processing")) return;
    const t = setTimeout(loadDocs, 1500);
    return () => clearTimeout(t);
  }, [docs, loadDocs]);
  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, which React would call as a cleanup.
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function modelName(id?: string | null) {
    return catalog?.providers.flatMap((p) => p.models).find((m) => m.id === id)?.name ?? id ?? null;
  }

  async function openConversation(id: string) {
    abortRef.current?.abort();
    setActiveId(id); setNavOpen(false); setMessages([]);
    try {
      const rows = await api<Msg[]>(`/conversations/${id}/messages`);
      setMessages(rows.map((m) => ({ ...m, citations: Array.isArray(m.citations) ? m.citations : [], modelName: modelName(m.model) })));
    } catch (e) { setNotice(`We couldn't open that chat. ${(e as Error).message}`); }
  }

  function newChat() {
    abortRef.current?.abort();
    setActiveId(null); setMessages([]); setNavOpen(false);
    setTimeout(() => inputRef.current?.focus(), 0);
  }

  async function renameConversation(id: string, title: string) {
    setRenaming(null);
    const t = title.trim();
    if (!t) return;
    setConversations((cs) => cs.map((c) => (c.id === id ? { ...c, title: t } : c)));
    await api(`/conversations/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: t }) }).catch(() => loadConversations());
  }

  async function deleteConversation(id: string) {
    setConfirmDelete(null);
    setConversations((cs) => cs.filter((c) => c.id !== id));
    if (id === activeId) newChat();
    await api(`/conversations/${id}`, { method: "DELETE" }).catch(() => loadConversations());
  }

  async function upload(files: FileList | File[] | null) {
    const list = Array.from(files || []);
    if (!list.length) return;
    setUploading(true); setLibOpen(true);
    const form = new FormData();
    list.forEach((f) => form.append("files", f));
    try {
      const data = await api<Doc[]>("/documents", { method: "POST", body: form });
      setRejected(data.filter((d) => d.status === "rejected"));
      const dupes = data.filter((d: Doc & { duplicate?: boolean }) => d.duplicate).map((d) => d.filename);
      if (dupes.length) setNotice(`Already in your library, so it was not processed again: ${dupes.join(", ")}.`);
      await loadDocs();
    } catch (e) { setNotice(`Upload failed. ${(e as Error).message}`); }
    setUploading(false);
  }

  async function removeDoc(id?: string) {
    if (!id) return;
    setConfirmDelete(null);
    setDocs((d) => d.filter((x) => x.id !== id));
    await api(`/documents/${id}`, { method: "DELETE" }).catch(() => {});
    loadDocs();
  }

  function stop() { abortRef.current?.abort(); }

  async function ask(q?: string) {
    const question = (q ?? input).trim();
    if (!question || busy) return;
    setInput(""); setBusy(true); setPhase("Searching your documents");
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "", streaming: true }]);
    const patch = (fn: (m: Msg) => Msg) => setMessages((all) => [...all.slice(0, -1), fn(all[all.length - 1])]);
    const controller = new AbortController();
    abortRef.current = controller;
    let convId = activeId;
    try {
      const r = await fetch(`${API}/chat`, {
        method: "POST", signal: controller.signal,
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({ question, conversation_id: activeId, provider: choice?.provider, model: choice?.model }),
      });
      if (!r.ok || !r.body) throw new Error(`The server responded with an error (${r.status}).`);
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split("\n\n");
        buf = events.pop() || "";
        for (const ev of events) {
          if (!ev.startsWith("data: ")) continue;
          const data = JSON.parse(ev.slice(6));
          if (data.type === "conversation") {
            convId = data.id;
            setActiveId(data.id);
            setConversations((cs) => [{ id: data.id, title: data.title, updated_at: new Date().toISOString() }, ...cs]);
          }
          if (data.type === "model") setPhase(`Writing with ${data.model_name}`);
          if (data.type === "token") patch((m) => ({ ...m, content: m.content + data.text }));
          if (data.type === "reset") { setPhase("Switching to the backup model"); patch((m) => ({ ...m, content: "" })); }
          if (data.type === "done") {
            patch(() => ({ role: "assistant", content: data.answer, citations: data.citations || [], ms: data.timing?.total_ms,
              provider: data.provider, model: data.model, modelName: data.model_name }));
          }
          if (data.type === "error") patch(() => ({ role: "assistant", content: data.message, error: true }));
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") patch((m) => ({ ...m, streaming: false, stopped: true }));
      else patch(() => ({ role: "assistant", content: `We couldn't get an answer. ${(err as Error).message} Please check your connection and try again.`, error: true }));
    }
    setBusy(false); setPhase("");
    if (convId) setConversations((cs) => {
      const c = cs.find((x) => x.id === convId);
      return c ? [{ ...c, updated_at: new Date().toISOString() }, ...cs.filter((x) => x.id !== convId)] : cs;
    });
    loadCatalog();  // refresh provider status badges
  }

  async function copy(i: number, m: Msg) {
    try { await navigator.clipboard.writeText(plainText(m)); setCopied(i); setTimeout(() => setCopied(null), 1500); } catch { /* clipboard blocked */ }
  }

  const ready = docs.filter((d) => d.status === "ready");
  const processing = docs.filter((d) => d.status === "processing");
  const searchable = ready.length > 0 || processing.some((d) => (d.progress || 0) > 10);
  const totalPages = ready.reduce((n, d) => n + (d.page_count || 0), 0);
  const active = conversations.find((c) => c.id === activeId);
  const groups = conversations.reduce<Record<string, Conversation[]>>((g, c) => ((g[groupLabel(c.updated_at)] ||= []).push(c), g), {});
  const suggestions = ready.length
    ? [`Summarize ${ready[0].filename.replace(/\.pdf$/i, "")}`, "What dosages are recommended?", "List the reported side effects", "What contraindications are mentioned?"]
    : [];

  return (
    <div className={`app ${navOpen ? "nav-open" : ""} ${libOpen ? "lib-open" : ""}`}
      onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setDragging(true); } }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files); }}>

      <aside className="nav" aria-label="Chat history">
        <div className="brand">
          <LogoMark size={36} />
          <div><b>MedDocs<span className="brand-ai"> AI</span></b><small>Evidence-grounded answers</small></div>
        </div>
        <button className="btn new-chat" onClick={newChat}>{Icon.plus} New chat</button>
        <div className="conv-scroll">
          {!conversations.length && <p className="nav-empty">Your conversations will appear here.</p>}
          {Object.entries(groups).map(([label, items]) => (
            <div key={label} className="conv-group">
              <div className="conv-group-label">{label}</div>
              {items.map((c) => (
                <div key={c.id} className={`conv ${c.id === activeId ? "is-active" : ""}`}>
                  {renaming?.id === c.id ? (
                    <input className="conv-rename" autoFocus value={renaming.title} maxLength={120}
                      onChange={(e) => setRenaming({ id: c.id, title: e.target.value })}
                      onBlur={() => renameConversation(c.id, renaming.title)}
                      onKeyDown={(e) => { if (e.key === "Enter") renameConversation(c.id, renaming.title); if (e.key === "Escape") setRenaming(null); }} />
                  ) : confirmDelete === c.id ? (
                    <div className="conv-confirm">
                      <span>Delete this chat?</span>
                      <button className="mini danger" onClick={() => deleteConversation(c.id)}>Delete</button>
                      <button className="mini" onClick={() => setConfirmDelete(null)}>Cancel</button>
                    </div>
                  ) : (
                    <>
                      <button className="conv-title" title={c.title} onClick={() => openConversation(c.id)}>{Icon.chat}<span>{c.title}</span></button>
                      <span className="conv-actions">
                        <button className="icon-btn" aria-label="Rename chat" onClick={() => setRenaming({ id: c.id, title: c.title })}>{Icon.pencil}</button>
                        <button className="icon-btn" aria-label="Delete chat" onClick={() => setConfirmDelete(c.id)}>{Icon.trash}</button>
                      </span>
                    </>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="disclaimer">{Icon.shield}<p>Answers come only from your documents. Not a substitute for professional medical advice, diagnosis or treatment.</p></div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn only-mobile" aria-label="Open chat history" onClick={() => setNavOpen(true)}>{Icon.menu}</button>
          <div className="topbar-title">
            <h1>{active?.title || "New chat"}</h1>
            <p>{ready.length ? `${ready.length} document${ready.length > 1 ? "s" : ""} · ${totalPages.toLocaleString()} pages` : "No documents yet"}{processing.length ? ` · ${processing.length} indexing` : ""}</p>
          </div>
          <ModelPicker catalog={catalog} choice={choice} onChange={setChoice} />
          <button className="icon-btn lib-toggle" aria-label="Open document library" onClick={() => setLibOpen(true)}>
            {Icon.library}{docs.length > 0 && <span className="count-badge">{docs.length}</span>}
          </button>
        </header>

        {notice && <div className="alert" role="status"><span>{notice}</span><button className="icon-btn" aria-label="Dismiss" onClick={() => setNotice("")}>{Icon.close}</button></div>}

        <section className="thread" aria-live="polite">
          {messages.length === 0 && (
            <div className="welcome">
              <LogoMark size={60} />
              {!loaded ? null : !docs.length ? (
                <>
                  <h2>Upload your first medical document</h2>
                  <p>Clinical guidelines, drug handbooks, research papers or patient leaflets. Ask anything and get answers with the exact document and page.</p>
                  <button className="btn primary lg" onClick={() => fileRef.current?.click()}>{Icon.upload} Upload PDFs</button>
                  <ul className="features">
                    <li><b>Grounded</b><span>Answers use only your documents, never outside guesses.</span></li>
                    <li><b>Cited</b><span>Every fact links to its document and page.</span></li>
                    <li><b>Private</b><span>Your library and chats are kept in your own workspace.</span></li>
                  </ul>
                </>
              ) : !searchable ? (
                <>
                  <h2>Indexing your documents</h2>
                  <p>{processing.map((d) => `${d.filename} (${d.progress || 0}%)`).join(", ")}. You can start asking as soon as the first pages are indexed.</p>
                </>
              ) : (
                <>
                  <h2>What would you like to know?</h2>
                  <p>Ask about {ready.length ? <b>{ready.length === 1 ? ready[0].filename : `${ready.length} documents`}</b> : "your documents"}. Every answer cites its sources.</p>
                  <div className="suggestions">{suggestions.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
                </>
              )}
            </div>
          )}

          {messages.map((m, i) => (
            <article key={i} className={`msg ${m.role} ${m.error ? "is-error" : ""}`}>
              {m.role === "assistant" && <span className="avatar"><LogoMark size={30} /></span>}
              <div className="bubble">
                {m.role === "user" ? <div className="user-text">{m.content}</div>
                  : m.content ? <Answer content={m.content} citations={m.citations} streaming={m.streaming} onCite={setSource} />
                  : m.stopped ? <p className="muted">Stopped before any answer was written.</p>
                  : <div className="thinking"><span /><span /><span /><em>{phase || "Thinking"}</em></div>}

                {!!m.citations?.length && (
                  <div className="sources">
                    <span className="sources-label">Sources</span>
                    {m.citations.map((c, n) => (
                      <button key={c.label} className="source-chip" onClick={() => setSource(c)} title={`${c.filename}, page ${c.page_number}`}>
                        <span className="num">{n + 1}</span><span className="name">{c.filename}</span><span className="page">p. {c.page_number}</span>
                      </button>
                    ))}
                  </div>
                )}
                {m.role === "assistant" && !m.streaming && !m.error && m.content && (
                  <div className="msg-meta">
                    {m.stopped && <span className="tag">Stopped</span>}
                    {m.modelName && <span>{m.modelName}</span>}
                    {!!m.ms && <span>{(m.ms / 1000).toFixed(1)} s</span>}
                    <button className="meta-btn" onClick={() => copy(i, m)}>{copied === i ? <>{Icon.check} Copied</> : <>{Icon.copy} Copy</>}</button>
                  </div>
                )}
              </div>
            </article>
          ))}
          <div ref={endRef} />
        </section>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(); }}>
          <div className="composer-box">
            <textarea ref={inputRef} value={input} rows={1} maxLength={2000}
              placeholder={searchable ? "Ask a question about your documents..." : docs.length ? "Indexing... you can type your question now" : "Upload a PDF to start asking questions"}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }} />
            {busy
              ? <button type="button" className="send is-stop" aria-label="Stop generating" onClick={stop}>{Icon.stop}</button>
              : <button className="send" disabled={!input.trim()} aria-label="Send">{Icon.send}</button>}
          </div>
          <small>Enter to send · Shift+Enter for a new line · Answers can be incomplete; always verify with the cited page.</small>
        </form>
      </main>

      <aside className="library" aria-label="Document library">
        <div className="lib-head">
          <div><h2>Library</h2><p>{docs.length ? `${ready.length} ready${processing.length ? `, ${processing.length} indexing` : ""}` : "No documents yet"}</p></div>
          <button className="icon-btn only-overlay" aria-label="Close library" onClick={() => setLibOpen(false)}>{Icon.close}</button>
        </div>
        <button className={`dropzone ${uploading ? "is-busy" : ""}`} onClick={() => fileRef.current?.click()} disabled={uploading}>
          <span className="dz-icon">{uploading ? <span className="spinner" /> : Icon.upload}</span>
          <span className="dz-title">{uploading ? "Uploading..." : "Add PDFs"}</span>
          <span className="dz-sub">Drop files anywhere or click. Up to 10 files, 25 MB each.</span>
        </button>
        <input ref={fileRef} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />

        <ul className="doc-list">
          {rejected.map((d, i) => (
            <li key={`r${i}`} className="doc is-failed">
              <span className="doc-icon">{Icon.file}</span>
              <div className="doc-body"><b title={d.filename}>{d.filename}</b><span className="status">{d.error}</span></div>
              <button className="icon-btn" aria-label="Dismiss" onClick={() => setRejected((r) => r.filter((_, j) => j !== i))}>{Icon.close}</button>
            </li>
          ))}
          {docs.map((d) => (
            <li key={d.id} className={`doc is-${d.status}`}>
              <span className="doc-icon">{Icon.file}</span>
              <div className="doc-body">
                <b title={d.filename}>{d.filename}</b>
                {d.status === "ready" && <span className="status"><i className="dot tone-ok" />{(d.page_count || 0).toLocaleString()} pages · {(d.chunk_count || 0).toLocaleString()} passages</span>}
                {d.status === "processing" && (
                  <>
                    <span className="status">{(d.progress || 0) > 10 ? "Indexing · already searchable" : "Reading pages"}{d.page_count ? ` · ${d.page_count.toLocaleString()} pages` : ""}</span>
                    <span className="progress" role="progressbar" aria-valuenow={d.progress || 0} aria-valuemin={0} aria-valuemax={100}>
                      <span style={{ width: `${Math.max(d.progress || 0, 3)}%` }} />
                    </span>
                  </>
                )}
                {d.status === "failed" && <span className="status">{d.error || "Processing failed"}</span>}
              </div>
              {d.status === "processing" ? <span className="pct">{d.progress || 0}%</span>
                : confirmDelete === d.id ? (
                  <span className="doc-confirm">
                    <button className="mini danger" onClick={() => removeDoc(d.id)}>Remove</button>
                    <button className="mini" onClick={() => setConfirmDelete(null)}>Keep</button>
                  </span>
                ) : <button className="icon-btn" aria-label={`Remove ${d.filename}`} onClick={() => setConfirmDelete(d.id || null)}>{Icon.trash}</button>}
            </li>
          ))}
          {loaded && !docs.length && !rejected.length && <li className="doc-empty">Documents you add appear here.</li>}
        </ul>
      </aside>

      <div className="scrim" onClick={() => { setNavOpen(false); setLibOpen(false); }} />
      {dragging && <div className="drag-overlay"><div>{Icon.upload}<b>Drop PDFs to add them to your library</b></div></div>}

      {source && (
        <div className="modal" onClick={() => setSource(null)} onKeyDown={(e) => e.key === "Escape" && setSource(null)}>
          <div className="sheet" role="dialog" aria-modal="true" aria-label="Source excerpt" onClick={(e) => e.stopPropagation()}>
            <div className="sheet-head">
              <div><span className="eyebrow">{Icon.quote} Source excerpt</span><h3>{source.filename}</h3><span className="page-tag">Page {source.page_number}</span></div>
              <button className="icon-btn" autoFocus aria-label="Close" onClick={() => setSource(null)}>{Icon.close}</button>
            </div>
            <blockquote>{source.excerpt}{source.excerpt.length >= 600 ? "..." : ""}</blockquote>
            <p className="sheet-note">This is the passage the answer was based on. Check the full page in your PDF for complete context.</p>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><ErrorBoundary><App /></ErrorBoundary></StrictMode>);
