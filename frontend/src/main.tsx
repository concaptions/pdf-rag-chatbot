import { Component, StrictMode, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

type Doc = { id?: string; filename: string; status: string; error?: string | null; page_count?: number; chunk_count?: number };
type Citation = { label: string; filename: string; page_number: number; excerpt: string };
type Msg = { role: "user" | "assistant"; content: string; citations?: Citation[]; error?: boolean; ms?: number };

function workspaceId(): string {
  try {
    let id = localStorage.getItem("workspace_id");
    if (!id) { id = crypto.randomUUID(); localStorage.setItem("workspace_id", id); }
    return id;
  } catch { return crypto.randomUUID(); }
}
const WS = workspaceId();
const headers = { "X-Workspace-Id": WS };
const SUGGESTIONS = ["Summarize the key findings of these documents", "What dosages are recommended?", "List the reported side effects", "What contraindications are mentioned?"];

const Icon = {
  upload: <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V4m0 0l-4 4m4-4l4 4" /><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" /></svg>,
  file: <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z" /><path d="M14 3v5h5" /></svg>,
  send: <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>,
  close: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>,
  shield: <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"><path d="M12 3l8 3v6c0 4.5-3.4 8.4-8 9-4.6-.6-8-4.5-8-9V6z" /></svg>,
  logo: <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>,
};

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="fatal">
        <h2>Something went wrong</h2>
        <p>{this.state.error.message}</p>
        <button className="btn primary" onClick={() => location.reload()}>Reload</button>
      </div>
    );
  }
}

function Answer({ msg, onCite }: { msg: Msg; onCite: (c: Citation) => void }) {
  const parts = msg.content.split(/(\[S\d+\])/g);
  return (
    <div className="answer">
      {parts.map((p, i) => {
        const c = msg.citations?.find((x) => `[${x.label}]` === p);
        if (c) {
          const n = msg.citations!.indexOf(c) + 1;
          return <button key={i} className="ref" title={`${c.filename}, page ${c.page_number}`} onClick={() => onCite(c)}>{n}</button>;
        }
        if (/^\[S\d+\]$/.test(p)) return null;
        return <span key={i}>{p}</span>;
      })}
    </div>
  );
}

function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [rejected, setRejected] = useState<Doc[]>([]);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [source, setSource] = useState<Citation | null>(null);
  const [apiError, setApiError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadDocs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/documents`, { headers });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || `HTTP ${r.status}`);
      setDocs(Array.isArray(data) ? data : []);
      setApiError("");
    } catch (e) {
      setApiError(`Cannot reach the server. ${(e as Error).message}`);
    } finally { setLoaded(true); }
  }, []);

  useEffect(() => {
    loadDocs();
    fetch(`${API}/messages`, { headers })
      .then((r) => (r.ok ? r.json() : []))
      .then((d) => setMessages(Array.isArray(d) ? d.map((m: Msg) => ({ ...m, citations: Array.isArray(m.citations) ? m.citations : [] })) : []))
      .catch(() => {});
  }, [loadDocs]);

  useEffect(() => {
    if (!docs.some((d) => d.status === "processing")) return;
    const t = setTimeout(loadDocs, 1500);
    return () => clearTimeout(t);
  }, [docs, loadDocs]);

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, which React would call as a cleanup.
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function upload(files: FileList | File[] | null) {
    const list = Array.from(files || []);
    if (!list.length) return;
    setUploading(true);
    const form = new FormData();
    list.forEach((f) => form.append("files", f));
    try {
      const r = await fetch(`${API}/documents`, { method: "POST", headers, body: form });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || `HTTP ${r.status}`);
      setRejected(Array.isArray(data) ? data.filter((d: Doc) => d.status === "rejected") : []);
      await loadDocs();
    } catch (e) { setApiError(`Upload failed. ${(e as Error).message}`); }
    setUploading(false);
  }

  async function remove(id?: string) {
    if (!id) return;
    setDocs((d) => d.filter((x) => x.id !== id));
    await fetch(`${API}/documents/${id}`, { method: "DELETE", headers }).catch(() => {});
    loadDocs();
  }

  async function ask(q?: string) {
    const question = (q ?? input).trim();
    if (!question || busy) return;
    setInput(""); setBusy(true);
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "" }]);
    const patch = (fn: (m: Msg) => Msg) => setMessages((all) => [...all.slice(0, -1), fn(all[all.length - 1])]);
    try {
      const r = await fetch(`${API}/chat`, { method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
      if (!r.ok || !r.body) throw new Error(`Server error ${r.status}`);
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
          if (data.type === "token") patch((m) => ({ ...m, content: m.content + data.text }));
          if (data.type === "done") patch(() => ({ role: "assistant", content: data.answer, citations: data.citations || [], ms: data.timing?.total_ms }));
          if (data.type === "error") patch(() => ({ role: "assistant", content: data.message, error: true }));
        }
      }
    } catch (err) {
      patch(() => ({ role: "assistant", content: `Request failed. ${(err as Error).message}`, error: true }));
    }
    setBusy(false);
  }

  const ready = docs.filter((d) => d.status === "ready").length;
  const processing = docs.filter((d) => d.status === "processing").length;

  return (
    <div className="app"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files); }}>
      <aside className="sidebar">
        <div className="brand"><span className="logo">{Icon.logo}</span><div><b>MedDocs AI</b><small>Evidence-grounded answers</small></div></div>

        <button className={`dropzone ${uploading ? "is-busy" : ""}`} onClick={() => fileRef.current?.click()} disabled={uploading}>
          <span className="dz-icon">{uploading ? <span className="spinner" /> : Icon.upload}</span>
          <span className="dz-title">{uploading ? "Uploading..." : "Upload medical PDFs"}</span>
          <span className="dz-sub">Drag and drop or click. Up to 10 files, 25 MB each</span>
        </button>
        <input ref={fileRef} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />

        <div className="section-head">
          <span>Documents</span>
          <span className="count">{ready} ready{processing ? `, ${processing} processing` : ""}</span>
        </div>

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
                <span className="status">
                  {d.status === "ready" && <><i className="dot" />{d.page_count} pages, {d.chunk_count} passages</>}
                  {d.status === "processing" && <><span className="spinner sm" />Extracting and indexing</>}
                  {d.status === "failed" && <>{d.error || "Processing failed"}</>}
                </span>
              </div>
              <button className="icon-btn" aria-label={`Remove ${d.filename}`} onClick={() => remove(d.id)}>{Icon.close}</button>
            </li>
          ))}
          {loaded && !docs.length && !rejected.length && <li className="doc-empty">No documents yet</li>}
        </ul>

        <div className="disclaimer">{Icon.shield}<p>Answers are drawn only from your documents. Not a substitute for professional medical advice, diagnosis or treatment.</p></div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div><h1>Ask your documents</h1><p>Every answer cites the source document and page.</p></div>
          {messages.length > 0 && <span className="pill">{messages.filter((m) => m.role === "user").length} questions</span>}
        </header>

        {apiError && <div className="alert" role="alert">{apiError}<button className="icon-btn" onClick={() => setApiError("")}>{Icon.close}</button></div>}

        <section className="thread">
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">{Icon.logo}</div>
              <h2>{ready ? "What would you like to know?" : "Start by uploading a medical PDF"}</h2>
              <p>{ready ? "Ask a question and get an answer grounded in your documents, with page citations." : "Clinical guidelines, drug labels, research papers or patient leaflets."}</p>
              {ready > 0 && <div className="suggestions">{SUGGESTIONS.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>}
            </div>
          )}
          {messages.map((m, i) => (
            <article key={i} className={`msg ${m.role} ${m.error ? "is-error" : ""}`}>
              {m.role === "assistant" && <span className="avatar">{Icon.logo}</span>}
              <div className="bubble">
                {m.role === "assistant"
                  ? m.content ? <Answer msg={m} onCite={setSource} /> : <div className="thinking"><span /><span /><span /> Searching your documents</div>
                  : <div className="answer">{m.content}</div>}
                {!!m.citations?.length && (
                  <div className="sources">
                    <span className="sources-label">Sources</span>
                    {m.citations.map((c, n) => (
                      <button key={c.label} className="source-chip" onClick={() => setSource(c)}>
                        <span className="num">{n + 1}</span><span className="name">{c.filename}</span><span className="page">p. {c.page_number}</span>
                      </button>
                    ))}
                  </div>
                )}
                {m.ms && <span className="meta">{(m.ms / 1000).toFixed(1)} s</span>}
              </div>
            </article>
          ))}
          <div ref={endRef} />
        </section>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(); }}>
          <div className="composer-box">
            <textarea value={input} rows={1} maxLength={2000} disabled={busy}
              placeholder={ready ? "Ask a question about your documents..." : "Upload a PDF to start asking questions"}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }} />
            <button className="send" disabled={busy || !input.trim()} aria-label="Send">{busy ? <span className="spinner sm light" /> : Icon.send}</button>
          </div>
          <small>Enter to send, Shift+Enter for a new line</small>
        </form>
      </main>

      {dragging && <div className="drag-overlay"><div>{Icon.upload}<b>Drop PDFs to upload</b></div></div>}

      {source && (
        <div className="modal" onClick={() => setSource(null)}>
          <div className="sheet" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <div className="sheet-head">
              <div><span className="eyebrow">Source excerpt</span><h3>{source.filename}</h3><span className="page-tag">Page {source.page_number}</span></div>
              <button className="icon-btn" aria-label="Close" onClick={() => setSource(null)}>{Icon.close}</button>
            </div>
            <blockquote>{source.excerpt}</blockquote>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><ErrorBoundary><App /></ErrorBoundary></StrictMode>);
