import { StrictMode, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

type Doc = { id?: string; filename: string; status: string; error?: string | null; page_count?: number; chunk_count?: number; duplicate?: boolean };
type Citation = { label: string; filename: string; page_number: number; excerpt: string };
type Msg = { role: "user" | "assistant"; content: string; citations?: Citation[]; error?: boolean };

function workspaceId(): string {
  try {
    let id = localStorage.getItem("workspace_id");
    if (!id) { id = crypto.randomUUID(); localStorage.setItem("workspace_id", id); }
    return id;
  } catch { return crypto.randomUUID(); }
}
const WS = workspaceId();
const headers = { "X-Workspace-Id": WS };

function Answer({ msg, onCite }: { msg: Msg; onCite: (c: Citation) => void }) {
  const parts = msg.content.split(/(\[S\d+\])/g);
  return (
    <p className="answer">
      {parts.map((p, i) => {
        const c = msg.citations?.find((x) => `[${x.label}]` === p);
        if (c) return <button key={i} className="cite" title={`${c.filename}, page ${c.page_number}`} onClick={() => onCite(c)}>{c.filename} p.{c.page_number}</button>;
        if (/^\[S\d+\]$/.test(p)) return null;
        return <span key={i}>{p}</span>;
      })}
    </p>
  );
}

function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [rejected, setRejected] = useState<Doc[]>([]);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [source, setSource] = useState<Citation | null>(null);
  const [apiError, setApiError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  const loadDocs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/documents`, { headers });
      if (!r.ok) throw new Error((await r.json()).detail);
      setDocs(await r.json()); setApiError("");
    } catch (e) { setApiError(`Cannot reach the server: ${(e as Error).message}`); }
  }, []);

  useEffect(() => {
    loadDocs();
    fetch(`${API}/messages`, { headers }).then((r) => (r.ok ? r.json() : [])).then(setMessages).catch(() => {});
  }, [loadDocs]);

  useEffect(() => {
    if (!docs.some((d) => d.status === "processing")) return;
    const t = setTimeout(loadDocs, 1500);
    return () => clearTimeout(t);
  }, [docs, loadDocs]);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    try {
      const r = await fetch(`${API}/documents`, { method: "POST", headers, body: form });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail);
      setRejected(data.filter((d: Doc) => d.status === "rejected"));
      await loadDocs();
    } catch (e) { setApiError(`Upload failed: ${(e as Error).message}`); }
    setUploading(false);
  }

  async function remove(id?: string) {
    if (!id) return;
    await fetch(`${API}/documents/${id}`, { method: "DELETE", headers }).catch(() => {});
    loadDocs();
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
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
          if (data.type === "done") patch(() => ({ role: "assistant", content: data.answer, citations: data.citations }));
          if (data.type === "error") patch(() => ({ role: "assistant", content: data.message, error: true }));
        }
      }
    } catch (err) {
      patch(() => ({ role: "assistant", content: `Request failed: ${(err as Error).message}`, error: true }));
    }
    setBusy(false);
  }

  const ready = docs.filter((d) => d.status === "ready").length;
  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Medical PDF Assistant</h1>
        <label className={`drop ${uploading ? "busy" : ""}`}>
          <input type="file" accept="application/pdf,.pdf" multiple onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />
          {uploading ? "Uploading..." : "Upload PDFs"}
          <small>Multiple files, up to 25 MB each</small>
        </label>
        {rejected.map((d, i) => <div key={i} className="doc failed"><b>{d.filename}</b><span>{d.error}</span></div>)}
        <ul className="docs">
          {docs.map((d) => (
            <li key={d.id} className={`doc ${d.status}`}>
              <b>{d.filename}</b>
              <span>{d.status === "ready" ? `Ready, ${d.page_count} pages` : d.status === "processing" ? "Processing..." : `Failed: ${d.error}`}</span>
              <button aria-label={`Remove ${d.filename}`} onClick={() => remove(d.id)}>x</button>
            </li>
          ))}
        </ul>
        <p className="disclaimer">Answers come only from your uploaded documents. This tool is not a substitute for professional medical advice, diagnosis or treatment.</p>
      </aside>
      <main className="chat">
        {apiError && <div className="banner">{apiError}</div>}
        <div className="messages">
          {messages.length === 0 && <div className="empty">{ready ? "Ask a question about your documents." : "Upload one or more medical PDFs to begin."}</div>}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role} ${m.error ? "error" : ""}`}>
              {m.role === "assistant" ? (m.content ? <Answer msg={m} onCite={setSource} /> : <p className="typing">Searching documents...</p>) : <p>{m.content}</p>}
              {!!m.citations?.length && (
                <div className="sources">Sources: {m.citations.map((c) => <button key={c.label} className="cite" onClick={() => setSource(c)}>{c.filename} p.{c.page_number}</button>)}</div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form className="composer" onSubmit={ask}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder={ready ? "Ask about your documents..." : "Upload a PDF first"} disabled={busy} maxLength={2000} />
          <button disabled={busy || !input.trim()}>{busy ? "..." : "Ask"}</button>
        </form>
      </main>
      {source && (
        <div className="modal" onClick={() => setSource(null)}>
          <div className="card" onClick={(e) => e.stopPropagation()}>
            <h2>{source.filename}, page {source.page_number}</h2>
            <blockquote>{source.excerpt}</blockquote>
            <button onClick={() => setSource(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
