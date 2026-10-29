import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type Citation = { label: string; document_id?: string; filename: string; page_number: number; excerpt: string };
export type ModelInfo = { id: string; name: string; description: string };
export type ProviderInfo = { id: string; name: string; configured: boolean; status: string; reason?: string | null; models: ModelInfo[] };
export type Catalog = { default: { provider: string; model: string }; providers: ProviderInfo[] };
export type Choice = { provider: string; model: string };

const svg = (d: string, size = 18, sw = 1.8) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d.split("|").map((p, i) => <path key={i} d={p} />)}
  </svg>
);

export const Icon = {
  upload: svg("M12 16V4|M8 8l4-4 4 4|M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2", 22),
  file: svg("M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z|M14 3v5h5"),
  send: svg("M5 12h14|M13 6l6 6-6 6", 18, 2),
  stop: <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2.5" fill="currentColor" /></svg>,
  close: svg("M6 6l12 12|M18 6L6 18", 16, 2),
  plus: svg("M12 5v14|M5 12h14", 18, 2),
  chat: svg("M21 12a8 8 0 01-11.6 7.1L4 20l.9-4.4A8 8 0 1121 12z", 16),
  pencil: svg("M4 20h4L19 9l-4-4L4 16z|M14 6l4 4", 15),
  trash: svg("M4 7h16|M10 11v6|M14 11v6|M6 7l1 13h10l1-13|M9 7V4h6v3", 15),
  check: svg("M5 12l5 5 9-10", 15, 2.2),
  copy: svg("M9 9h11v11H9z|M5 15H4V4h11v1", 15),
  menu: svg("M4 7h16|M4 12h16|M4 17h16", 20),
  library: svg("M5 4h4v16H5z|M11 4h4v16h-4z|M17 5l3 .8-3.7 14-3-.8", 20),
  chevron: svg("M6 9l6 6 6-6", 16, 2),
  shield: svg("M12 3l8 3v6c0 4.5-3.4 8.4-8 9-4.6-.6-8-4.5-8-9V6z", 16),
  spark: svg("M12 3v4|M12 17v4|M3 12h4|M17 12h4|M6 6l2.5 2.5|M15.5 15.5L18 18|M6 18l2.5-2.5|M15.5 8.5L18 6", 16),
  quote: svg("M7 17h3l2-4V7H6v6h3|M14 17h3l2-4V7h-6v6h3", 16),
};

export function LogoMark({ size = 36 }: { size?: number }) {
  return (
    <svg className="logo-mark" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <linearGradient id="logo-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#14a89c" />
          <stop offset="1" stopColor="#0a5f73" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill="url(#logo-g)" />
      <path d="M21 13h15l10 10v26a3 3 0 0 1-3 3H21a3 3 0 0 1-3-3V16a3 3 0 0 1 3-3z" fill="#fff" />
      <path d="M36 13v8a2 2 0 0 0 2 2h8z" fill="#bfe9e4" />
      <path d="M29 29h6v6h6v6h-6v6h-6v-6h-6v-6h6z" fill="#0d7c74" />
    </svg>
  );
}

/** Number citation labels by first appearance, which matches the server's citation order. */
export function numberCitations(text: string): Map<string, number> {
  const order = new Map<string, number>();
  for (const m of text.matchAll(/\[(S\d+)\]/g)) if (!order.has(m[1])) order.set(m[1], order.size + 1);
  return order;
}

export function Answer({ content, citations, streaming, onCite }: {
  content: string; citations?: Citation[]; streaming?: boolean; onCite: (c: Citation) => void;
}) {
  const order = numberCitations(content);
  const byLabel = new Map((citations || []).map((c) => [c.label, c]));
  const text = (streaming ? content.replace(/\[S?\d*$/, "") : content)
    .replace(/\[(S\d+)\]/g, (_, label: string) => (streaming || byLabel.has(label) ? `[${order.get(label)}](#cite-${label})` : ""));
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("#cite-")) {
              const c = byLabel.get(href.slice(6));
              return (
                <button type="button" className="ref" disabled={!c} title={c ? `${c.filename}, page ${c.page_number}` : undefined}
                  onClick={() => c && onCite(c)}>{children}</button>
              );
            }
            return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function statusLabel(p: ProviderInfo): { tone: string; text: string } {
  if (!p.configured) return { tone: "off", text: "Not set up" };
  if (p.status === "ok") return { tone: "ok", text: "Active" };
  if (p.status === "error") return { tone: "warn", text: p.reason ? cap(p.reason) : "Unavailable" };
  return { tone: "idle", text: "Ready" };
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function ModelPicker({ catalog, choice, onChange }: { catalog: Catalog | null; choice: Choice | null; onChange: (c: Choice) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", close); };
  }, [open]);

  if (!catalog || !choice) return <div className="picker-btn is-loading">Loading models...</div>;
  const provider = catalog.providers.find((p) => p.id === choice.provider);
  const model = provider?.models.find((m) => m.id === choice.model);
  const st = provider ? statusLabel(provider) : { tone: "idle", text: "" };

  return (
    <div className="picker" ref={ref}>
      <button type="button" className="picker-btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className={`dot tone-${st.tone}`} />
        <span className="picker-name">{model?.name ?? choice.model}</span>
        <span className="picker-provider">{provider?.id === "anthropic" ? "Anthropic" : "OpenAI"}</span>
        {Icon.chevron}
      </button>
      {open && (
        <div className="picker-menu" role="listbox">
          {catalog.providers.map((p) => {
            const s = statusLabel(p);
            return (
              <div key={p.id} className="picker-group">
                <div className="picker-group-head">
                  <span>{p.name}</span>
                  <span className={`badge tone-${s.tone}`}>{s.text}</span>
                </div>
                {p.models.map((m) => {
                  const selected = p.id === choice.provider && m.id === choice.model;
                  return (
                    <button key={m.id} type="button" role="option" aria-selected={selected} disabled={!p.configured}
                      className={`picker-item ${selected ? "is-selected" : ""}`}
                      onClick={() => { onChange({ provider: p.id, model: m.id }); setOpen(false); }}>
                      <span className="picker-item-text"><b>{m.name}</b><small>{m.description}</small></span>
                      {selected && <span className="picker-check">{Icon.check}</span>}
                    </button>
                  );
                })}
                {!p.configured && <p className="picker-note">Add the {p.id === "anthropic" ? "ANTHROPIC" : "OPENAI"}_API_KEY on the server to enable.</p>}
              </div>
            );
          })}
          <p className="picker-foot">{Icon.shield} If the selected model fails, the other provider answers automatically.</p>
        </div>
      )}
    </div>
  );
}
