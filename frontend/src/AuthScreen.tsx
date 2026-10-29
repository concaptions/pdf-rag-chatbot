import { useState } from "react";
import { supabase } from "./supabase";
import { Icon, LogoMark } from "./ui";

function friendly(message: string): string {
  const m = message.toLowerCase();
  if (m.includes("invalid login credentials")) return "Incorrect email or password.";
  if (m.includes("email not confirmed")) return "Please confirm your email first. Check your inbox for the confirmation link.";
  if (m.includes("already registered") || m.includes("already been registered")) return "An account with this email already exists. Sign in instead.";
  if (m.includes("rate limit") || m.includes("too many")) return "Too many attempts. Please wait a minute and try again.";
  if (m.includes("password")) return message;
  if (m.includes("fetch") || m.includes("network")) return "We couldn't reach the sign-in service. Check your connection and try again.";
  return "Something went wrong. Please try again.";
}

export function AuthScreen() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const signup = mode === "signup";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(""); setInfo("");
    const addr = email.trim().toLowerCase();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(addr)) return setError("Enter a valid email address.");
    if (signup && password.length < 8) return setError("Use at least 8 characters for your password.");
    if (!password) return setError("Enter your password.");
    setBusy(true);
    try {
      if (signup) {
        const { data, error } = await supabase.auth.signUp({ email: addr, password, options: { emailRedirectTo: window.location.origin } });
        if (error) throw error;
        if (!data.session) {  // email confirmation is switched on for the project
          setInfo(`We sent a confirmation link to ${addr}. Open it, then sign in here.`);
          setMode("signin");
        }
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email: addr, password });
        if (error) throw error;
      }
    } catch (err) {
      setError(friendly((err as Error).message || ""));
    }
    setBusy(false);
  }

  function switchMode(next: "signin" | "signup") {
    setMode(next); setError(""); setInfo("");
  }

  return (
    <div className="auth">
      <div className="auth-card">
        <div className="auth-brand"><LogoMark size={44} /><div><b>MedDocs<span className="brand-ai"> AI</span></b><small>Evidence-grounded answers from your medical PDFs</small></div></div>
        <h1>{signup ? "Create your account" : "Welcome back"}</h1>
        <p className="auth-sub">{signup ? "Your documents and chats are private to your account and available on any device." : "Sign in to continue to your documents and chats."}</p>

        <div className="auth-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={!signup} className={!signup ? "is-active" : ""} onClick={() => switchMode("signin")}>Sign in</button>
          <button type="button" role="tab" aria-selected={signup} className={signup ? "is-active" : ""} onClick={() => switchMode("signup")}>Sign up</button>
        </div>

        <form onSubmit={submit} noValidate>
          <label className="field">
            <span>Email</span>
            <input type="email" autoComplete="email" autoFocus value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </label>
          <label className="field">
            <span>Password</span>
            <div className="password">
              <input type={show ? "text" : "password"} autoComplete={signup ? "new-password" : "current-password"} value={password}
                onChange={(e) => setPassword(e.target.value)} placeholder={signup ? "At least 8 characters" : "Your password"} />
              <button type="button" className="reveal" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}>{show ? "Hide" : "Show"}</button>
            </div>
          </label>
          {error && <div className="auth-msg is-error" role="alert">{error}</div>}
          {info && <div className="auth-msg is-info" role="status">{Icon.check}<span>{info}</span></div>}
          <button className="btn primary auth-submit" disabled={busy}>
            {busy ? <span className="spinner light" /> : signup ? "Create account" : "Sign in"}
          </button>
        </form>
        <p className="auth-switch">
          {signup ? "Already have an account? " : "New to MedDocs AI? "}
          <button type="button" onClick={() => switchMode(signup ? "signin" : "signup")}>{signup ? "Sign in" : "Create an account"}</button>
        </p>
      </div>
      <p className="auth-foot">{Icon.shield} Not a substitute for professional medical advice, diagnosis or treatment.</p>
    </div>
  );
}
