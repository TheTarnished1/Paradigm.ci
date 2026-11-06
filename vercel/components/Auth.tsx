"use client";

import { FormEvent, useState } from "react";
import { Eye, EyeOff, FileText } from "lucide-react";
import { api } from "@/lib/api";
import { CREDIT, PrivacyNotice } from "./Privacy";

type Mode = "login" | "signup" | "recover";

const COPY: Record<Mode, { title: string; intro: string; action: string }> = {
  login: { title: "Welcome back", intro: "Log in to your chats and your assistant.", action: "Log in" },
  signup: { title: "Create your account", intro: "Just a username and password. No email needed.", action: "Create account" },
  recover: {
    title: "Reset your password",
    intro: "Use the recovery code you saved when you created your account.",
    action: "Reset password",
  },
};

export default function Auth({ onSignedIn }: { onSignedIn: (recoveryCode?: string) => void }) {
  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);

  function switchMode(next: Mode) {
    setMode(next);
    setError("");
    setNotice("");
    setPassword("");
    setShowPassword(false);
    setCode("");
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setNotice("");
    setBusy(true);
    try {
      if (mode === "login") {
        await api("login", "POST", { username, password });
        onSignedIn();
      } else if (mode === "signup") {
        const result = await api<{ recovery_code: string }>("signup", "POST", { username, password });
        onSignedIn(result.recovery_code);
      } else {
        await api("recover", "POST", { username, code, password });
        switchMode("login");
        setNotice("Your password is changed. You can log in now.");
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const copy = COPY[mode];
  const newPassword = mode !== "login";

  return (
    <main className="auth">
      <header className="auth-brand">
        <h1 className="headline display">
          Paradigm<span className="dot">.</span>
        </h1>
        <p className="tagline">
          Here to serve. <em>Not replace.</em>
        </p>
      </header>

      <section className="auth-about" aria-label="About Paradigm">
        <ul className="auth-points">
          <li>Your own assistant, set up with your instructions and documents.</li>
          <li>General chat that searches the web when you need current answers.</li>
          <li>It remembers what matters to you, and any chat can be shared with a link.</li>
        </ul>
        <figure className="auth-preview glass" aria-label="Example conversation">
          <p className="bubble">When is the Harper proposal due?</p>
          <div className="reply">
            <div className="reply-mark" aria-hidden="true">
              <img src="/logo.svg" alt="" width={16} height={16} />
            </div>
            <div className="reply-body">
              <p>Friday at noon. The brief also asks for a one-page budget, which isn&apos;t in your draft yet.</p>
              <div className="sources">
                <span className="source">
                  <span className="source-icon" aria-hidden="true">
                    <FileText size={14} />
                  </span>
                  <span className="source-label">Harper brief.pdf</span>
                </span>
              </div>
            </div>
          </div>
        </figure>
      </section>

      <div className="auth-panel">
        <form className="glass card form" onSubmit={submit} aria-labelledby="auth-title">
          {mode !== "recover" && (
            <div className="segmented auth-switch" role="group" aria-label="Account">
              {(["login", "signup"] as Mode[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  aria-pressed={mode === item}
                  className={mode === item ? "segment active" : "segment"}
                  onClick={() => switchMode(item)}
                >
                  {item === "login" ? "Log in" : "Create account"}
                </button>
              ))}
            </div>
          )}

          <div>
            <h2 id="auth-title" className="auth-title">
              {copy.title}
            </h2>
            <p className="muted small">{copy.intro}</p>
          </div>

          <div className="field">
            <label htmlFor="auth-username">Username</label>
            <input
              id="auth-username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              autoCapitalize="none"
              spellCheck={false}
              maxLength={30}
              aria-describedby={mode === "signup" ? "auth-username-hint" : undefined}
              required
            />
            {mode === "signup" && (
              <p id="auth-username-hint" className="hint">
                3–30 letters, numbers, dots, dashes or underscores.
              </p>
            )}
          </div>

          {mode === "recover" && (
            <div className="field">
              <label htmlFor="auth-code">Recovery code</label>
              <input
                id="auth-code"
                value={code}
                onChange={(event) => setCode(event.target.value)}
                autoComplete="off"
                autoCapitalize="none"
                spellCheck={false}
                aria-describedby="auth-code-hint"
                required
              />
              <p id="auth-code-hint" className="hint">
                Lost it? If you&apos;re still logged in on another device, make a new one in Settings, under Account.
              </p>
            </div>
          )}

          <div className="field">
            <label htmlFor="auth-password">{mode === "recover" ? "New password" : "Password"}</label>
            <div className="password">
              <input
                id="auth-password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={newPassword ? "new-password" : "current-password"}
                minLength={newPassword ? 8 : undefined}
                aria-describedby={newPassword ? "auth-password-hint" : undefined}
                required
              />
              <button
                type="button"
                className="icon-button small"
                aria-label="Show password"
                aria-pressed={showPassword}
                title={showPassword ? "Hide password" : "Show password"}
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={17} aria-hidden="true" /> : <Eye size={17} aria-hidden="true" />}
              </button>
            </div>
            {newPassword && (
              <p id="auth-password-hint" className="hint">
                At least 8 characters.
              </p>
            )}
            {mode === "login" && (
              <button type="button" className="link field-link" onClick={() => switchMode("recover")}>
                Forgot password?
              </button>
            )}
          </div>

          {mode === "signup" && (
            <p className="hint">
              Your messages and documents go to Groq, our AI provider, to write replies. Next, you&apos;ll get a
              recovery code to keep in case you forget your password.
            </p>
          )}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {notice && (
            <p className="success" role="status">
              {notice}
            </p>
          )}
          <button className="button primary wide" disabled={busy}>
            {busy ? "One moment..." : copy.action}
          </button>
          {mode === "recover" && (
            <button type="button" className="link field-link auth-back" onClick={() => switchMode("login")}>
              Back to log in
            </button>
          )}
        </form>

        <footer className="auth-footer">
          <button
            type="button"
            className="link"
            onClick={() => setShowPrivacy(!showPrivacy)}
            aria-expanded={showPrivacy}
          >
            Privacy
          </button>
          <span className="small muted">{CREDIT}</span>
        </footer>
        {showPrivacy && (
          <section className="glass card prose small" aria-label="Privacy">
            <PrivacyNotice />
          </section>
        )}
      </div>
    </main>
  );
}
