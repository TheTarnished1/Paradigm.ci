"use client";

import { FormEvent, useState } from "react";
import { api } from "@/lib/api";
import { CREDIT, PrivacyNotice } from "./Privacy";

type Tab = "login" | "signup" | "recover";
const TABS: [Tab, string][] = [
  ["login", "Log in"],
  ["signup", "Create account"],
  ["recover", "Forgot password"],
];

export default function Auth({ onSignedIn }: { onSignedIn: (recoveryCode?: string) => void }) {
  const [tab, setTab] = useState<Tab>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);

  function switchTab(next: Tab) {
    setTab(next);
    setError("");
    setNotice("");
    setPassword("");
    setConfirm("");
    setCode("");
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setNotice("");
    if (tab === "signup" && password !== confirm) {
      setError("Those passwords don't match.");
      return;
    }
    setBusy(true);
    try {
      if (tab === "login") {
        await api("login", "POST", { username, password });
        onSignedIn();
      } else if (tab === "signup") {
        const result = await api<{ recovery_code: string }>("signup", "POST", { username, password });
        onSignedIn(result.recovery_code);
      } else {
        await api("recover", "POST", { username, code, password });
        switchTab("login");
        setNotice("Your password is changed. You can log in now.");
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const action = { login: "Log in", signup: "Create account", recover: "Reset password" }[tab];

  return (
    <main className="auth">
      <header>
        <h1 className="headline display">
          Paradigm<span className="dot">.</span>
        </h1>
        <p className="tagline">
          Here to serve. <em>Not replace.</em>
        </p>
      </header>

      <div className="tabs" role="tablist" aria-label="Account">
        {TABS.map(([id, label]) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={tab === id}
            className={tab === id ? "tab active" : "tab"}
            onClick={() => switchTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <form className="glass card form" onSubmit={submit}>
        <label>
          Username
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            maxLength={30}
            required
          />
        </label>
        {tab === "recover" && (
          <label>
            Recovery code
            <input value={code} onChange={(event) => setCode(event.target.value)} autoComplete="off" required />
          </label>
        )}
        <label>
          {tab === "recover" ? "New password" : "Password"}
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete={tab === "login" ? "current-password" : "new-password"}
            required
          />
        </label>
        {tab === "signup" && (
          <>
            <label>
              Repeat password
              <input
                type="password"
                value={confirm}
                onChange={(event) => setConfirm(event.target.value)}
                autoComplete="new-password"
                required
              />
            </label>
            <p className="small muted">
              Your messages and documents go to Groq, our AI provider, to create replies. You&apos;ll get a recovery
              code next, in case you ever forget your password.
            </p>
          </>
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
          {busy ? "One moment..." : action}
        </button>
      </form>

      <footer className="auth-footer">
        <button type="button" className="link" onClick={() => setShowPrivacy(!showPrivacy)} aria-expanded={showPrivacy}>
          Privacy
        </button>
        <span className="small muted">{CREDIT}</span>
      </footer>
      {showPrivacy && (
        <section className="glass card prose small">
          <PrivacyNotice />
        </section>
      )}
    </main>
  );
}
