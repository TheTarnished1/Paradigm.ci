"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ApiError, Me, Setup, uploadDocuments } from "@/lib/api";
import { CREDIT, PrivacyNotice } from "./Privacy";

type Tab = "setup" | "status" | "account" | "privacy";
const TABS: [Tab, string][] = [
  ["setup", "Setup"],
  ["status", "Status"],
  ["account", "Account"],
  ["privacy", "Privacy"],
];

type Props = {
  me: Me;
  model: string;
  onChanged: (me: Me) => void;
  onSignedOut: () => void;
  onClose: () => void;
};

const message = (error: unknown) => (error instanceof Error ? error.message : "Something went wrong. Try again.");

function SetupTab({ me, onChanged }: Pick<Props, "me" | "onChanged">) {
  const [setup, setSetup] = useState<Setup>(me.setup);
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const field = (key: keyof Setup) => ({
    value: setup[key],
    onChange: (event: { target: { value: string } }) => setSetup({ ...setup, [key]: event.target.value }),
  });

  async function save(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSaved("");
    setBusy(true);
    try {
      const documents = files.length ? (await uploadDocuments(files)).documents : me.documents;
      await api("setup", "PUT", setup);
      onChanged({ ...me, setup, documents });
      setFiles([]);
      setSaved("Saved");
    } catch (failure) {
      setError(message(failure));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="form" onSubmit={save}>
      <label>
        Assistant name
        <input {...field("ci_name")} maxLength={40} required />
      </label>
      <label>
        Who it works for
        <input {...field("business_name")} maxLength={80} placeholder="Your name or your business" />
      </label>
      <label>
        Personality
        <textarea {...field("base_personality")} maxLength={300} rows={2} placeholder="Warm and patient, with plain language." />
      </label>
      <label>
        What it should do
        <textarea {...field("core_directive")} maxLength={1000} rows={3} placeholder="Answer customer questions using our documents." />
      </label>
      <label>
        Documents it can answer from
        <input
          type="file"
          accept="application/pdf"
          multiple
          onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
        />
        <span className="small muted">
          Up to 5 PDFs and 4 MB in total. New uploads replace your current documents.
          {me.documents.length > 0 && ` Current: ${me.documents.join(", ")}.`}
        </span>
      </label>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {saved && (
        <p className="success" role="status">
          {saved}
        </p>
      )}
      <button className="button primary wide" disabled={busy}>
        {busy ? "Saving..." : "Save"}
      </button>
    </form>
  );
}

function StatusTab({ me, model }: Pick<Props, "me" | "model">) {
  const [checks, setChecks] = useState<Record<string, boolean> | null>(null);
  const [failed, setFailed] = useState(false);
  const label = me.models.find((item) => item.id === model)?.label ?? model;

  useEffect(() => {
    api<Record<string, boolean>>("status")
      .then(setChecks)
      .catch(() => setFailed(true));
  }, []);

  const online = checks && Object.values(checks).every(Boolean);
  return (
    <div className="status">
      {!checks && !failed && <p role="status">Checking...</p>}
      {(checks || failed) && (
        <p className={online ? "pill online" : "pill offline"} role="status">
          {online ? "Online" : "Something isn't working"}
        </p>
      )}
      <dl>
        <dt>Plan</dt>
        <dd>
          {me.plan.label} ({me.plan.messages_per_day} messages a day, {me.plan.tool_calls} tool calls a message)
        </dd>
        <dt>Model</dt>
        <dd>{label}</dd>
        <dt>Assistant</dt>
        <dd>{me.setup.ci_name}</dd>
        <dt>Documents</dt>
        <dd>{me.documents.length ? me.documents.join(", ") : "None"}</dd>
        {checks &&
          Object.entries(checks).map(([name, working]) => (
            <div key={name} className="status-row">
              <dt>{name}</dt>
              <dd>{working ? "Online" : "Offline"}</dd>
            </div>
          ))}
      </dl>
    </div>
  );
}

function AccountTab({ me, onSignedOut }: Pick<Props, "me" | "onSignedOut">) {
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function logOut() {
    try {
      await api("logout", "POST");
      onSignedOut();
    } catch (failure) {
      setError(message(failure));
    }
  }

  async function newCode() {
    setError("");
    try {
      setCode((await api<{ recovery_code: string }>("recovery-code", "POST")).recovery_code);
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 401) onSignedOut();
      else setError(message(failure));
    }
  }

  async function deleteAccount(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      await api("delete-account", "POST", { password });
      onSignedOut();
    } catch (failure) {
      setError(message(failure));
    }
  }

  return (
    <div className="form">
      <p>
        Logged in as <strong>{me.username}</strong>
      </p>
      <button className="button wide" onClick={logOut}>
        Log out
      </button>
      <button className="button wide" onClick={newCode}>
        Create a new recovery code
      </button>
      {code && (
        <div role="status">
          <code className="code-box">{code}</code>
          <p className="small muted">Save this now, because it won&apos;t be shown again. It replaces any older code.</p>
        </div>
      )}
      <a className="button wide" href="/api/export" download="paradigm-data.json">
        Download my data
      </a>
      <details className="danger-zone">
        <summary>Delete my account</summary>
        <form className="form" onSubmit={deleteAccount}>
          <p className="small">This permanently deletes your account, chats, setup and documents.</p>
          <label>
            Your password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          <button className="button danger wide">Delete my account</button>
        </form>
      </details>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <p className="small muted">{CREDIT}</p>
    </div>
  );
}

export default function Settings({ me, model, onChanged, onSignedOut, onClose }: Props) {
  const [tab, setTab] = useState<Tab>("setup");
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className="glass dialog"
      aria-label="Settings"
      onClose={onClose}
      onClick={(event) => event.target === dialogRef.current && dialogRef.current?.close()}
    >
      <div className="dialog-inner">
        <header className="dialog-head">
          <h2 className="dialog-title">Settings</h2>
          <button className="icon-button" aria-label="Close settings" onClick={() => dialogRef.current?.close()}>
            ×
          </button>
        </header>
        <div className="tabs" role="tablist" aria-label="Settings">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              className={tab === id ? "tab active" : "tab"}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </div>
        {tab === "setup" && <SetupTab me={me} onChanged={onChanged} />}
        {tab === "status" && <StatusTab me={me} model={model} />}
        {tab === "account" && <AccountTab me={me} onSignedOut={onSignedOut} />}
        {tab === "privacy" && (
          <div className="prose small">
            <PrivacyNotice />
          </div>
        )}
      </div>
    </dialog>
  );
}
