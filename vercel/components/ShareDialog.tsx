"use client";

import { Link as LinkIcon, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { CopyButton } from "./Reply";

type Props = { chatId: string; title: string; onClose: () => void };

export default function ShareDialog({ chatId, title, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stopped, setStopped] = useState(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const create = () =>
    run(async () => {
      const { token } = await api<{ token: string }>(`conversations/${chatId}/share`, "POST");
      setUrl(`${window.location.origin}/share/${token}`);
      setStopped(false);
    });

  const stop = () =>
    run(async () => {
      await api(`conversations/${chatId}/share`, "DELETE");
      setUrl("");
      setStopped(true);
    });

  return (
    <dialog
      ref={dialogRef}
      className="glass dialog narrow"
      aria-label="Share chat"
      onClose={onClose}
      onClick={(event) => event.target === dialogRef.current && dialogRef.current?.close()}
    >
      <div className="dialog-inner form">
        <header className="dialog-head">
          <h2 className="dialog-title">Share chat</h2>
          <button className="icon-button" aria-label="Close" onClick={() => dialogRef.current?.close()}>
            <X size={18} />
          </button>
        </header>
        <p className="muted small">
          Anyone with the link can read <strong>{title}</strong> as it is now. Later messages aren&apos;t shared, and
          your username isn&apos;t shown. Creating a new link turns off the old one.
        </p>
        {url && (
          <div className="share-link">
            <input value={url} readOnly aria-label="Share link" onFocus={(event) => event.target.select()} />
            <CopyButton text={url} label="Copy link" />
          </div>
        )}
        {stopped && <p className="success" role="status">Sharing is off. Old links no longer work.</p>}
        {error && <p className="error" role="alert">{error}</p>}
        <button className="button primary wide" onClick={create} disabled={busy}>
          <LinkIcon size={16} aria-hidden="true" /> {url ? "Create a new link" : "Create link"}
        </button>
        <button className="button wide" onClick={stop} disabled={busy}>
          Stop sharing
        </button>
      </div>
    </dialog>
  );
}
