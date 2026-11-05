"use client";

import { useCallback, useEffect, useState } from "react";
import Auth from "@/components/Auth";
import Workspace from "@/components/Workspace";
import { api, ApiError, Me } from "@/lib/api";

type State = "loading" | "signed-out" | "failed" | "ready";

export default function Page() {
  const [state, setState] = useState<State>("loading");
  const [me, setMe] = useState<Me | null>(null);
  const [recoveryCode, setRecoveryCode] = useState<string | null>(null);

  const load = useCallback(async () => {
    setState("loading");
    try {
      setMe(await api<Me>("me"));
      setState("ready");
    } catch (error) {
      setMe(null);
      setState(error instanceof ApiError && error.status === 401 ? "signed-out" : "failed");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (state === "loading") {
    return (
      <main className="center-screen" aria-busy="true">
        <p className="wordmark">
          Paradigm<span className="dot">.</span>
        </p>
        <p className="muted">Starting up. This can take a few seconds if it&apos;s been asleep.</p>
      </main>
    );
  }

  if (state === "failed") {
    return (
      <main className="center-screen">
        <p className="wordmark">
          Paradigm<span className="dot">.</span>
        </p>
        <p role="alert">Paradigm couldn&apos;t start.</p>
        <button className="button primary" onClick={load}>
          Try again
        </button>
      </main>
    );
  }

  if (state === "signed-out" || !me) {
    return (
      <Auth
        onSignedIn={(code) => {
          setRecoveryCode(code ?? null);
          load();
        }}
      />
    );
  }

  if (recoveryCode) {
    return (
      <main className="center-screen narrow">
        <h1 className="headline">Save your recovery code</h1>
        <p>It&apos;s the only way back in if you forget your password, and it won&apos;t be shown again.</p>
        <code className="code-box">{recoveryCode}</code>
        <button className="button primary wide" onClick={() => setRecoveryCode(null)}>
          I&apos;ve saved it
        </button>
      </main>
    );
  }

  return <Workspace me={me} onChanged={setMe} onSignedOut={() => { setMe(null); setState("signed-out"); }} />;
}
