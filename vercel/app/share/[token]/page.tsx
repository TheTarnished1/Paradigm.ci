"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api, Message } from "@/lib/api";
import Reply from "@/components/Reply";
import { CREDIT } from "@/components/Privacy";

type Shared = { title: string; created: number; messages: Omit<Message, "error">[] };

export default function SharedChat() {
  const { token } = useParams<{ token: string }>();
  const [chat, setChat] = useState<Shared | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api<Shared>(`shared/${encodeURIComponent(token)}`)
      .then(setChat)
      .catch((failure) => setError(failure instanceof Error ? failure.message : "This link doesn't work."));
  }, [token]);

  return (
    <main className="shared">
      <header className="shared-top column">
        <a className="brand" href="/">
          Paradigm<span className="dot">.</span>
        </a>
        <a className="button" href="/">
          Start your own chat
        </a>
      </header>
      <div className="column shared-body">
        {!chat && !error && <p role="status" className="muted">Loading...</p>}
        {error && (
          <section className="welcome">
            <h1 className="headline">Link not found<span className="dot">.</span></h1>
            <p className="lede">{error}</p>
          </section>
        )}
        {chat && (
          <>
            <h1 className="headline shared-title">{chat.title}</h1>
            <p className="eyebrow">Shared {new Date(chat.created * 1000).toLocaleDateString()}</p>
            <div className="shared-messages">
              {chat.messages.map((message, index) =>
                message.role === "user" ? (
                  <div key={index} className="bubble">{message.content}</div>
                ) : (
                  <Reply key={index} message={{ ...message, error: false }} />
                ),
              )}
            </div>
            <p className="small muted shared-foot">
              This is a copy of a chat with Paradigm. AI can make mistakes, so check important facts. {CREDIT}.
            </p>
          </>
        )}
      </div>
    </main>
  );
}
