"use client";

import { FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, ApiError, Me, Message, Mode, sendChat } from "@/lib/api";
import { CREDIT } from "./Privacy";
import Settings from "./Settings";

const SUGGESTIONS: Record<Mode, string[]> = {
  assistant: ["What can you help me with?", "Help me write a message", "Help me plan my week"],
  general: ["Explain something simply", "Help me write a message", "Give me an idea to try"],
};

// Some models add markers like 【web_search】 where they used a tool. The server removes them when it saves.
const TOOL_MARKERS = /\s?【[^】]{0,40}】/g;

type Props = { me: Me; onChanged: (me: Me) => void; onSignedOut: () => void };

function Reply({ message }: { message: Message }) {
  return (
    <article className={message.error ? "reply failed" : "reply"}>
      <div className="prose">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ children, href }) => (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ),
          }}
        >
          {message.content.replace(TOOL_MARKERS, "")}
        </ReactMarkdown>
      </div>
      {message.sources.length > 0 && <p className="sources">Sources: {message.sources.join(" · ")}</p>}
    </article>
  );
}

export default function Workspace({ me, onChanged, onSignedOut }: Props) {
  const [mode, setMode] = useState<Mode | null>(null);
  const [chats, setChats] = useState<{ id: string; title: string }[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState("");
  const [model, setModel] = useState(me.models[0]?.id ?? "");
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const name = me.setup.ci_name || "Paradigm";
  const speaker = mode === "assistant" ? name : "Paradigm";

  const fail = useCallback(
    (error: unknown) => {
      if (error instanceof ApiError && error.status === 401) onSignedOut();
      else setNotice(error instanceof Error ? error.message : "Something went wrong. Try again.");
    },
    [onSignedOut],
  );

  const loadChats = useCallback(
    (current: Mode) => api<{ id: string; title: string }[]>(`conversations?mode=${current}`).then(setChats).catch(fail),
    [fail],
  );

  useEffect(() => {
    if (mode) loadChats(mode);
  }, [mode, loadChats]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, status]);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (event: globalThis.KeyboardEvent) => event.key === "Escape" && setMenuOpen(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [menuOpen]);

  function chooseMode(next: Mode) {
    setMode(next);
    setChatId(null);
    setMessages([]);
    setNotice("");
    setMenuOpen(false);
  }

  function newChat() {
    setChatId(null);
    setMessages([]);
    setNotice("");
    setMenuOpen(false);
    inputRef.current?.focus();
  }

  async function openChat(id: string) {
    setMenuOpen(false);
    setNotice("");
    setChatId(id);
    try {
      setMessages(await api<Message[]>(`conversations/${id}`));
    } catch (error) {
      fail(error);
    }
  }

  async function removeChat(id: string) {
    try {
      await api(`conversations/${id}`, "DELETE");
      setChats((list) => list.filter((chat) => chat.id !== id));
      if (id === chatId) newChat();
    } catch (error) {
      fail(error);
    }
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || sending || !mode) return;
    setDraft("");
    setNotice("");
    setSending(true);
    setStatus("Thinking");
    const before = messages;
    setMessages([
      ...before,
      { role: "user", content: message, sources: [], error: false },
      { role: "assistant", content: "", sources: [], error: false },
    ]);
    const updateReply = (change: (reply: Message) => Message) =>
      setMessages((list) => [...list.slice(0, -1), change(list[list.length - 1])]);

    try {
      await sendChat({ message, mode, conversation_id: chatId, model: model || null }, (event) => {
        if (event.type === "conversation") setChatId(event.id);
        else if (event.type === "status") setStatus(event.text);
        else if (event.type === "text") {
          setStatus(null);
          updateReply((reply) => ({ ...reply, content: reply.content + event.text }));
        } else if (event.type === "done") updateReply((reply) => ({ ...reply, sources: event.sources }));
        else if (event.type === "error") updateReply((reply) => ({ ...reply, content: event.text, error: true }));
      });
      loadChats(mode);
    } catch (error) {
      // Nothing was saved, so put the message back to send again
      setMessages(before);
      setDraft(message);
      fail(error);
    } finally {
      setSending(false);
      setStatus(null);
      inputRef.current?.focus();
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    send(draft);
  }

  function handleKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send(draft);
    }
  }

  const suggestions = mode ? [...SUGGESTIONS[mode]] : [];
  if (mode === "assistant" && me.documents.length) suggestions[1] = "Summarize my documents";
  const lastIndex = messages.length - 1;

  return (
    <div className="workspace">
      {menuOpen && <button className="scrim" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
      <aside className={menuOpen ? "sidebar glass open" : "sidebar glass"} aria-label="Navigation">
        <button className="brand" onClick={() => { setMode(null); setMenuOpen(false); }}>
          Paradigm<span className="dot">.</span>
        </button>
        <nav className="modes">
          {(["assistant", "general"] as Mode[]).map((item) => (
            <button
              key={item}
              className={mode === item ? "nav-item active" : "nav-item"}
              aria-current={mode === item ? "page" : undefined}
              onClick={() => chooseMode(item)}
            >
              {item === "assistant" ? "Assistant" : "General chat"}
            </button>
          ))}
        </nav>
        {mode && (
          <>
            <button className="button wide" onClick={newChat}>
              New chat
            </button>
            <label className="field-label">
              Model
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {me.models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            {me.more_models && <p className="small muted">More models come with the Paid plan.</p>}
            <div className="recent">
              {chats.length > 0 && <h2 className="eyebrow">Recent</h2>}
              <ul>
                {chats.map((chat) => (
                  <li key={chat.id} className={chat.id === chatId ? "chat-row active" : "chat-row"}>
                    <button className="chat-title" onClick={() => openChat(chat.id)} title={chat.title}>
                      {chat.title}
                    </button>
                    <button
                      className="chat-delete"
                      onClick={() => removeChat(chat.id)}
                      aria-label={`Delete chat: ${chat.title}`}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
        <div className="sidebar-end">
          <button className="nav-item" onClick={() => { setSettingsOpen(true); setMenuOpen(false); }}>
            Settings
          </button>
          <p className="small muted">{CREDIT}</p>
        </div>
      </aside>

      <main className="stage">
        <button className="menu-button glass" aria-label="Open menu" onClick={() => setMenuOpen(true)}>
          <span aria-hidden="true">☰</span>
        </button>

        {!mode && (
          <section className="column home">
            <h1 className="headline">
              Welcome, {me.username}
              <span className="dot">.</span>
            </h1>
            <p className="lede">How do you want to work today?</p>
            <div className="mode-cards">
              <button className="glass card mode-card" onClick={() => chooseMode("assistant")}>
                <span className="eyebrow">Assistant</span>
                <span className="card-title">{name}</span>
                <span className="muted">Your assistant, with your personality, instructions and documents.</span>
              </button>
              <button className="glass card mode-card" onClick={() => chooseMode("general")}>
                <span className="eyebrow">Open</span>
                <span className="card-title">General chat</span>
                <span className="muted">Ask anything. It doesn&apos;t use your setup or your documents.</span>
              </button>
            </div>
          </section>
        )}

        {mode && (
          <>
            <div className="messages" role="log" aria-label="Conversation" aria-live="polite">
              <div className="column">
                {messages.length === 0 && (
                  <section className="welcome">
                    <h1 className="headline">
                      {mode === "assistant" ? `Hi, I'm ${name}` : "General chat"}
                      <span className="dot">.</span>
                    </h1>
                    <p className="lede">
                      {mode === "assistant" && me.setup.business_name
                        ? `Here to help ${me.setup.business_name}.`
                        : "What can I help you with?"}
                    </p>
                    <div className="suggestions">
                      {suggestions.map((item) => (
                        <button key={item} className="suggestion" onClick={() => send(item)}>
                          {item}
                        </button>
                      ))}
                    </div>
                  </section>
                )}
                {messages.map((message, index) =>
                  message.role === "user" ? (
                    <div key={index} className="bubble">
                      {message.content}
                    </div>
                  ) : index === lastIndex && !message.content && status ? (
                    <p key={index} className="thinking" role="status">
                      {status}
                      <span className="dots" aria-hidden="true" />
                    </p>
                  ) : (
                    <Reply key={index} message={message} />
                  ),
                )}
                {notice && (
                  <p className="notice" role="alert">
                    {notice}
                  </p>
                )}
                <div ref={endRef} />
              </div>
            </div>
            <form className="composer" onSubmit={submit}>
              <div className="glass composer-box">
                <textarea
                  ref={inputRef}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleKey}
                  placeholder={`Message ${speaker}`}
                  aria-label={`Message ${speaker}`}
                  rows={1}
                  maxLength={4000}
                />
                <button className="send" disabled={sending || !draft.trim()} aria-label="Send message">
                  <span aria-hidden="true">↑</span>
                </button>
              </div>
            </form>
          </>
        )}
      </main>

      {settingsOpen && (
        <Settings
          me={me}
          model={model}
          onChanged={onChanged}
          onSignedOut={onSignedOut}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}
