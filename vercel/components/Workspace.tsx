"use client";

import {
  ArrowUp,
  Check,
  Copy,
  FileText,
  Globe,
  Lightbulb,
  MessageSquare,
  NotebookPen,
  PanelLeft,
  Paperclip,
  PenLine,
  Plus,
  Search,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";
import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, ApiError, ChatSummary, Me, Message, Mode, sendChat, Source, uploadDocuments } from "@/lib/api";
import { CREDIT } from "./Privacy";
import Settings from "./Settings";

type Suggestion = { icon: ReactNode; title: string; prompt: string };

const SUGGESTIONS: Record<Mode, Suggestion[]> = {
  assistant: [
    { icon: <Sparkles size={18} />, title: "What can you do?", prompt: "What can you help me with?" },
    { icon: <PenLine size={18} />, title: "Draft a message", prompt: "Help me write a clear, friendly message." },
    { icon: <NotebookPen size={18} />, title: "Plan my week", prompt: "Help me plan my week. Ask me what's on first." },
  ],
  general: [
    { icon: <Lightbulb size={18} />, title: "Explain simply", prompt: "Explain something simply. Ask me what first." },
    { icon: <Globe size={18} />, title: "What's new", prompt: "What's in the news today? Search the web." },
    { icon: <PenLine size={18} />, title: "Help me write", prompt: "Help me write a clear, friendly message." },
  ],
};

// Some models add markers like 【web_search】 where they used a tool. The server removes them when it saves.
const TOOL_MARKERS = /\s?【[^】]{0,40}】/g;
const DAY = 24 * 60 * 60 * 1000;

function greeting(name: string) {
  const hour = new Date().getHours();
  const part = hour < 5 ? "Up late" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  return `${part}, ${name}`;
}

function groupChats(chats: ChatSummary[]) {
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  const groups: [string, ChatSummary[]][] = [
    ["Today", []],
    ["Yesterday", []],
    ["Previous 7 days", []],
    ["Older", []],
  ];
  for (const chat of chats) {
    const when = chat.updated * 1000;
    const index = when >= startOfToday ? 0 : when >= startOfToday - DAY ? 1 : when >= startOfToday - 7 * DAY ? 2 : 3;
    groups[index][1].push(chat);
  }
  return groups.filter(([, items]) => items.length > 0);
}

function sourceParts(source: Source) {
  return typeof source === "string" ? { label: source, url: "" } : { label: source.label, url: source.url ?? "" };
}

function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }
  return (
    <button className="icon-button small" onClick={copy} aria-label={copied ? "Copied" : label} title={copied ? "Copied" : label}>
      {copied ? <Check size={16} /> : <Copy size={16} />}
    </button>
  );
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [text, setText] = useState("");
  useEffect(() => setText(ref.current?.innerText ?? ""), [children]);
  return (
    <div className="code-block">
      <div className="code-bar">
        <span>Code</span>
        <CopyButton text={text} label="Copy code" />
      </div>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  return (
    <div className="sources" aria-label="Sources">
      {sources.map(sourceParts).map(({ label, url }, index) => {
        const isWeb = Boolean(url);
        const content = (
          <>
            <span className="source-icon" aria-hidden="true">
              {isWeb ? <Globe size={14} /> : <FileText size={14} />}
            </span>
            <span className="source-label">{label}</span>
          </>
        );
        return isWeb ? (
          <a key={index} className="source" href={url} target="_blank" rel="noopener noreferrer">
            {content}
          </a>
        ) : (
          <span key={index} className="source">
            {content}
          </span>
        );
      })}
    </div>
  );
}

function Reply({ message }: { message: Message }) {
  const text = message.content.replace(TOOL_MARKERS, "");
  return (
    <article className={message.error ? "reply failed" : "reply"}>
      <div className="reply-mark" aria-hidden="true">
        <img src="/logo.svg" alt="" width={16} height={16} />
      </div>
      <div className="reply-body">
        <div className="prose">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ children, href }) => (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              ),
              pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
            }}
          >
            {text}
          </ReactMarkdown>
        </div>
        {message.sources.length > 0 && <Sources sources={message.sources} />}
        {!message.error && text && (
          <div className="reply-actions">
            <CopyButton text={text} label="Copy reply" />
          </div>
        )}
      </div>
    </article>
  );
}

type Props = { me: Me; onChanged: (me: Me) => void; onSignedOut: () => void };

export default function Workspace({ me, onChanged, onSignedOut }: Props) {
  const [mode, setMode] = useState<Mode | null>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState("");
  const [model, setModel] = useState(me.models[0]?.id ?? "");
  const [menuOpen, setMenuOpen] = useState(false);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
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
    (current: Mode) => api<ChatSummary[]>(`conversations?mode=${current}`).then(setChats).catch(fail),
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

  const groups = useMemo(() => {
    const term = filter.trim().toLowerCase();
    return groupChats(term ? chats.filter((chat) => chat.title.toLowerCase().includes(term)) : chats);
  }, [chats, filter]);

  function chooseMode(next: Mode) {
    if (sending) return;
    setMode(next);
    setChatId(null);
    setMessages([]);
    setNotice("");
    setFilter("");
    setMenuOpen(false);
  }

  function newChat() {
    if (sending) return;
    setChatId(null);
    setMessages([]);
    setNotice("");
    setMenuOpen(false);
    if (!mode) setMode("general");
    setTimeout(() => inputRef.current?.focus());
  }

  async function openChat(id: string) {
    if (sending) return;
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

  async function attach(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!files.length) return;
    setUploading(true);
    setNotice("");
    try {
      const { documents } = await uploadDocuments(files);
      onChanged({ ...me, documents });
      setNotice(`Added ${documents.join(", ")}. ${name} will answer from ${documents.length > 1 ? "them" : "it"}.`);
    } catch (error) {
      fail(error);
    } finally {
      setUploading(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
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
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await sendChat(
        { message, mode, conversation_id: chatId, model: model || null },
        (event) => {
          if (event.type === "conversation") setChatId(event.id);
          else if (event.type === "status") setStatus(event.text);
          else if (event.type === "text") {
            setStatus(null);
            updateReply((reply) => ({ ...reply, content: reply.content + event.text }));
          } else if (event.type === "done") updateReply((reply) => ({ ...reply, sources: event.sources }));
          else if (event.type === "error") updateReply((reply) => ({ ...reply, content: event.text, error: true }));
        },
        controller.signal,
      );
      if (controller.signal.aborted) {
        updateReply((reply) => (reply.content ? reply : { ...reply, content: "Stopped.", error: true }));
      }
      loadChats(mode);
    } catch (error) {
      // Nothing was saved, so put the message back to send again
      setMessages(before);
      setDraft(message);
      fail(error);
    } finally {
      abortRef.current = null;
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
  if (mode === "assistant" && me.documents.length) {
    suggestions[1] = { icon: <FileText size={18} />, title: "Sum up my documents", prompt: "Summarize my documents." };
  }
  const lastIndex = messages.length - 1;
  const workspaceClass = ["workspace", sidebarHidden ? "sidebar-hidden" : ""].join(" ");

  return (
    <div className={workspaceClass}>
      {menuOpen && <button className="scrim" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
      <aside className={menuOpen ? "sidebar glass open" : "sidebar glass"} aria-label="Chats">
        <div className="sidebar-top">
          <button className="brand" onClick={() => { if (!sending) { setMode(null); setMenuOpen(false); } }}>
            Paradigm<span className="dot">.</span>
          </button>
          <button className="icon-button hide-sidebar" aria-label="Hide sidebar" title="Hide sidebar" onClick={() => setSidebarHidden(true)}>
            <PanelLeft size={18} />
          </button>
          <button className="icon-button close-menu" aria-label="Close menu" onClick={() => setMenuOpen(false)}>
            <X size={18} />
          </button>
        </div>

        <button className="button primary wide new-chat" onClick={newChat} disabled={sending}>
          <Plus size={18} aria-hidden="true" /> New chat
        </button>

        {mode && (
          <label className="search">
            <Search size={16} aria-hidden="true" />
            <input
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Search chats"
              aria-label="Search chats"
            />
          </label>
        )}

        <nav className="recent" aria-label="Recent chats">
          {!mode && <p className="small muted empty-hint">Pick Assistant or General chat to see your chats.</p>}
          {mode && groups.length === 0 && (
            <p className="small muted empty-hint">{filter ? "No chats match that." : "Your chats will show up here."}</p>
          )}
          {groups.map(([label, items]) => (
            <section key={label}>
              <h2 className="eyebrow">{label}</h2>
              <ul>
                {items.map((chat) => (
                  <li key={chat.id} className={chat.id === chatId ? "chat-row active" : "chat-row"}>
                    <button className="chat-title" onClick={() => openChat(chat.id)} title={chat.title} aria-current={chat.id === chatId ? "page" : undefined}>
                      <MessageSquare size={14} aria-hidden="true" />
                      <span>{chat.title}</span>
                    </button>
                    <button className="chat-delete" onClick={() => removeChat(chat.id)} aria-label={`Delete chat: ${chat.title}`} title="Delete chat">
                      <Trash2 size={15} />
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </nav>

        <button className="account" onClick={() => { setSettingsOpen(true); setMenuOpen(false); }} aria-label="Open settings">
          <span className="avatar" aria-hidden="true">{me.username.slice(0, 1).toUpperCase()}</span>
          <span className="account-text">
            <span className="account-name">{me.username}</span>
            <span className="account-plan">{me.plan.label} plan</span>
          </span>
        </button>
        <p className="small muted credit">{CREDIT}</p>
      </aside>

      <main className="stage">
        <header className="topbar">
          <button className="icon-button menu-button" aria-label="Open menu" onClick={() => setMenuOpen(true)}>
            <PanelLeft size={18} />
          </button>
          {sidebarHidden && (
            <button className="icon-button show-sidebar" aria-label="Show sidebar" title="Show sidebar" onClick={() => setSidebarHidden(false)}>
              <PanelLeft size={18} />
            </button>
          )}
          <div className="segmented" role="tablist" aria-label="Mode">
            {(["assistant", "general"] as Mode[]).map((item) => (
              <button
                key={item}
                role="tab"
                aria-selected={mode === item}
                className={mode === item ? "segment active" : "segment"}
                onClick={() => chooseMode(item)}
              >
                {item === "assistant" ? name : "General chat"}
              </button>
            ))}
          </div>
          {mode && (
            <label className="model-picker">
              <span className="sr-only">Model</span>
              <select value={model} onChange={(event) => setModel(event.target.value)} disabled={sending}>
                {me.models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          )}
        </header>

        {!mode && (
          <section className="column home">
            <h1 className="headline">
              {greeting(me.username)}
              <span className="dot">.</span>
            </h1>
            <p className="lede">How do you want to work today?</p>
            <div className="mode-cards">
              <button className="glass card mode-card" onClick={() => chooseMode("assistant")}>
                <span className="mode-icon" aria-hidden="true"><Sparkles size={20} /></span>
                <span className="card-title">{name}</span>
                <span className="muted">Your assistant, with your personality, instructions and documents.</span>
                <span className="card-meta">
                  {me.documents.length ? `${me.documents.length} document${me.documents.length > 1 ? "s" : ""} ready` : "No documents yet"}
                </span>
              </button>
              <button className="glass card mode-card" onClick={() => chooseMode("general")}>
                <span className="mode-icon" aria-hidden="true"><MessageSquare size={20} /></span>
                <span className="card-title">General chat</span>
                <span className="muted">Ask anything. It can search the web, and doesn&apos;t use your setup.</span>
                <span className="card-meta">Web search on</span>
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
                      {mode === "assistant" ? `Hi, I'm ${name}` : greeting(me.username)}
                      <span className="dot">.</span>
                    </h1>
                    <p className="lede">
                      {mode === "assistant" && me.setup.business_name
                        ? `Here to help ${me.setup.business_name}. What do you need?`
                        : "What can I help you with?"}
                    </p>
                    <div className="suggestions">
                      {suggestions.map((item) => (
                        <button key={item.title} className="suggestion glass" onClick={() => send(item.prompt)}>
                          <span className="suggestion-icon" aria-hidden="true">{item.icon}</span>
                          <span className="suggestion-title">{item.title}</span>
                          <span className="suggestion-text">{item.prompt}</span>
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
                    <div key={index} className="thinking" role="status">
                      <span className="pulse" aria-hidden="true" />
                      {status}
                      <span className="dots" aria-hidden="true" />
                    </div>
                  ) : (
                    <Reply key={index} message={message} />
                  ),
                )}
                {notice && (
                  <p className="notice" role="status">
                    {notice}
                  </p>
                )}
                <div ref={endRef} />
              </div>
            </div>
            <form className="composer" onSubmit={submit}>
              <div className="glass composer-box">
                {mode === "assistant" && (
                  <>
                    <input
                      ref={fileRef}
                      type="file"
                      accept="application/pdf"
                      multiple
                      hidden
                      onChange={attach}
                    />
                    <button
                      type="button"
                      className="icon-button attach"
                      onClick={() => fileRef.current?.click()}
                      disabled={uploading || sending}
                      aria-label="Add PDFs for your assistant"
                      title="Add PDFs. They replace your current documents."
                    >
                      <Paperclip size={18} />
                    </button>
                  </>
                )}
                <textarea
                  ref={inputRef}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleKey}
                  placeholder={uploading ? "Reading your documents..." : `Message ${speaker}`}
                  aria-label={`Message ${speaker}`}
                  rows={1}
                  maxLength={4000}
                />
                {sending ? (
                  <button type="button" className="send stop" onClick={stop} aria-label="Stop replying" title="Stop">
                    <Square size={14} fill="currentColor" />
                  </button>
                ) : (
                  <button className="send" disabled={!draft.trim()} aria-label="Send message" title="Send">
                    <ArrowUp size={18} />
                  </button>
                )}
              </div>
              <p className="composer-hint small muted">
                Enter to send, Shift + Enter for a new line. Paradigm can make mistakes, so check important facts.
              </p>
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
