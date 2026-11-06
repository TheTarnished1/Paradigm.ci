"use client";

import {
  ArrowUp,
  Check,
  ChevronLeft,
  ChevronRight,
  FileText,
  Globe,
  Lightbulb,
  MessageCircleDashed,
  MessageSquare,
  Mic,
  MoreHorizontal,
  NotebookPen,
  PanelLeft,
  Paperclip,
  Pencil,
  PenLine,
  Pin,
  PinOff,
  Plus,
  RotateCcw,
  Search,
  Share2,
  Sparkles,
  Square,
  Telescope,
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
import { api, ApiError, ChatRequest, ChatSummary, Me, Message, Mode, sendChat, uploadDocuments } from "@/lib/api";
import { CREDIT } from "./Privacy";
import Reply, { CopyButton, SpeakButton } from "./Reply";
import Settings from "./Settings";
import ShareDialog from "./ShareDialog";

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

const DAY = 24 * 60 * 60 * 1000;
const HISTORY_MESSAGES = 10;

function greeting(name: string) {
  const hour = new Date().getHours();
  const part = hour < 5 ? "Up late" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  return `${part}, ${name}`;
}

function groupChats(chats: ChatSummary[]) {
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  const groups: [string, ChatSummary[]][] = [
    ["Pinned", []],
    ["Today", []],
    ["Yesterday", []],
    ["Previous 7 days", []],
    ["Older", []],
  ];
  for (const chat of chats) {
    const when = chat.updated * 1000;
    const age = when >= startOfToday ? 1 : when >= startOfToday - DAY ? 2 : when >= startOfToday - 7 * DAY ? 3 : 4;
    groups[chat.pinned ? 0 : age][1].push(chat);
  }
  return groups.filter(([, items]) => items.length > 0);
}

// The Web Speech API isn't in TypeScript's DOM types yet, so this describes the part used here
type Recognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
};

function speechRecognition(): (new () => Recognition) | null {
  const scope = window as unknown as Record<string, (new () => Recognition) | undefined>;
  return scope.SpeechRecognition ?? scope.webkitSpeechRecognition ?? null;
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
  const [temporary, setTemporary] = useState(false);
  const [research, setResearch] = useState(false);
  const [rowMenu, setRowMenu] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null);
  const [sharing, setSharing] = useState<ChatSummary | null>(null);
  const [editing, setEditing] = useState<{ index: number; text: string } | null>(null);
  const [canListen, setCanListen] = useState(false);
  const [listening, setListening] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<Recognition | null>(null);
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
  }, [messages, status, sending]);

  useEffect(() => setCanListen(Boolean(speechRecognition())), []);
  useEffect(() => () => recognitionRef.current?.stop(), []);

  useEffect(() => {
    if (!menuOpen && !rowMenu) return;
    const close = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMenuOpen(false);
      setRowMenu(null);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [menuOpen, rowMenu]);

  const groups = useMemo(() => {
    const term = filter.trim().toLowerCase();
    return groupChats(term ? chats.filter((chat) => chat.title.toLowerCase().includes(term)) : chats);
  }, [chats, filter]);

  function reset() {
    setChatId(null);
    setMessages([]);
    setNotice("");
    setEditing(null);
    setMenuOpen(false);
    setRowMenu(null);
  }

  function chooseMode(next: Mode) {
    if (sending) return;
    reset();
    setMode(next);
    setFilter("");
  }

  function newChat(asTemporary = false) {
    if (sending) return;
    reset();
    setTemporary(asTemporary);
    if (!mode) setMode("general");
    setTimeout(() => inputRef.current?.focus());
  }

  async function openChat(id: string) {
    if (sending) return;
    reset();
    setTemporary(false);
    setChatId(id);
    try {
      setMessages(await api<Message[]>(`conversations/${id}`));
    } catch (error) {
      fail(error);
    }
  }

  async function removeChat(id: string) {
    setRowMenu(null);
    try {
      await api(`conversations/${id}`, "DELETE");
      setChats((list) => list.filter((chat) => chat.id !== id));
      if (id === chatId) newChat();
    } catch (error) {
      fail(error);
    }
  }

  async function updateChat(id: string, change: { title?: string; pinned?: boolean }) {
    setRowMenu(null);
    setRenaming(null);
    try {
      await api(`conversations/${id}`, "PATCH", change);
      if (mode) await loadChats(mode);
    } catch (error) {
      fail(error);
    }
  }

  function submitRename(event: FormEvent) {
    event.preventDefault();
    if (!renaming) return;
    const title = renaming.title.trim();
    const old = chats.find((chat) => chat.id === renaming.id)?.title;
    if (title && title !== old) updateChat(renaming.id, { title });
    else setRenaming(null);
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

  function listen() {
    if (listening) return recognitionRef.current?.stop();
    const Speech = speechRecognition();
    if (!Speech) return;
    const recognition = new Speech();
    const start = draft.trim();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language;
    recognition.onresult = (event) => {
      const heard = Array.from(event.results, (result) => result[0].transcript).join("");
      setDraft(start ? `${start} ${heard}` : heard);
    };
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
    } catch {
      recognitionRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  /** Sends a message. With `from`, everything from that message on is replaced, for editing and regenerating. */
  async function send(text: string, from?: number, regenerate = false) {
    const message = text.trim();
    if (!message || sending || !mode) return;
    recognitionRef.current?.stop();
    const original = messages;
    const before = from === undefined ? messages : messages.slice(0, from);
    const deepResearch = research;
    if (from === undefined) setDraft("");
    setEditing(null);
    setNotice("");
    setResearch(false);
    setSending(true);
    setStatus(deepResearch ? "Researching" : "Thinking");
    setMessages([
      ...before,
      { role: "user", content: message, sources: [], error: false },
      { role: "assistant", content: "", sources: [], error: false },
    ]);
    const updateReply = (change: (reply: Message) => Message) =>
      setMessages((list) => [...list.slice(0, -1), change(list[list.length - 1])]);
    const controller = new AbortController();
    abortRef.current = controller;

    const body: ChatRequest = { message, mode, conversation_id: temporary ? null : chatId, model: model || null };
    if (deepResearch) body.research = true;
    if (temporary) {
      body.temporary = true;
      body.history = before
        .filter((past) => !past.error)
        .slice(-HISTORY_MESSAGES)
        .map((past) => ({ role: past.role, content: past.content.slice(0, 8000) }));
    } else if (regenerate) {
      body.regenerate = true;
    } else if (from !== undefined && chatId) {
      body.keep = from;
    }

    try {
      await sendChat(
        body,
        (event) => {
          if (event.type === "conversation") setChatId(event.id);
          else if (event.type === "status") setStatus(event.text);
          else if (event.type === "text") {
            setStatus(null);
            updateReply((reply) => ({ ...reply, content: reply.content + event.text }));
          } else if (event.type === "done") {
            updateReply((reply) => ({
              ...reply,
              sources: event.sources,
              followups: event.followups,
              id: event.id ?? undefined,
              versions: event.versions,
              version: event.version,
            }));
          } else if (event.type === "error") {
            updateReply((reply) => ({ ...reply, content: event.text, error: true }));
          }
        },
        controller.signal,
      );
      if (controller.signal.aborted) {
        // The server saves what was written so far, or notes that it stopped
        updateReply((reply) => (reply.content ? reply : { ...reply, content: "Stopped.", error: true }));
      }
      if (!temporary) loadChats(mode);
    } catch (error) {
      // Nothing was saved, so put things back to try again
      setMessages(original);
      if (from === undefined) setDraft(message);
      fail(error);
    } finally {
      abortRef.current = null;
      setSending(false);
      setStatus(null);
      inputRef.current?.focus();
    }
  }

  function regenerate(replyIndex: number) {
    const question = messages[replyIndex - 1];
    // Saved chats keep the old reply as a version. Temporary chats have nothing saved to keep.
    if (question?.role === "user") send(question.content, replyIndex - 1, !temporary);
  }

  async function showVersion(index: number, id: number, version: number) {
    try {
      const chosen = await api<Pick<Message, "content" | "sources" | "error" | "versions" | "version">>(
        `messages/${id}/version`,
        "POST",
        { index: version },
      );
      setMessages((list) => list.map((message, at) => (at === index ? { ...message, ...chosen, followups: [] } : message)));
    } catch (error) {
      fail(error);
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

  function handleEditKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Escape") setEditing(null);
    else if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && editing) {
      event.preventDefault();
      send(editing.text, editing.index);
    }
  }

  const suggestions = mode ? [...SUGGESTIONS[mode]] : [];
  if (mode === "assistant" && me.documents.length) {
    suggestions[1] = { icon: <FileText size={18} />, title: "Sum up my documents", prompt: "Summarize my documents." };
  }
  const lastIndex = messages.length - 1;
  const last = messages[lastIndex];
  const followups = !sending && last?.role === "assistant" ? (last.followups ?? []) : [];
  const currentChat = chats.find((chat) => chat.id === chatId);
  const workspaceClass = ["workspace", sidebarHidden ? "sidebar-hidden" : "", temporary ? "temporary" : ""].join(" ");

  return (
    <div className={workspaceClass}>
      {menuOpen && <button className="scrim" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
      <aside className={menuOpen ? "sidebar glass open" : "sidebar glass"} aria-label="Chats">
        <div className="sidebar-top">
          <button className="brand" onClick={() => { if (!sending) { setMode(null); setTemporary(false); setMenuOpen(false); } }}>
            Paradigm<span className="dot">.</span>
          </button>
          <button className="icon-button hide-sidebar" aria-label="Hide sidebar" title="Hide sidebar" onClick={() => setSidebarHidden(true)}>
            <PanelLeft size={18} />
          </button>
          <button className="icon-button close-menu" aria-label="Close menu" onClick={() => setMenuOpen(false)}>
            <X size={18} />
          </button>
        </div>

        <button className="button primary wide new-chat" onClick={() => newChat()} disabled={sending}>
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
                  <li key={chat.id}>
                    {renaming?.id === chat.id ? (
                      <form className="chat-row renaming" onSubmit={submitRename}>
                        <input
                          autoFocus
                          value={renaming.title}
                          maxLength={80}
                          aria-label="Chat name"
                          onChange={(event) => setRenaming({ id: chat.id, title: event.target.value })}
                          onKeyDown={(event) => event.key === "Escape" && setRenaming(null)}
                          onBlur={submitRename}
                        />
                        <button className="chat-more" aria-label="Save name" onMouseDown={(event) => event.preventDefault()}>
                          <Check size={15} />
                        </button>
                      </form>
                    ) : (
                      <div className={chat.id === chatId ? "chat-row active" : "chat-row"}>
                        <button className="chat-title" onClick={() => openChat(chat.id)} title={chat.title} aria-current={chat.id === chatId ? "page" : undefined}>
                          {chat.pinned ? <Pin size={14} aria-hidden="true" /> : <MessageSquare size={14} aria-hidden="true" />}
                          <span>{chat.title}</span>
                        </button>
                        <button
                          className="chat-more"
                          onClick={() => setRowMenu(rowMenu === chat.id ? null : chat.id)}
                          aria-label={`Options for ${chat.title}`}
                          aria-expanded={rowMenu === chat.id}
                          title="Options"
                        >
                          <MoreHorizontal size={16} />
                        </button>
                      </div>
                    )}
                    {rowMenu === chat.id && (
                      <div className="chat-menu" role="group" aria-label={`Options for ${chat.title}`}>
                        <button onClick={() => updateChat(chat.id, { pinned: !chat.pinned })}>
                          {chat.pinned ? <PinOff size={15} aria-hidden="true" /> : <Pin size={15} aria-hidden="true" />}
                          {chat.pinned ? "Unpin" : "Pin"}
                        </button>
                        <button onClick={() => { setRowMenu(null); setRenaming({ id: chat.id, title: chat.title }); }}>
                          <Pencil size={15} aria-hidden="true" /> Rename
                        </button>
                        <button onClick={() => { setRowMenu(null); setSharing(chat); }}>
                          <Share2 size={15} aria-hidden="true" /> Share
                        </button>
                        <button className="danger-text" onClick={() => removeChat(chat.id)}>
                          <Trash2 size={15} aria-hidden="true" /> Delete
                        </button>
                      </div>
                    )}
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
            <div className="topbar-actions">
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
              {currentChat && !temporary && (
                <button className="icon-button" onClick={() => setSharing(currentChat)} aria-label="Share chat" title="Share chat">
                  <Share2 size={18} />
                </button>
              )}
              <button
                className={temporary ? "icon-button toggle on" : "icon-button toggle"}
                onClick={() => newChat(!temporary)}
                disabled={sending}
                aria-pressed={temporary}
                aria-label="Temporary chat"
                title={temporary ? "Leave temporary chat" : "Temporary chat: not saved, no memory"}
              >
                <MessageCircleDashed size={18} />
              </button>
            </div>
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
                      {temporary ? "Temporary chat" : mode === "assistant" ? `Hi, I'm ${name}` : greeting(me.username)}
                      <span className="dot">.</span>
                    </h1>
                    <p className="lede">
                      {temporary
                        ? "This chat won't be saved to your history, and nothing from it goes into memory."
                        : mode === "assistant" && me.setup.business_name
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
                {messages.map((message, index) => {
                  if (message.role === "user") {
                    return editing?.index === index ? (
                      <form
                        key={index}
                        className="edit-box glass"
                        onSubmit={(event) => { event.preventDefault(); send(editing.text, index); }}
                      >
                        <textarea
                          autoFocus
                          value={editing.text}
                          maxLength={4000}
                          rows={3}
                          aria-label="Edit message"
                          onChange={(event) => setEditing({ index, text: event.target.value })}
                          onKeyDown={handleEditKey}
                        />
                        <div className="edit-actions">
                          <button type="button" className="button" onClick={() => setEditing(null)}>
                            Cancel
                          </button>
                          <button className="button primary" disabled={!editing.text.trim()}>
                            Send
                          </button>
                        </div>
                      </form>
                    ) : (
                      <div key={index} className="user-turn">
                        <div className="bubble">{message.content}</div>
                        {!sending && (
                          <div className="bubble-actions">
                            <CopyButton text={message.content} label="Copy message" />
                            <button
                              className="icon-button small"
                              onClick={() => setEditing({ index, text: message.content })}
                              aria-label="Edit message"
                              title="Edit"
                            >
                              <Pencil size={15} />
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  }
                  if (index === lastIndex && !message.content && status) {
                    return (
                      <div key={index} className="thinking" role="status">
                        <span className="pulse" aria-hidden="true" />
                        {status}
                        <span className="dots" aria-hidden="true" />
                      </div>
                    );
                  }
                  const finished = !(sending && index === lastIndex);
                  const { id, versions = 0, version = 0 } = message;
                  const actions = (text: string) => (
                    <>
                      {index === lastIndex && id && versions > 1 && (
                        <span className="versions" role="group" aria-label="Reply versions">
                          <button
                            className="icon-button small"
                            onClick={() => showVersion(index, id, version - 1)}
                            disabled={version === 0}
                            aria-label="Previous version"
                          >
                            <ChevronLeft size={16} />
                          </button>
                          <span className="small muted">{version + 1}/{versions}</span>
                          <button
                            className="icon-button small"
                            onClick={() => showVersion(index, id, version + 1)}
                            disabled={version === versions - 1}
                            aria-label="Next version"
                          >
                            <ChevronRight size={16} />
                          </button>
                        </span>
                      )}
                      {!message.error && text && <CopyButton text={text} label="Copy reply" />}
                      {!message.error && text && <SpeakButton text={text} />}
                      {index === lastIndex && (
                        <button
                          className="icon-button small"
                          onClick={() => regenerate(index)}
                          aria-label={message.error ? "Try again" : "Regenerate reply"}
                          title={message.error ? "Try again" : "Regenerate"}
                        >
                          <RotateCcw size={15} />
                        </button>
                      )}
                    </>
                  );
                  return <Reply key={index} message={message} actions={finished ? actions : undefined} />;
                })}
                {followups.length > 0 && (
                  <div className="followups" aria-label="Suggested follow-ups">
                    {followups.map((question) => (
                      <button key={question} className="followup glass" onClick={() => send(question)}>
                        {question}
                      </button>
                    ))}
                  </div>
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
                    <input ref={fileRef} type="file" accept="application/pdf" multiple hidden onChange={attach} />
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
                <button
                  type="button"
                  className={research ? "research-toggle on" : "research-toggle"}
                  onClick={() => setResearch(!research)}
                  disabled={sending}
                  aria-pressed={research}
                  aria-label="Deep research"
                  title="Deep research: more searches and a longer report with sources"
                >
                  <Telescope size={16} aria-hidden="true" />
                  <span className="research-label">Research</span>
                </button>
                <textarea
                  ref={inputRef}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleKey}
                  placeholder={
                    uploading
                      ? "Reading your documents..."
                      : listening
                        ? "Listening..."
                        : research
                          ? "What should I research?"
                          : `Message ${speaker}`
                  }
                  aria-label={`Message ${speaker}`}
                  rows={1}
                  maxLength={4000}
                />
                {canListen && !sending && (
                  <button
                    type="button"
                    className={listening ? "icon-button mic on" : "icon-button mic"}
                    onClick={listen}
                    aria-pressed={listening}
                    aria-label={listening ? "Stop voice typing" : "Voice typing"}
                    title={listening ? "Stop voice typing" : "Voice typing"}
                  >
                    <Mic size={18} />
                  </button>
                )}
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
                {temporary ? "Temporary chat: not saved and not used for memory. " : ""}
                Paradigm can make mistakes, so check important facts.
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
      {sharing && <ShareDialog chatId={sharing.id} title={sharing.title} onClose={() => setSharing(null)} />}
    </div>
  );
}
