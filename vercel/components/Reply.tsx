"use client";

import { Check, Copy, FileText, Globe, Square, Volume2 } from "lucide-react";
import { ReactNode, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message, Source } from "@/lib/api";

// Some models add markers like 【web_search】 where they used a tool. The server removes them when it saves.
const TOOL_MARKERS = /\s?【[^】]{0,40}】/g;
// Replies end with a FOLLOWUPS: line that becomes buttons. While it streams in, even its first letters are hidden.
const FOLLOWUP_LINE = /\n[ \t]*\**(?:follow-?ups?\**:[^\n]*|F(?:O(?:L(?:L(?:O(?:W(?:-?(?:U(?:P(?:S)?)?)?)?)?)?)?)?)?)\s*$/i;

export function visibleText(text: string) {
  return text.replace(TOOL_MARKERS, "").replace(FOLLOWUP_LINE, "");
}

function sourceParts(source: Source) {
  return typeof source === "string" ? { label: source, url: "" } : { label: source.label, url: source.url ?? "" };
}

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
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

function spoken(text: string) {
  return text
    .replace(/```[\s\S]*?```/g, " (code) ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[#*_`>|~]/g, "");
}

/** Reads a reply aloud with the browser's own voice, where the browser supports it. */
export function SpeakButton({ text }: { text: string }) {
  const [supported, setSupported] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  useEffect(() => setSupported("speechSynthesis" in window), []);
  useEffect(() => () => { if (speaking) speechSynthesis.cancel(); }, [speaking]);
  if (!supported) return null;

  function toggle() {
    speechSynthesis.cancel();
    if (speaking) return setSpeaking(false);
    const utterance = new SpeechSynthesisUtterance(spoken(text));
    utterance.onend = utterance.onerror = () => setSpeaking(false);
    speechSynthesis.speak(utterance);
    setSpeaking(true);
  }
  const label = speaking ? "Stop reading" : "Read aloud";
  return (
    <button className="icon-button small" onClick={toggle} aria-label={label} title={label} aria-pressed={speaking}>
      {speaking ? <Square size={14} fill="currentColor" /> : <Volume2 size={16} />}
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
        const content = (
          <>
            <span className="source-icon" aria-hidden="true">
              {url ? <Globe size={14} /> : <FileText size={14} />}
            </span>
            <span className="source-label">{label}</span>
          </>
        );
        return url ? (
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

/** One assistant reply. `actions` receives the reply's visible text, for copy and read-aloud buttons. */
export default function Reply({ message, actions }: { message: Message; actions?: (text: string) => ReactNode }) {
  const text = visibleText(message.content);
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
        {actions && <div className="reply-actions">{actions(text)}</div>}
      </div>
    </article>
  );
}
