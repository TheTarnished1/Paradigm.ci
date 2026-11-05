export type Mode = "assistant" | "general";

export type Setup = {
  ci_name: string;
  business_name: string;
  base_personality: string;
  core_directive: string;
};

export type Me = {
  username: string;
  setup: Setup;
  documents: string[];
  plan: { label: string; messages_per_day: number; tool_calls: number };
  models: { id: string; label: string }[];
  more_models: boolean;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  sources: string[];
  error: boolean;
};

export type ChatEvent =
  | { type: "conversation"; id: string }
  | { type: "text"; text: string }
  | { type: "status"; text: string }
  | { type: "done"; sources: string[] }
  | { type: "error"; text: string };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function failure(response: Response) {
  let message = "Something went wrong. Try again.";
  try {
    const detail = (await response.json()).detail;
    // Validation errors arrive as a list, and aren't written for people
    if (typeof detail === "string") message = detail;
  } catch {}
  return new ApiError(response.status, message);
}

export async function api<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Couldn't reach Paradigm. Check your connection and try again.");
  }
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function uploadDocuments(files: File[]): Promise<{ documents: string[] }> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const response = await fetch("/api/documents", { method: "POST", body: form });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Sends a message and calls onEvent for each piece of the reply as it arrives. */
export async function sendChat(
  body: { message: string; mode: Mode; conversation_id: string | null; model: string | null },
  onEvent: (event: ChatEvent) => void,
) {
  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Couldn't reach Paradigm. Check your connection and try again.");
  }
  if (!response.ok || !response.body) throw await failure(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
  }
}
