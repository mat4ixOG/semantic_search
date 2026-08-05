import type {
  Conversation,
  Health,
  Message,
  Settings,
  StreamEvent,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${path} failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const getHealth = () => request<Health>("/api/health");

export const getSettings = () => request<Settings>("/api/config");

export const listConversations = () =>
  request<{ conversations: Conversation[] }>("/api/conversations").then(
    (data) => data.conversations,
  );

export const createConversation = () =>
  request<Conversation>("/api/conversations", { method: "POST" });

export const getConversation = (id: string) =>
  request<{ id: string; messages: Message[] }>(`/api/conversations/${id}`);

export const deleteConversation = (id: string) =>
  request<{ deleted: string }>(`/api/conversations/${id}`, {
    method: "DELETE",
  });

/**
 * POST + ReadableStream rather than EventSource, which only speaks GET and
 * could not carry the question in a body.
 */
export async function askStream(
  body: {
    question: string;
    conversation_id: string | null;
    agent: boolean;
    fast: boolean;
  },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream failed with ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A partial frame stays in the
    // buffer until the rest of it arrives.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame
        .split("\n")
        .find((candidate) => candidate.startsWith("data:"));

      if (!line) continue;

      try {
        onEvent(JSON.parse(line.slice(5).trim()) as StreamEvent);
      } catch {
        // A malformed frame should not tear down the whole stream.
      }
    }
  }
}
