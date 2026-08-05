"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import Message from "@/components/Message";
import Sidebar from "@/components/Sidebar";
import {
  askStream,
  createConversation,
  deleteConversation,
  getConversation,
  getHealth,
  getSettings,
  listConversations,
} from "@/lib/api";
import type {
  Conversation,
  Health,
  Message as MessageData,
  Settings,
} from "@/lib/types";

const EXAMPLES = [
  "Summarise chapter 2",
  "What does the document say about eligibility?",
  "How many days are there between 2026-01-01 and today?",
];

export default function Page() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [agent, setAgent] = useState(true);
  const [fast, setFast] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await listConversations());
    } catch {
      // The sidebar is not worth an error banner.
    }
  }, []);

  useEffect(() => {
    getSettings()
      .then((loaded) => {
        setSettings(loaded);
        setAgent(loaded.agent);
      })
      .catch(() => setError("Cannot reach the API. Is uvicorn running?"));

    getHealth().then(setHealth).catch(() => setHealth(null));

    refreshConversations();
  }, [refreshConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function openConversation(id: string) {
    setActiveId(id);
    setError(null);

    try {
      const conversation = await getConversation(id);
      setMessages(conversation.messages);
    } catch {
      setMessages([]);
    }
  }

  function startNewChat() {
    setActiveId(null);
    setMessages([]);
    setError(null);
    textareaRef.current?.focus();
  }

  async function removeConversation(id: string) {
    await deleteConversation(id).catch(() => null);

    if (id === activeId) startNewChat();

    refreshConversations();
  }

  async function send(text: string) {
    const trimmed = text.trim();

    if (!trimmed || busy) return;

    setBusy(true);
    setError(null);
    setQuestion("");

    let conversationId = activeId;

    if (!conversationId) {
      try {
        conversationId = (await createConversation()).id;
        setActiveId(conversationId);
      } catch {
        setError("Cannot reach the API. Is uvicorn running?");
        setBusy(false);
        return;
      }
    }

    setMessages((current) => [
      ...current,
      { role: "user", content: trimmed },
      { role: "assistant", content: "", pending: true },
    ]);

    /** Replace the trailing placeholder as events arrive. */
    const updateLast = (patch: Partial<MessageData>) =>
      setMessages((current) => {
        const next = [...current];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });

    try {
      let streamed = "";

      await askStream(
        { question: trimmed, conversation_id: conversationId, agent, fast },
        (event) => {
          if (event.type === "retrieval") {
            updateLast({
              payload: {
                question: trimmed,
                answer: "",
                answered: false,
                sources: [],
                retrieval: event.trace,
              },
            });
          } else if (event.type === "token") {
            streamed += event.text;
            updateLast({ content: streamed });
          } else if (event.type === "done") {
            updateLast({
              content: event.payload.answer,
              payload: event.payload,
              pending: false,
            });
          } else if (event.type === "error") {
            setError(event.message);
            updateLast({ pending: false });
          }
        },
      );
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "The request failed",
      );
      updateLast({ pending: false });
    } finally {
      setBusy(false);
      refreshConversations();
    }
  }

  return (
    <div className="shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        settings={settings}
        onSelect={openConversation}
        onDelete={removeConversation}
        onNew={startNewChat}
      />

      <main className="main">
        <div className="topbar">
          <span
            className={`status-dot ${health?.ingested ? "" : "bad"}`}
            title={
              health
                ? `${health.chunks} chunks indexed`
                : "API unreachable"
            }
          />
          <span style={{ fontSize: 13, color: "var(--muted)" }}>
            {health
              ? `${health.chunks} chunks indexed`
              : "waiting for the API"}
          </span>

          <div className="topbar-spacer" />

          <button
            className={`toggle ${fast ? "on" : ""}`}
            onClick={() => setFast((value) => !value)}
            title={
              settings
                ? `Answer with ${settings.fast_model} instead of ${settings.model}. Much faster, noticeably weaker.`
                : "Answer with the small model"
            }
          >
            fast {fast ? "on" : "off"}
          </button>

          <button
            className={`toggle ${agent ? "on" : ""}`}
            onClick={() => setAgent((value) => !value)}
            title="Plan and execute agent: decomposes the question into tool calls"
          >
            agent {agent ? "on" : "off"}
          </button>
        </div>

        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty">
              <h2>Ask the documents</h2>
              <p>
                Hybrid retrieval with query expansion, HyDE, reranking and
                compression. Every answer shows how it was found.
              </p>
              <div className="examples">
                {EXAMPLES.map((example) => (
                  <button
                    className="example"
                    key={example}
                    onClick={() => send(example)}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread">
              {messages.map((message, index) => (
                <Message key={index} message={message} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer">
          {error ? <div className="error">{error}</div> : null}

          <div className="composer-inner">
            <textarea
              ref={textareaRef}
              rows={1}
              value={question}
              placeholder="Ask about the indexed documents..."
              onChange={(event) => {
                setQuestion(event.target.value);
                event.target.style.height = "auto";
                event.target.style.height = `${event.target.scrollHeight}px`;
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  send(question);
                }
              }}
            />
            <button
              className="send"
              disabled={busy || !question.trim()}
              onClick={() => send(question)}
            >
              {busy ? "..." : "Send"}
            </button>
          </div>

          <div className="hint">
            Enter to send, Shift+Enter for a new line
            {health && !health.ingested
              ? " · nothing indexed yet, run python ingest.py"
              : ""}
          </div>
        </div>
      </main>
    </div>
  );
}
