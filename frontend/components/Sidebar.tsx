"use client";

import type { Conversation, Settings } from "@/lib/types";

type Props = {
  conversations: Conversation[];
  activeId: string | null;
  settings: Settings | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
};

export default function Sidebar({
  conversations,
  activeId,
  settings,
  onSelect,
  onDelete,
  onNew,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div className="brand">Semantic Search</div>
        <button className="new-chat" onClick={onNew}>
          + New chat
        </button>
      </div>

      <div className="conversations">
        {conversations.map((conversation) => (
          <div
            key={conversation.id}
            className={`conversation ${
              conversation.id === activeId ? "active" : ""
            }`}
            onClick={() => onSelect(conversation.id)}
          >
            <span className="conversation-title">{conversation.title}</span>
            <button
              className="conversation-delete"
              title="Delete conversation"
              onClick={(event) => {
                event.stopPropagation();
                onDelete(conversation.id);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {settings ? (
        <div className="sidebar-foot">
          <div>
            answer <code>{settings.model}</code>
          </div>
          <div>
            utility <code>{settings.utility_model}</code>
          </div>
          <div>
            embed <code>{settings.embedding_model}</code>
          </div>
        </div>
      ) : null}
    </aside>
  );
}
