import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config

DB_PATH = Path(__file__).resolve().parent / config.DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    payload          TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_by_conversation
    ON messages (conversation_id, id);
"""

TITLE_LENGTH = 60

_ready = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    """
    A connection per operation. SQLite connections are not safe to share
    across threads, and the API serves requests from a thread pool.
    """

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def setup():
    global _ready

    if _ready:
        return

    with connect() as connection:
        connection.executescript(SCHEMA)

    _ready = True


def create_conversation(title: str = None) -> dict:
    setup()

    conversation_id = uuid.uuid4().hex
    timestamp = now()

    with connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, title or "New conversation", timestamp, timestamp),
        )

    return {
        "id": conversation_id,
        "title": title or "New conversation",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": 0,
    }


def conversation_exists(conversation_id: str) -> bool:
    setup()

    with connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

    return row is not None


def list_conversations(limit: int = 50) -> list[dict]:
    setup()

    with connect() as connection:
        rows = connection.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, "
            "       COUNT(m.id) AS messages "
            "FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id "
            "GROUP BY c.id "
            "ORDER BY c.updated_at DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_messages(conversation_id: str) -> list[dict]:
    setup()

    with connect() as connection:
        rows = connection.execute(
            "SELECT role, content, payload, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:
        message = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }

        if row["payload"]:
            message["payload"] = json.loads(row["payload"])

        messages.append(message)

    return messages


def add_message(conversation_id: str, role: str, content: str,
                payload: dict = None):
    setup()

    timestamp = now()

    with connect() as connection:
        connection.execute(
            "INSERT INTO messages "
            "(conversation_id, role, content, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                json.dumps(payload) if payload else None,
                timestamp,
            ),
        )

        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )

        # The first question names the conversation, so the sidebar shows
        # something meaningful instead of "New conversation".
        if role == "user":
            connection.execute(
                "UPDATE conversations SET title = ? "
                "WHERE id = ? AND title = 'New conversation'",
                (summarise_title(content), conversation_id),
            )


def summarise_title(question: str) -> str:
    title = " ".join(question.split())

    if len(title) <= TITLE_LENGTH:
        return title

    return title[:TITLE_LENGTH].rstrip() + "..."


def delete_conversation(conversation_id: str) -> bool:
    setup()

    with connect() as connection:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )

        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )

    return cursor.rowcount > 0


def history_for(conversation_id: str, max_messages: int = None) -> list[dict]:
    """
    The recent turns in the shape the prompts expect: role and content only.
    """

    max_messages = max_messages or config.MEMORY_MAX_MESSAGES

    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in get_messages(conversation_id)
    ]

    return messages[-max_messages:]
