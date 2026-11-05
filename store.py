"""Accounts, setups and saved chats. Every chat query is scoped to the user who owns it."""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid

MAX_FAILED_LOGINS = 5
LOCK_SECONDS = 300
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{3,30}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    salt BLOB NOT NULL,
    password_hash BLOB NOT NULL,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0,
    identity TEXT,
    doc_names TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    title TEXT NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    error INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations(user_id, mode, updated);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, id);
"""


def data_dir():
    return os.environ.get("PARADIGM_DATA", "data")


def user_docs_dir(user_id):
    return os.path.join(data_dir(), "users", str(int(user_id)), "faiss_index")


# ponytail: SQLite on local disk, fine for one server. It needs a persistent disk, and hosts like
# Streamlit Cloud wipe files on restart. Move to Postgres when hosting or traffic needs it.
def connect():
    os.makedirs(data_dir(), exist_ok=True)
    db = sqlite3.connect(os.path.join(data_dir(), "paradigm.db"))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_db():
    with connect() as db:
        db.executescript(SCHEMA)


def _hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)


def create_user(username, password):
    """Returns (user_id, None), or (None, a message explaining what to fix)."""
    if not USERNAME_PATTERN.fullmatch(username):
        return None, "Usernames need 3–30 letters, numbers, dots, dashes or underscores."
    if len(password) < MIN_PASSWORD_LENGTH:
        return None, f"Passwords need at least {MIN_PASSWORD_LENGTH} characters."
    salt = secrets.token_bytes(16)
    try:
        with connect() as db:
            cursor = db.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, _hash(password, salt)),
            )
            return cursor.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "That username is taken."


def verify_user(username, password):
    """Returns the matching user's (id, username), or None. Wrong passwords lock the account briefly."""
    with connect() as db:
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user is None:
            _hash(password, b"\0" * 16)  # same work as a real check, so timing doesn't reveal usernames
            return None
        matches = hmac.compare_digest(_hash(password, user["salt"]), user["password_hash"])
        if user["locked_until"] > time.time():
            return None
        if matches:
            db.execute("UPDATE users SET failed_logins = 0 WHERE id = ?", (user["id"],))
            return user["id"], user["username"]
        failed = user["failed_logins"] + 1
        if failed >= MAX_FAILED_LOGINS:
            db.execute(
                "UPDATE users SET failed_logins = 0, locked_until = ? WHERE id = ?",
                (time.time() + LOCK_SECONDS, user["id"]),
            )
        else:
            db.execute("UPDATE users SET failed_logins = ? WHERE id = ?", (failed, user["id"]))
        return None


def load_setup(user_id):
    """Returns (identity or None if never saved, document names)."""
    with connect() as db:
        row = db.execute("SELECT identity, doc_names FROM users WHERE id = ?", (user_id,)).fetchone()
    return (json.loads(row["identity"]) if row["identity"] else None), json.loads(row["doc_names"])


def save_setup(user_id, identity, doc_names):
    with connect() as db:
        db.execute(
            "UPDATE users SET identity = ?, doc_names = ? WHERE id = ?",
            (json.dumps(identity), json.dumps(doc_names), user_id),
        )


def create_conversation(user_id, mode, title):
    conversation_id = uuid.uuid4().hex
    with connect() as db:
        db.execute(
            "INSERT INTO conversations (id, user_id, mode, title, updated) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user_id, mode, title, time.time()),
        )
    return conversation_id


def list_conversations(user_id, mode, limit=30):
    with connect() as db:
        rows = db.execute(
            "SELECT id, title FROM conversations WHERE user_id = ? AND mode = ? "
            "ORDER BY updated DESC LIMIT ?",
            (user_id, mode, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(user_id, conversation_id):
    with connect() as db:
        db.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
        )


def load_messages(user_id, conversation_id):
    with connect() as db:
        rows = db.execute(
            "SELECT m.role, m.content, m.sources, m.error FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE m.conversation_id = ? AND c.user_id = ? ORDER BY m.id",
            (conversation_id, user_id),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]),
            "error": bool(row["error"]),
        }
        for row in rows
    ]


def add_message(user_id, conversation_id, role, content, sources=(), error=False):
    with connect() as db:
        owned = db.execute(
            "UPDATE conversations SET updated = ? WHERE id = ? AND user_id = ?",
            (time.time(), conversation_id, user_id),
        ).rowcount
        if not owned:
            raise PermissionError("That chat belongs to another user.")
        db.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, error) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(list(sources)), int(error)),
        )
