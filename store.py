"""Accounts, setups, documents and saved chats. Every chat query is scoped to the user who owns it.

Uses Postgres when DATABASE_URL is set, and a local SQLite file otherwise. Document indexes go to
Neon object storage when AWS_ENDPOINT_URL_S3 is set, and to the database otherwise.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import urllib.request
import uuid
from functools import lru_cache

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    update,
)
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

MAX_FAILED_LOGINS = 5
LOCK_SECONDS = 300
SESSION_SECONDS = 30 * 24 * 60 * 60
MIN_PASSWORD_LENGTH = 8
MAX_INDEX_BYTES = 30 * 1024 * 1024  # stops one user filling a small free database
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{3,30}")
BUCKET = "assets"  # declared in neon.ts

MINUTE = 60
DAY = 24 * 60 * 60

MODELS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "qwen/qwen3.8-27b": "Qwen 3.8",
}
DEFAULT_PLAN = "free"
# ponytail: Paid is set by hand with `python store.py set-plan <username> paid`, with no payments yet.
# Everyone shares one Groq key, so Paid can't exceed the shared limits below until that key is on a
# paid Groq plan. Add real billing and raise the shared limits together.
PLANS = {
    "free": {
        "label": "Free",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "messages_per_minute": 6,
        "messages_per_day": 30,
        "tool_calls": 2,
    },
    "paid": {
        "label": "Paid",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        "messages_per_minute": 15,
        "messages_per_day": 100,
        "tool_calls": 10,
    },
}

# Each rule: (per user?, most allowed, time window in seconds, what to tell the user).
# ponytail: counts requests, not tokens. Groq's free tier allows 30 requests a minute and 200K tokens
# a day, so 150 model calls a day assumes about 1,300 tokens each. Raise these on a paid Groq plan.
LIMITS = {
    "model_call": [
        (False, 20, MINUTE, "Paradigm is busy right now. Please try again in a minute."),
        (False, 150, DAY, "Paradigm has reached today's limit for everyone. Please try again tomorrow."),
    ],
    "upload": [
        (True, 5, DAY, "You've reached today's limit of 5 document uploads. Please come back tomorrow."),
    ],
    "recovery": [
        (False, 30, 60 * MINUTE, "Too many password resets right now. Please try again in an hour."),
    ],
    "signup": [
        (False, 20, 60 * MINUTE, "Too many new accounts right now. Please try again in an hour."),
    ],
}

metadata = MetaData()
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(30), nullable=False),
    Column("username_key", String(30), nullable=False, unique=True),  # lowercase, for matching
    Column("salt", LargeBinary, nullable=False),
    Column("password_hash", LargeBinary, nullable=False),
    Column("failed_logins", Integer, nullable=False, default=0),
    Column("locked_until", Float, nullable=False, default=0),
    Column("identity", Text),
    Column("doc_names", Text, nullable=False, default="[]"),
)
documents = Table(
    "documents",
    metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("index_data", LargeBinary, nullable=False),
)
conversations = Table(
    "conversations",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("mode", String(20), nullable=False),
    Column("title", Text, nullable=False),
    Column("updated", Float, nullable=False),
    Index("conversations_by_user", "user_id", "mode", "updated"),
)
messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True),
    Column(
        "conversation_id",
        String(32),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("sources", Text, nullable=False, default="[]"),
    Column("error", Integer, nullable=False, default=0),
    Index("messages_by_conversation", "conversation_id", "id"),
)
sessions = Table(
    "sessions",
    metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("expires", Float, nullable=False),
)
recovery_codes = Table(
    "recovery_codes",
    metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("code_hash", String(64), nullable=False),
)
plans = Table(
    "plans",
    metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("plan", String(20), nullable=False),
)
usage = Table(
    "usage",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("kind", String(20), nullable=False),
    Column("user_id", Integer),
    Column("at", Float, nullable=False),
    Index("usage_by_kind", "kind", "at"),
    Index("usage_by_user", "user_id", "kind", "at"),
)


def database_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        return re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", url)
    folder = os.environ.get("PARADIGM_DATA", "data")
    os.makedirs(folder, exist_ok=True)
    return f"sqlite:///{os.path.join(folder, 'paradigm.db')}"


@lru_cache(maxsize=1)
def engine():
    # pre_ping and recycle: free hosted databases sleep and drop idle connections
    url = database_url()
    # Neon's pooled connections run through PgBouncer, so psycopg's prepared statements stay off
    connect_args = {"prepare_threshold": None} if url.startswith("postgresql") else {}
    db_engine = create_engine(url, pool_pre_ping=True, pool_recycle=300, connect_args=connect_args)
    if db_engine.dialect.name == "sqlite":
        event.listen(
            db_engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys = ON"),
        )
    return db_engine


def init_db():
    metadata.create_all(engine())


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
        with engine().begin() as db:
            result = db.execute(
                insert(users).values(
                    username=username,
                    username_key=username.lower(),
                    salt=salt,
                    password_hash=_hash(password, salt),
                )
            )
            return result.inserted_primary_key[0], None
    except IntegrityError:
        return None, "That username is taken."


def verify_user(username, password):
    """Returns the matching user's (id, username), or None. Wrong passwords lock the account briefly."""
    with engine().begin() as db:
        user = db.execute(
            select(users).where(users.c.username_key == username.lower())
        ).mappings().first()
        if user is None:
            _hash(password, b"\0" * 16)  # same work as a real check, so timing doesn't reveal usernames
            return None
        matches = hmac.compare_digest(_hash(password, user["salt"]), user["password_hash"])
        if user["locked_until"] > time.time():
            return None
        this_user = update(users).where(users.c.id == user["id"])
        if matches:
            db.execute(this_user.values(failed_logins=0))
            return user["id"], user["username"]
        failed = user["failed_logins"] + 1
        if failed >= MAX_FAILED_LOGINS:
            db.execute(this_user.values(failed_logins=0, locked_until=time.time() + LOCK_SECONDS))
        else:
            db.execute(this_user.values(failed_logins=failed))
        return None


def _token_hash(token):
    # A plain hash is enough here: these tokens are long and random, unlike passwords
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id):
    """Returns a new login token. Only its hash is stored."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    with engine().begin() as db:
        db.execute(delete(sessions).where(sessions.c.expires < now))
        db.execute(
            insert(sessions).values(
                token_hash=_token_hash(token), user_id=user_id, expires=now + SESSION_SECONDS
            )
        )
    return token


def user_from_session(token):
    """Returns the (id, username) a login token belongs to, or None."""
    with engine().connect() as db:
        row = db.execute(
            select(users.c.id, users.c.username)
            .join(sessions, sessions.c.user_id == users.c.id)
            .where(sessions.c.token_hash == _token_hash(token), sessions.c.expires > time.time())
        ).first()
    return (row[0], row[1]) if row else None


def end_session(token):
    with engine().begin() as db:
        db.execute(delete(sessions).where(sessions.c.token_hash == _token_hash(token)))


def new_recovery_code(user_id):
    """Returns a new one-time recovery code, replacing any old one. Only its hash is stored."""
    code = secrets.token_urlsafe(18)
    with engine().begin() as db:
        db.execute(delete(recovery_codes).where(recovery_codes.c.user_id == user_id))
        db.execute(insert(recovery_codes).values(user_id=user_id, code_hash=_token_hash(code)))
    return code


def reset_password(username, code, new_password):
    """Sets a new password using a recovery code. Returns None, or a message explaining the problem."""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return f"Passwords need at least {MIN_PASSWORD_LENGTH} characters."
    wrong = "That username and recovery code don't match."
    salt = secrets.token_bytes(16)
    with engine().begin() as db:
        row = db.execute(
            select(users.c.id, recovery_codes.c.code_hash)
            .join(recovery_codes, recovery_codes.c.user_id == users.c.id)
            .where(users.c.username_key == username.lower())
        ).first()
        if row is None or not hmac.compare_digest(row[1], _token_hash(code)):
            return wrong
        db.execute(
            update(users)
            .where(users.c.id == row[0])
            .values(salt=salt, password_hash=_hash(new_password, salt), failed_logins=0, locked_until=0)
        )
        db.execute(delete(recovery_codes).where(recovery_codes.c.user_id == row[0]))
        db.execute(delete(sessions).where(sessions.c.user_id == row[0]))
    return None


def export_user_data(user_id):
    """Returns everything saved for this user that they can read, as JSON-ready data."""
    identity, doc_names = load_setup(user_id)
    chats = []
    with engine().connect() as db:
        username = db.execute(select(users.c.username).where(users.c.id == user_id)).scalar_one()
        rows = db.execute(
            select(conversations.c.id, conversations.c.mode, conversations.c.title).where(
                conversations.c.user_id == user_id
            )
        ).all()
    for chat_id, mode, title in rows:
        chats.append({"mode": mode, "title": title, "messages": load_messages(user_id, chat_id)})
    return {"username": username, "setup": identity, "documents": doc_names, "chats": chats}


def delete_account(user_id, password):
    """Deletes the account and all its data if the password is right. Returns True if deleted."""
    with engine().begin() as db:
        user = db.execute(
            select(users.c.salt, users.c.password_hash).where(users.c.id == user_id)
        ).first()
        if user is None or not hmac.compare_digest(_hash(password, user[0]), user[1]):
            return False
        bucket = _bucket()
        if bucket:
            bucket.delete_object(Bucket=BUCKET, Key=_document_key(user_id))
        db.execute(delete(usage).where(usage.c.user_id == user_id))
        db.execute(delete(users).where(users.c.id == user_id))
    return True


def load_setup(user_id):
    """Returns (identity or None if never saved, document names)."""
    with engine().connect() as db:
        row = db.execute(
            select(users.c.identity, users.c.doc_names).where(users.c.id == user_id)
        ).mappings().one()
    return (json.loads(row["identity"]) if row["identity"] else None), json.loads(row["doc_names"])


def save_setup(user_id, identity, doc_names):
    with engine().begin() as db:
        db.execute(
            update(users)
            .where(users.c.id == user_id)
            .values(identity=json.dumps(identity), doc_names=json.dumps(doc_names))
        )


def _bucket():
    """Returns an S3 client for Neon object storage, or None to keep documents in the database."""
    if not os.environ.get("AWS_ENDPOINT_URL_S3"):
        return None
    import boto3  # only needed when object storage is configured
    from botocore.config import Config

    # Credentials, endpoint and region come from the AWS_* variables. Neon needs path-style URLs.
    return boto3.client("s3", config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"))


def _document_key(user_id):
    return f"documents/{int(user_id)}.faiss"


def service_status():
    """Checks each configured service. Returns {service name: working}."""
    status = {}
    try:
        with engine().connect() as db:
            db.execute(select(1))
        status["Database"] = True
    except SQLAlchemyError:
        status["Database"] = False

    bucket = _bucket()
    if bucket:
        try:
            bucket.head_bucket(Bucket=BUCKET)
            status["Document storage"] = True
        except (BotoCoreError, ClientError):
            status["Document storage"] = False

    function_url = os.environ.get("NEON_FUNCTION_HELLO_BASE_URL")
    if function_url:
        try:
            with urllib.request.urlopen(function_url, timeout=10) as response:
                status["Functions"] = response.status == 200
        except OSError:
            status["Functions"] = False
    return status


def load_documents(user_id):
    """Returns this user's saved document index as bytes, or None."""
    bucket = _bucket()
    if bucket:
        try:
            return bucket.get_object(Bucket=BUCKET, Key=_document_key(user_id))["Body"].read()
        except bucket.exceptions.NoSuchKey:
            return None
    with engine().connect() as db:
        return db.execute(
            select(documents.c.index_data).where(documents.c.user_id == user_id)
        ).scalar_one_or_none()


def save_documents(user_id, index_data):
    bucket = _bucket()
    if bucket:
        bucket.put_object(Bucket=BUCKET, Key=_document_key(user_id), Body=index_data)
        return
    with engine().begin() as db:
        db.execute(delete(documents).where(documents.c.user_id == user_id))
        db.execute(insert(documents).values(user_id=user_id, index_data=index_data))


def get_plan(user_id):
    with engine().connect() as db:
        plan = db.execute(select(plans.c.plan).where(plans.c.user_id == user_id)).scalar_one_or_none()
    return plan if plan in PLANS else DEFAULT_PLAN


def set_plan(username, plan):
    """Sets a user's plan. Returns False if there's no such user or plan."""
    if plan not in PLANS:
        return False
    with engine().begin() as db:
        user_id = db.execute(
            select(users.c.id).where(users.c.username_key == username.lower())
        ).scalar_one_or_none()
        if user_id is None:
            return False
        db.execute(delete(plans).where(plans.c.user_id == user_id))
        db.execute(insert(plans).values(user_id=user_id, plan=plan))
    return True


def _message_rules(plan):
    details = PLANS[plan]
    per_day = details["messages_per_day"]
    return [
        (True, details["messages_per_minute"], MINUTE, "You're sending messages quickly. Wait a minute and try again."),
        (True, per_day, DAY, f"You've reached today's limit of {per_day} messages. Please come back tomorrow."),
    ]


def limit_message(kind, user_id=None, plan=None):
    """Returns what to tell the user if this action is over a limit, otherwise None."""
    now = time.time()
    rules = LIMITS.get(kind, []) + (_message_rules(plan) if plan else [])
    with engine().connect() as db:
        for per_user, most, window, message in rules:
            query = (
                select(func.count())
                .select_from(usage)
                .where(usage.c.kind == kind, usage.c.at > now - window)
            )
            if per_user:
                query = query.where(usage.c.user_id == user_id)
            if db.execute(query).scalar() >= most:
                return message
    return None


# ponytail: checking then recording can let a burst slip slightly over a limit. That's fine for
# protecting a quota, and Groq's own limits still apply. Use a transaction lock if limits get strict.
def record_usage(kind, user_id=None):
    now = time.time()
    with engine().begin() as db:
        db.execute(insert(usage).values(kind=kind, user_id=user_id, at=now))
        db.execute(delete(usage).where(usage.c.at < now - 2 * DAY))


def create_conversation(user_id, mode, title):
    conversation_id = uuid.uuid4().hex
    with engine().begin() as db:
        db.execute(
            insert(conversations).values(
                id=conversation_id, user_id=user_id, mode=mode, title=title, updated=time.time()
            )
        )
    return conversation_id


def list_conversations(user_id, mode, limit=30):
    with engine().connect() as db:
        rows = db.execute(
            select(conversations.c.id, conversations.c.title)
            .where(conversations.c.user_id == user_id, conversations.c.mode == mode)
            .order_by(conversations.c.updated.desc())
            .limit(limit)
        ).mappings().all()
    return [dict(row) for row in rows]


def delete_conversation(user_id, conversation_id):
    with engine().begin() as db:
        db.execute(
            delete(conversations).where(
                conversations.c.id == conversation_id, conversations.c.user_id == user_id
            )
        )


def load_messages(user_id, conversation_id):
    with engine().connect() as db:
        rows = db.execute(
            select(messages.c.role, messages.c.content, messages.c.sources, messages.c.error)
            .join(conversations, conversations.c.id == messages.c.conversation_id)
            .where(messages.c.conversation_id == conversation_id, conversations.c.user_id == user_id)
            .order_by(messages.c.id)
        ).mappings().all()
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
    with engine().begin() as db:
        owned = db.execute(
            update(conversations)
            .where(conversations.c.id == conversation_id, conversations.c.user_id == user_id)
            .values(updated=time.time())
        ).rowcount
        if not owned:
            raise PermissionError("That chat belongs to another user.")
        db.execute(
            insert(messages).values(
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources=json.dumps(list(sources)),
                error=int(error),
            )
        )


if __name__ == "__main__":
    # python store.py set-plan <username> <free|paid>
    import sys

    from dotenv import load_dotenv

    load_dotenv(".env.local")
    load_dotenv()
    if len(sys.argv) != 4 or sys.argv[1] != "set-plan":
        sys.exit("Usage: python store.py set-plan <username> <free|paid>")
    init_db()
    sys.exit(0 if set_plan(sys.argv[2], sys.argv[3]) else "No such user or plan.")
