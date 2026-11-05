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
    insert,
    select,
    update,
)
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

MAX_FAILED_LOGINS = 5
LOCK_SECONDS = 300
MIN_PASSWORD_LENGTH = 8
MAX_INDEX_BYTES = 30 * 1024 * 1024  # stops one user filling a small free database
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{3,30}")
BUCKET = "assets"  # declared in neon.ts

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
