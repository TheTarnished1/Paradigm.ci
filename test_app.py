# Run with: python test_app.py  (no Groq calls or model downloads happen here)
import io
import json
import os
import shutil
import tempfile
import time
from unittest.mock import patch

data_folder = tempfile.mkdtemp()
os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["PARADIGM_DATA"] = data_folder  # keeps test accounts out of the real database
# Blank values stop the app loading production settings from .env.local during tests
for name in ["DATABASE_URL", "AWS_ENDPOINT_URL_S3", "NEON_FUNCTION_HELLO_BASE_URL"]:
    os.environ[name] = ""
real_services = os.environ.get("TEST_ENV_FILE")  # variables for an empty, throwaway Neon branch
if real_services:
    from dotenv import dotenv_values

    os.environ.update(dotenv_values(real_services))

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import delete, func, select, update
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

import agent
import store
from ci_setup_tools import build_index, model_status, read_pdf

PASSWORD = "test-password-123"
store.init_db()

# --- Accounts ---
alice, problem = store.create_user("alice", PASSWORD)
bob, _ = store.create_user("bob", PASSWORD)
assert alice and bob and problem is None
assert store.create_user("ALICE", PASSWORD)[0] is None, "usernames ignore case"
assert store.create_user("x", PASSWORD)[0] is None, "username too short"
assert store.create_user("no spaces", PASSWORD)[0] is None, "bad username characters"
assert store.create_user("shortpw", "1234567")[0] is None, "password too short"
assert store.verify_user("alice", PASSWORD) == (alice, "alice")
assert store.verify_user("Alice", PASSWORD) == (alice, "alice")
assert store.verify_user("alice", "wrong-password") is None
assert store.verify_user("nobody", PASSWORD) is None

# Passwords are never stored as text
with store.engine().connect() as db:
    stored = db.execute(
        select(store.users.c.password_hash).where(store.users.c.id == alice)
    ).scalar_one()
assert PASSWORD.encode() not in stored

# Repeated failures lock the account, even for the right password, then it recovers
carol, _ = store.create_user("carol", PASSWORD)
for _ in range(store.MAX_FAILED_LOGINS):
    assert store.verify_user("carol", "wrong-password") is None
assert store.verify_user("carol", PASSWORD) is None
with store.engine().begin() as db:
    db.execute(update(store.users).where(store.users.c.id == carol).values(locked_until=0))
assert store.verify_user("carol", PASSWORD) == (carol, "carol")

# --- One user can't reach another user's chats ---
chat = store.create_conversation(alice, "assistant", "Private")
store.add_message(alice, chat, "user", "alice secret")
assert store.load_messages(bob, chat) == []
assert store.list_conversations(bob, "assistant") == []
store.delete_conversation(bob, chat)
assert store.load_messages(alice, chat)[0]["content"] == "alice secret"
try:
    store.add_message(bob, chat, "user", "intruder")
    raise AssertionError("bob wrote to alice's chat")
except PermissionError:
    pass
assert len(store.load_messages(alice, chat)) == 1
store.delete_conversation(alice, chat)
assert store.list_conversations(alice, "assistant") == []


# --- The app ---
class FakeChatGroq:
    fail = False
    calls = []
    models = []

    def __init__(self, **kwargs):
        self.model = kwargs.get("model_name")

    def bind_tools(self, tools):
        return self

    def stream(self, chat):
        FakeChatGroq.calls.append(list(chat))
        FakeChatGroq.models.append(self.model)
        if FakeChatGroq.fail:
            raise RuntimeError("secret-upstream-detail")
        yield AIMessageChunk(content="Hello ")
        yield AIMessageChunk(content="there")


def start():
    return AppTest.from_file("Ci_app.py", default_timeout=300 if real_services else 60).run()


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def field(at, key):
    return next(t for t in at.text_input if t.key == key)


def login(username):
    at = start()
    field(at, "login_username").set_value(username)
    field(at, "login_password").set_value(PASSWORD)
    button(at, "Log in").click().run()
    assert not at.exception, at.exception
    return at


def send(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception


def sent_to_model():
    return [m.content for m in FakeChatGroq.calls[-1][1:]]


with patch("langchain_groq.ChatGroq", FakeChatGroq):
    st.cache_resource.clear()

    # Nothing is reachable before logging in
    at = start()
    assert not at.exception, at.exception
    assert at.title[0].value == "Paradigm" and not at.chat_input

    # A wrong password is refused
    field(at, "login_username").set_value("alice")
    field(at, "login_password").set_value("wrong-password")
    button(at, "Log in").click().run()
    assert at.error and "user_id" not in at.session_state

    # Signing up works through the form, and logs you in on the home screen
    at = start()
    field(at, "signup_username").set_value("dana")
    field(at, "signup_password").set_value(PASSWORD)
    field(at, "signup_confirm").set_value("different-password")
    button(at, "Create account").click().run()
    assert at.error and "user_id" not in at.session_state
    field(at, "signup_confirm").set_value(PASSWORD)
    button(at, "Create account").click().run()
    assert not at.exception, at.exception
    button(at, "I've saved it").click().run()
    assert at.title[0].value == "Welcome, dana"
    assert not at.chat_input, "home screen asks for a mode first"

    # The home screen opens each mode
    button(at, "Open General chat").click().run()
    assert at.title[0].value == "General chat" and at.chat_input

    # Logging in opens on the choice of modes
    at = login("alice")
    assert at.title[0].value == "Welcome, alice"

    # Assistant mode uses the saved setup
    store.save_setup(
        alice,
        {
            "ci_name": "LogiBot",
            "business_name": "LogiCorp",
            "base_personality": "Calm",
            "core_directive": "Answer shipping questions",
        },
        [],
    )
    at = login("alice")
    button(at, "Open Assistant").click().run()
    assert at.title[0].value == "Hi, I'm LogiBot"
    send(at, "What is torque?")
    system = FakeChatGroq.calls[-1][0].content
    for expected in ["LogiBot", "LogiCorp", "Calm", "Answer shipping questions"]:
        assert expected in system, system
    assert "Hello there" in page_text(at) and not at.title

    # The current message is sent once, and earlier messages go in order
    send(at, "Second question")
    assert sent_to_model() == ["What is torque?", "Hello there", "Second question"], sent_to_model()

    # Failures show a friendly message, hide details, and stay out of later requests
    FakeChatGroq.fail = True
    send(at, "This one fails")
    assert "couldn't answer" in page_text(at)
    assert "secret-upstream-detail" not in page_text(at)
    FakeChatGroq.fail = False
    send(at, "Back again")
    assert "Sorry, I couldn't answer that. Please try again." not in sent_to_model()

    # General chat ignores the setup, and keeps its chats apart from Assistant chats
    button(at, "General chat").click().run()
    assert at.title[0].value == "General chat"
    assert not [b for b in at.button if b.label == "What is torque?"]
    send(at, "Hi")
    system = FakeChatGroq.calls[-1][0].content
    assert "LogiBot" not in system and "Answer shipping questions" not in system

    # Chats are still there after logging back in
    at = login("alice")
    button(at, "Open Assistant").click().run()
    button(at, "What is torque?").click().run()
    assert "Back again" in page_text(at)

    # Another user sees none of them
    at = login("bob")
    button(at, "Open Assistant").click().run()
    assert not [b for b in at.button if b.label == "What is torque?"]
    assert at.title[0].value == "Hi, I'm Paradigm"

    # A suggestion sends a message
    button(at, "Help me plan my week").click().run()
    assert not at.exception, at.exception
    assert FakeChatGroq.calls[-1][-1].content == "Help me plan my week"

    # Deleting a chat removes it
    conversation = store.list_conversations(bob, "assistant")[0]["id"]
    next(b for b in at.button if b.key == f"delete_{conversation}").click().run()
    assert store.list_conversations(bob, "assistant") == []

    # A user over the limit is told so, and nothing is sent or saved
    erin, _ = store.create_user("erin", PASSWORD)
    with patch.dict(store.PLANS["free"], {"messages_per_minute": 2}):
        at = login("erin")
        button(at, "Open General chat").click().run()
        send(at, "one")
        send(at, "two")
        calls_before = len(FakeChatGroq.calls)
        send(at, "three")
        assert len(FakeChatGroq.calls) == calls_before, "a blocked message must not reach Groq"
        assert at.warning and "quickly" in at.warning[0].value
        chat_id = store.list_conversations(erin, "general")[0]["id"]
        assert "three" not in [m["content"] for m in store.load_messages(erin, chat_id)]

        # Another user isn't affected by erin's limit
        at = login("bob")
        button(at, "Open General chat").click().run()
        send(at, "still fine")
        assert not at.warning

    # A limit for everyone applies across users
    with patch.dict(store.LIMITS, {"model_call": [(False, 1, store.MINUTE, "Everyone limit test")]}):
        at = login("bob")
        button(at, "Open General chat").click().run()
        send(at, "blocked")
        assert at.warning and at.warning[0].value == "Everyone limit test"

    # Sign-ups stop at the limit, and failed attempts don't use it up
    with patch.dict(store.LIMITS, {"signup": [(False, 1, 3600, "Signup limit test")]}):
        with store.engine().begin() as db:
            db.execute(delete(store.usage).where(store.usage.c.kind == "signup"))
        at = start()
        field(at, "signup_username").set_value("x")
        field(at, "signup_password").set_value(PASSWORD)
        field(at, "signup_confirm").set_value(PASSWORD)
        button(at, "Create account").click().run()
        assert at.error and "Signup limit test" not in at.error[0].value, "a bad name isn't counted"
        field(at, "signup_username").set_value("frank")
        button(at, "Create account").click().run()
        button(at, "I've saved it").click().run()
        assert at.title[0].value == "Welcome, frank"
        at = start()
        field(at, "signup_username").set_value("grace")
        field(at, "signup_password").set_value(PASSWORD)
        field(at, "signup_confirm").set_value(PASSWORD)
        button(at, "Create account").click().run()
        assert at.error[0].value == "Signup limit test"
        assert store.verify_user("grace", PASSWORD) is None

# --- Staying logged in ---
token = store.create_session(alice)
assert store.user_from_session(token) == (alice, "alice")
assert store.user_from_session("made-up-token") is None
with store.engine().connect() as db:
    saved = db.execute(select(store.sessions.c.token_hash)).scalars().all()
assert token not in saved, "login tokens must be stored hashed"
store.end_session(token)
assert store.user_from_session(token) is None, "logging out must end the session"
token = store.create_session(alice)
with store.engine().begin() as db:
    db.execute(update(store.sessions).values(expires=time.time() - 1))
assert store.user_from_session(token) is None, "expired sessions must stop working"

# --- Password recovery ---
ivan, _ = store.create_user("ivan", PASSWORD)
session = store.create_session(ivan)
code = store.new_recovery_code(ivan)
assert store.reset_password("ivan", "wrong-code", "new-password-1") is not None
assert store.reset_password("nobody", code, "new-password-1") is not None
assert store.reset_password("alice", code, "new-password-1") is not None, "codes belong to one user"
assert store.reset_password("ivan", code, "short") is not None
assert store.verify_user("ivan", PASSWORD), "failed attempts must not change the password"
assert store.reset_password("IVAN", code, "new-password-1") is None
assert store.verify_user("ivan", PASSWORD) is None and store.verify_user("ivan", "new-password-1")
assert store.user_from_session(session) is None, "a reset must log out other sessions"
assert store.reset_password("ivan", code, "new-password-2") is not None, "codes only work once"
old_code = store.new_recovery_code(ivan)
newer_code = store.new_recovery_code(ivan)
assert store.reset_password("ivan", old_code, "new-password-3") is not None, "a new code replaces the old"
assert store.reset_password("ivan", newer_code, "new-password-3") is None

# --- Export and deletion ---
jo, _ = store.create_user("joan", PASSWORD)
store.save_setup(jo, {"ci_name": "JoBot"}, ["jo.pdf"])
jo_chat = store.create_conversation(jo, "general", "Jo chat")
store.add_message(jo, jo_chat, "user", "jo private")
store.record_usage("message", jo)
store.save_documents(jo, b"jo documents")
store.create_session(jo)
store.new_recovery_code(jo)
store.set_plan("joan", "paid")
exported = store.export_user_data(jo)
assert exported["username"] == "joan" and exported["chats"][0]["messages"][0]["content"] == "jo private"
assert "password_hash" not in json.dumps(exported) and "salt" not in exported
assert "jo private" not in json.dumps(store.export_user_data(alice)), "exports hold one user's data"
assert store.delete_account(jo, "wrong-password") is False
assert store.verify_user("joan", PASSWORD), "a wrong password must not delete anything"
assert store.delete_account(jo, PASSWORD) is True
assert store.verify_user("joan", PASSWORD) is None and store.load_documents(jo) is None
with store.engine().connect() as db:
    for table in (store.conversations, store.sessions, store.recovery_codes, store.plans, store.usage):
        left = db.execute(select(func.count()).select_from(table).where(table.c.user_id == jo)).scalar()
        assert left == 0, f"{table.name} kept data after deletion"
    assert not db.execute(select(store.messages).where(store.messages.c.conversation_id == jo_chat)).all()
assert store.verify_user("alice", PASSWORD), "deleting one account must not touch another"

with patch("langchain_groq.ChatGroq", FakeChatGroq):
    # Signing up shows a recovery code once, and it really works
    at = start()
    field(at, "signup_username").set_value("kay")
    field(at, "signup_password").set_value(PASSWORD)
    field(at, "signup_confirm").set_value(PASSWORD)
    button(at, "Create account").click().run()
    assert at.warning and at.code, "the recovery code must be shown"
    shown_code = at.code[0].value
    button(at, "I've saved it").click().run()
    assert not at.code and at.title[0].value == "Welcome, kay"
    at = start()
    field(at, "reset_username").set_value("kay")
    field(at, "reset_code").set_value(shown_code)
    field(at, "reset_password").set_value("kays-new-password")
    button(at, "Reset password").click().run()
    assert at.success and not at.error
    assert store.verify_user("kay", "kays-new-password")

# --- Plans ---
with patch("langchain_groq.ChatGroq", FakeChatGroq):
    hana, _ = store.create_user("hana", PASSWORD)
    assert store.get_plan(hana) == "free", "new users start on Free"
    assert store.set_plan("nobody", "paid") is False and store.set_plan("hana", "gold") is False
    paid_only = "qwen/qwen3.8-27b"

    # Free users aren't offered Paid models
    at = login("hana")
    button(at, "Open General chat").click().run()
    assert "Qwen 3.8" not in at.selectbox[0].options
    assert len(at.selectbox[0].options) == 2

    # Paid users can choose it
    assert store.set_plan("HANA", "paid")
    assert store.get_plan(hana) == "paid"
    at = login("hana")
    button(at, "Open General chat").click().run()
    at.selectbox[0].select("Qwen 3.8").run()
    send(at, "use the paid model")
    assert FakeChatGroq.models[-1] == paid_only

    # Moving back to Free applies at the next message, even in an open session
    assert store.set_plan("hana", "free")
    send(at, "after downgrade")
    assert FakeChatGroq.models[-1] != paid_only, "a downgrade must apply straight away"

    # Per-day limits follow the plan
    with patch.dict(store.PLANS["free"], {"messages_per_day": 1}):
        assert "limit of 1" in store.limit_message("message", hana, plan="free")
        assert store.limit_message("message", hana, plan="paid") is None


# --- Tool calling ---
def tool_call(name, query, call_id="call_1"):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": json.dumps({"query": query}), "id": call_id, "index": 0}],
    )


class ScriptedModel:
    """Plays back one response per model call. An exception in the script is raised."""

    def __init__(self, *script):
        self.script = list(script)
        self.tools_offered = []
        self.next_has_tools = False

    def bind_tools(self, tools):
        self.next_has_tools = True
        return self

    def stream(self, chat):
        self.tools_offered.append(self.next_has_tools)
        self.next_has_tools = False
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        yield from step


def turn(llm, tools, limit=2):
    chat = [HumanMessage(content="question")]
    return agent.run_turn(llm, chat, tools, limit, lambda stream: "".join(stream)), chat


def tool_results(chat):
    return [m.content for m in chat if isinstance(m, ToolMessage)]


web_hits = [{"title": "Result [title]", "href": "https://example.com/page", "body": "A useful fact"}]
with patch.object(agent, "DDGS") as ddgs:
    search = ddgs.return_value.text
    search.return_value = web_hits
    tools, sources = agent.make_tools()
    assert [t.name for t in tools] == ["web_search"]

    # A tool result goes back to the model, and the source is kept for the user
    llm = ScriptedModel([tool_call("web_search", "python")], [AIMessageChunk(content="The answer")])
    reply, chat = turn(llm, tools)
    assert reply == "The answer" and "A useful fact" in tool_results(chat)[0]
    assert sources == ["Result title (example.com)"], sources

    # The plan's limit stops further tool calls, and the model still answers
    search.reset_mock()
    llm = ScriptedModel([tool_call("web_search", "a")], [tool_call("web_search", "b")], [AIMessageChunk(content="Done")])
    reply, chat = turn(llm, tools, limit=2)
    assert reply == "Done" and llm.tools_offered == [True, True, False], llm.tools_offered
    assert any("no more tool calls" in str(m.content) for m in chat), "the model is told to finish"
    assert search.call_count == 2

    # Several calls in one response can't get past the limit
    search.reset_mock()
    many = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "web_search", "args": json.dumps({"query": str(i)}), "id": f"c{i}", "index": i} for i in range(5)
        ],
    )
    llm = ScriptedModel([many], [AIMessageChunk(content="Done")])
    reply, chat = turn(llm, tools, limit=2)
    assert search.call_count == 2 and reply == "Done"
    assert len(tool_results(chat)) == 5, "every call needs an answer"

    # A failing tool is reported to the model, without details
    search.side_effect = RuntimeError("secret-search-detail")
    llm = ScriptedModel([tool_call("web_search", "x")], [AIMessageChunk(content="Sorry")])
    reply, chat = turn(llm, tools)
    assert reply == "Sorry" and "secret-search-detail" not in tool_results(chat)[0]

# A made-up tool name doesn't crash the reply
llm = ScriptedModel([tool_call("delete_everything", "x")], [AIMessageChunk(content="Fine")])
reply, chat = turn(llm, tools)
assert reply == "Fine" and "no tool called" in tool_results(chat)[0]

# A bad tool request falls back to answering without tools
llm = ScriptedModel(RuntimeError("tool_use_failed"), [AIMessageChunk(content="No tools needed")])
assert turn(llm, tools)[0] == "No tools needed" and llm.tools_offered == [True, False]

# A failure with no tools in play still reaches the user's error message
try:
    turn(ScriptedModel(RuntimeError("down")), [])
    raise AssertionError("the failure was swallowed")
except RuntimeError:
    pass

# Document search only appears when documents exist, and its sources are kept
tools, sources = agent.make_tools(lambda query: ("Refund text", ["policy.pdf (page 1)"]))
assert [t.name for t in tools] == ["web_search", "search_documents"]
assert tools[1].invoke({"query": "refund"}) == "Refund text" and sources == ["policy.pdf (page 1)"]

# Upload limits count per user, and old usage gets cleaned up and stops counting
with patch.dict(store.LIMITS, {"upload": [(True, 1, store.DAY, "Upload limit test")]}):
    assert store.limit_message("upload", alice) is None
    store.record_usage("upload", alice)
    assert store.limit_message("upload", alice) == "Upload limit test"
    assert store.limit_message("upload", bob) is None
    with store.engine().begin() as db:
        db.execute(update(store.usage).values(at=time.time() - 3 * store.DAY))
    assert store.limit_message("upload", alice) is None
    store.record_usage("upload", bob)
    with store.engine().connect() as db:
        left = db.execute(select(func.count()).select_from(store.usage)).scalar()
    assert left == 1, "usage older than two days must be removed"

# --- Uploaded PDFs become searchable, survive saving, and cite the original file name ---
PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 10 50 Td (Refund policy) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R>>
startxref
0
%%EOF
"""
embeddings = DeterministicFakeEmbedding(size=8)
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "0.pdf")
    with open(path, "wb") as f:
        f.write(PDF)
    docs = read_pdf(path, source="policy.pdf")
index = build_index(docs, embeddings)
found = index.similarity_search("refund", k=1)[0]
assert "Refund policy" in found.page_content and found.metadata["source"] == "policy.pdf"
assert build_index([], embeddings) is None, "no readable text is reported, not a crash"

# Saved indexes come back intact, belong to one user, and replace the old one
assert store.load_documents(alice) is None
store.save_documents(alice, b"old")
store.save_documents(alice, index.serialize_to_bytes())
assert store.load_documents(bob) is None
restored = FAISS.deserialize_from_bytes(
    store.load_documents(alice), embeddings, allow_dangerous_deserialization=True
)
assert "Refund policy" in restored.similarity_search("refund", k=1)[0].page_content

# Deleting a chat also deletes its messages
chat = store.create_conversation(alice, "general", "Temporary")
store.add_message(alice, chat, "user", "soon gone")
store.delete_conversation(alice, chat)
with store.engine().connect() as db:
    left = db.execute(select(store.messages).where(store.messages.c.conversation_id == chat)).all()
assert left == [], "messages must not outlive their chat"


# --- The status check reports what Groq says ---
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def groq_lists(*models):
    body = json.dumps({"data": [{"id": model} for model in models]}).encode()
    return patch("urllib.request.urlopen", return_value=FakeResponse(body))


with groq_lists("model-a", "model-b"):
    assert model_status("model-a", "key") == (True, "Online")
with groq_lists("model-b"):
    assert model_status("model-a", "key")[0] is False
with patch("urllib.request.urlopen", side_effect=OSError("no network")):
    online, message = model_status("model-a", "key")
    assert online is False and "no network" not in message

# Every configured service answers
status = store.service_status()
assert status and all(status.values()), status
if real_services:
    assert set(status) == {"Database", "Document storage", "Functions"}, status
    assert store.engine().dialect.name == "postgresql"
else:
    assert set(status) == {"Database"} and store.engine().dialect.name == "sqlite"

shutil.rmtree(data_folder, ignore_errors=True)
print("ok")
