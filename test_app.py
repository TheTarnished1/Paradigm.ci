# Run with: python test_app.py  (no Groq calls or model downloads happen here)
import io
import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

data_folder = tempfile.mkdtemp()
os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["PARADIGM_DATA"] = data_folder  # keeps test accounts out of the real database

import streamlit as st
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

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
with store.connect() as db:
    stored = db.execute("SELECT password_hash FROM users WHERE id = ?", (alice,)).fetchone()[0]
assert PASSWORD.encode() not in stored

# Repeated failures lock the account, even for the right password, then it recovers
carol, _ = store.create_user("carol", PASSWORD)
for _ in range(store.MAX_FAILED_LOGINS):
    assert store.verify_user("carol", "wrong-password") is None
assert store.verify_user("carol", PASSWORD) is None
with store.connect() as db:
    db.execute("UPDATE users SET locked_until = 0 WHERE id = ?", (carol,))
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

    def __init__(self, **kwargs):
        pass

    def stream(self, chat):
        FakeChatGroq.calls.append(chat)
        if FakeChatGroq.fail:
            raise RuntimeError("secret-upstream-detail")
        yield SimpleNamespace(content="Hello ")
        yield SimpleNamespace(content="there")


def start():
    return AppTest.from_file("Ci_app.py", default_timeout=60).run()


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
    assert at.title[0].value == "⚡ Paradigm" and not at.chat_input

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
try:
    store.user_docs_dir("../../evil")
    raise AssertionError("a user id must not be able to change the folder")
except ValueError:
    pass


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

shutil.rmtree(data_folder, ignore_errors=True)
print("ok")
