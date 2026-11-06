# Run with: DATABASE_URL=<an empty, throwaway Postgres database> python api/test_api.py
# It creates test accounts, so never point it at production. No Groq or web search calls happen.
import io
import json
import os
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
os.environ["GROQ_API_KEY"] = "test-key-not-real"
assert "production" not in os.environ.get("NEON_BRANCH", ""), "never run these tests on production"

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlalchemy import delete, func, select, update

import _chat
import _store as store
import index

store.init_db()
PASSWORD = "test-password-123"
BASE = "https://testserver"


def browser():
    return TestClient(index.app, base_url=BASE)


def sign_up(name):
    client = browser()
    response = client.post("/api/signup", json={"username": name, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return client


def word_chunk(text):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=None))])


def tool_chunk(name, query, call_id="c1", index_=0):
    function = NS(name=name, arguments=json.dumps({"query": query}))
    return NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=index_, id=call_id, function=function)]))])


class FakeGroq:
    """Plays back one response per model call. An exception in the script is raised."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **options):
        self.calls.append({**options, "messages": list(options["messages"])})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return iter(step)


def events(response):
    return [json.loads(line) for line in response.text.splitlines()]


def send(client, message, groq=None, mode="general", **extra):
    groq = groq or FakeGroq([word_chunk("Hello "), word_chunk("there")])
    with patch.object(_chat, "client", lambda: groq):
        return client.post("/api/chat", json={"message": message, "mode": mode, **extra}), groq


# --- Accounts ---
alice = sign_up("alice")
bob = sign_up("bob")
assert alice.cookies.get(index.COOKIE) != bob.cookies.get(index.COOKIE)
assert browser().post("/api/signup", json={"username": "ALICE", "password": PASSWORD}).status_code == 400
assert browser().post("/api/signup", json={"username": "x", "password": PASSWORD}).status_code == 400
assert browser().post("/api/signup", json={"username": "shortpw", "password": "1234567"}).status_code == 400
assert browser().get("/api/me").status_code == 401, "nothing is reachable before logging in"
assert alice.get("/api/me").json()["username"] == "alice"

response = browser().post("/api/signup", json={"username": "cookiejar", "password": PASSWORD})
cookie = response.headers["set-cookie"].lower()
assert "httponly" in cookie and "samesite=lax" in cookie and "secure" in cookie, cookie
assert response.json()["recovery_code"]

with store.engine().connect() as db:
    stored_hash = db.execute(select(store.users.c.password_hash).limit(1)).scalar_one()
    stored_tokens = db.execute(select(store.sessions.c.token_hash)).scalars().all()
assert PASSWORD.encode() not in stored_hash, "passwords are never stored as text"
assert alice.cookies.get(index.COOKIE) not in stored_tokens, "login tokens are stored hashed"

wrong = browser().post("/api/login", json={"username": "alice", "password": "wrong-password"})
assert wrong.status_code == 401
unknown = browser().post("/api/login", json={"username": "nobody", "password": PASSWORD})
assert unknown.json()["detail"] == wrong.json()["detail"], "responses mustn't reveal which usernames exist"

# Repeated failures lock the account, even for the right password
locked = sign_up("carol")
for _ in range(store.MAX_FAILED_LOGINS):
    browser().post("/api/login", json={"username": "carol", "password": "wrong-password"})
assert browser().post("/api/login", json={"username": "carol", "password": PASSWORD}).status_code == 401
with store.engine().begin() as db:
    db.execute(update(store.users).where(store.users.c.username == "carol").values(locked_until=0))
assert browser().post("/api/login", json={"username": "carol", "password": PASSWORD}).status_code == 200

# Logging out ends the session on the server, even if the cookie is kept
old_cookie = locked.cookies.get(index.COOKIE)
locked.post("/api/logout")
replay = browser()
replay.cookies.set(index.COOKIE, old_cookie)
assert replay.get("/api/me").status_code == 401

# Expired sessions stop working
expiring = sign_up("erin")
with store.engine().begin() as db:
    user_id = db.execute(select(store.users.c.id).where(store.users.c.username == "erin")).scalar_one()
    db.execute(update(store.sessions).where(store.sessions.c.user_id == user_id).values(expires=time.time() - 1))
assert expiring.get("/api/me").status_code == 401

# Requests from other sites are refused
foreign = alice.post("/api/logout", headers={"Origin": "https://evil.example"})
assert foreign.status_code == 403
assert alice.get("/api/me").status_code == 200
# The same site through a proxy still works
proxied = browser().post(
    "/api/login",
    json={"username": "bob", "password": PASSWORD},
    headers={"Origin": "https://paradigm.example", "X-Forwarded-Host": "paradigm.example"},
)
assert proxied.status_code == 200

# --- Password recovery ---
ivan = browser()
code = ivan.post("/api/signup", json={"username": "ivan", "password": PASSWORD}).json()["recovery_code"]
recover = lambda name, value, password: browser().post(
    "/api/recover", json={"username": name, "code": value, "password": password}
)
assert recover("ivan", "wrong-code", "new-password-1").status_code == 400
assert recover("alice", code, "new-password-1").status_code == 400, "codes belong to one user"
assert recover("ivan", code, "short").status_code == 400
assert recover("IVAN", code, "new-password-1").status_code == 200
assert browser().post("/api/login", json={"username": "ivan", "password": PASSWORD}).status_code == 401
assert browser().post("/api/login", json={"username": "ivan", "password": "new-password-1"}).status_code == 200
assert ivan.get("/api/me").status_code == 401, "a reset logs out other sessions"
assert recover("ivan", code, "new-password-2").status_code == 400, "codes only work once"

# --- Chats ---
response, groq = send(alice, "What is torque?")
stream = events(response)
assert [e["type"] for e in stream] == ["conversation", "text", "text", "done"], stream
chat_id = stream[0]["id"]
listed = alice.get("/api/conversations", params={"mode": "general"}).json()
assert [(c["id"], c["title"]) for c in listed] == [(chat_id, "What is torque?")] and listed[0]["updated"] > 0
assert [m["content"] for m in alice.get(f"/api/conversations/{chat_id}").json()] == ["What is torque?", "Hello there"]

# Earlier messages go to the model in order, with the current message once
response, groq = send(alice, "Second question", conversation_id=chat_id)
sent = [m["content"] for m in groq.calls[0]["messages"][1:]]
assert sent == ["What is torque?", "Hello there", "Second question"], sent

# One user can't read, add to or delete another user's chat
assert bob.get(f"/api/conversations/{chat_id}").json() == []
assert send(bob, "intruder", conversation_id=chat_id)[0].status_code == 404
bob.delete(f"/api/conversations/{chat_id}")
assert len(alice.get(f"/api/conversations/{chat_id}").json()) == 4
assert bob.get("/api/conversations", params={"mode": "general"}).json() == []

# Failures show a friendly message, hide details, and stay out of later requests
response, _ = send(alice, "This fails", FakeGroq(RuntimeError("secret-upstream-detail")), conversation_id=chat_id)
assert events(response)[-1]["type"] == "error" and "secret-upstream-detail" not in response.text
response, groq = send(alice, "Back again", conversation_id=chat_id)
assert "Sorry" not in json.dumps(groq.calls[0]["messages"]), "failed replies aren't sent back to the model"

# Deleting a chat removes its messages
alice.delete(f"/api/conversations/{chat_id}")
with store.engine().connect() as db:
    assert not db.execute(select(store.messages).where(store.messages.c.conversation_id == chat_id)).all()

# --- Modes and setup ---
setup = {"ci_name": "LogiBot", "business_name": "LogiCorp", "base_personality": "Calm", "core_directive": "Ship things"}
assert alice.put("/api/setup", json=setup).status_code == 200
assert alice.put("/api/setup", json={**setup, "ci_name": ""}).status_code == 422, "the name is required"
assert alice.get("/api/me").json()["setup"]["ci_name"] == "LogiBot"
_, groq = send(alice, "hi", mode="assistant")
system = groq.calls[0]["messages"][0]["content"]
for expected in ["LogiBot", "LogiCorp", "Calm", "Ship things"]:
    assert expected in system, system
_, groq = send(alice, "hi", mode="general")
assert "LogiBot" not in groq.calls[0]["messages"][0]["content"], "General chat ignores the setup"
assert "Paradigm" in send(bob, "hi", mode="assistant")[1].calls[0]["messages"][0]["content"]

# --- Plans ---
dana = sign_up("dana")
free = dana.get("/api/me").json()
assert free["plan"]["label"] == "Free" and free["more_models"] is True
paid_only = "qwen/qwen3.8-27b"
assert paid_only not in [m["id"] for m in free["models"]]
assert send(dana, "hi", model=paid_only)[0].status_code == 403, "the server refuses models not on the plan"
assert not store.set_plan("nobody", "paid") and not store.set_plan("dana", "gold")
assert store.set_plan("DANA", "paid")
assert paid_only in [m["id"] for m in dana.get("/api/me").json()["models"]]
_, groq = send(dana, "hi", model=paid_only)
assert groq.calls[0]["model"] == paid_only
assert store.set_plan("dana", "free")
assert send(dana, "hi", model=paid_only)[0].status_code == 403, "a downgrade applies straight away"

# --- Limits ---
frank = sign_up("frank")
with patch.dict(store.PLANS["free"], {"messages_per_minute": 2}):
    assert send(frank, "one")[0].status_code == 200
    assert send(frank, "two")[0].status_code == 200
    response, groq = send(frank, "three")
    assert response.status_code == 429 and "quickly" in response.json()["detail"]
    assert groq.calls == [], "a blocked message never reaches Groq"
    assert send(bob, "bob is fine")[0].status_code == 200, "one user's limit doesn't affect another"
with patch.dict(store.LIMITS, {"model_call": [(False, 1, store.MINUTE, "Everyone limit")]}):
    assert send(bob, "blocked")[0].json()["detail"] == "Everyone limit", "shared limits apply across users"
with store.engine().begin() as db:
    db.execute(delete(store.usage).where(store.usage.c.kind == "signup"))
with patch.dict(store.LIMITS, {"signup": [(False, 1, 3600, "Signup limit")]}):
    assert browser().post("/api/signup", json={"username": "x", "password": PASSWORD}).status_code == 400
    assert browser().post("/api/signup", json={"username": "grace", "password": PASSWORD}).status_code == 200
    blocked = browser().post("/api/signup", json={"username": "hank", "password": PASSWORD})
    assert blocked.status_code == 429 and blocked.json()["detail"] == "Signup limit"

# Old usage stops counting and gets cleaned up
with store.engine().begin() as db:
    db.execute(update(store.usage).values(at=time.time() - 3 * store.DAY))
store.record_usage("upload")
with store.engine().connect() as db:
    assert db.execute(select(func.count()).select_from(store.usage)).scalar() == 1


# --- Documents ---
def pdf_with(text):
    # Writes a one-page PDF holding the text
    content = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode()
    body = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 500 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"\nendstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\ntrailer<</Root 1 0 R>>\nstartxref\n0\n%%EOF\n"
    )
    return body


def blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def upload(client, *files):
    return client.post("/api/documents", files=[("files", (name, data, "application/pdf")) for name, data in files])


joan = sign_up("joan")
response = upload(joan, ("../../policy.pdf", pdf_with("Refunds are available within thirty days of purchase")))
assert response.status_code == 200 and response.json()["documents"] == ["policy.pdf"], response.text
assert joan.get("/api/me").json()["documents"] == ["policy.pdf"]
assert upload(joan, ("scan.pdf", blank_pdf())).status_code == 400, "scanned files are reported"
assert upload(joan, ("bad.pdf", b"not a pdf")).status_code == 400
assert upload(joan, ("big.pdf", b"x" * (index.MAX_UPLOAD_BYTES + 1))).status_code == 413
assert upload(joan, *[(f"{n}.pdf", b"x") for n in range(index.MAX_PDFS + 1)]).status_code == 400
assert joan.get("/api/me").json()["documents"] == ["policy.pdf"], "failed uploads keep the old documents"

# Relevant documents reach the prompt in Assistant mode and are cited, but not in General chat
_, groq = send(joan, "What is your refund policy?", mode="assistant")
assert "thirty days" in groq.calls[0]["messages"][0]["content"]
response, _ = send(joan, "What is your refund policy?", mode="assistant")
assert events(response)[-1]["sources"] == [{"label": "policy.pdf (page 1)"}]
_, groq = send(joan, "What is your refund policy?", mode="general")
assert "thirty days" not in json.dumps(groq.calls[0]["messages"])
assert store.search_documents(1, "refund") == ("", []), "one user's documents never reach another"
assert store.search_documents(1, "!!!") == ("", []), "punctuation alone is safe"
with patch.dict(store.LIMITS, {"upload": [(True, 0, store.DAY, "Upload limit")]}):
    assert upload(joan, ("policy.pdf", pdf_with("x"))).status_code == 429

# --- Tool calling ---
web_hits = [
    {"title": "Result [title]", "href": "https://example.com/page", "body": "A useful fact"},
    {"title": "Bad link", "href": "javascript:alert(1)", "body": "Unsafe"},
]


def turn(groq, limit=2, search=None):
    chat = [{"role": "user", "content": "question"}]
    results = list(_chat.run_turn(groq, "model", chat, limit, search))
    text = "".join(value for kind, value in results if kind == "text")
    return text, chat, dict(results).get("sources"), results


def tool_results(chat):
    return [m["content"] for m in chat if m["role"] == "tool"]


with patch.object(_chat, "DDGS") as ddgs:
    search = ddgs.return_value.text
    search.return_value = web_hits

    # A tool result goes back to the model, and the source is kept for the user
    text, chat, sources, results = turn(FakeGroq([tool_chunk("web_search", "python")], [word_chunk("The answer")]))
    assert text == "The answer" and "A useful fact" in tool_results(chat)[0]
    assert sources[0] == {"label": "Result title (example.com)", "url": "https://example.com/page"}, sources
    assert sources[1]["url"] == "", "only web links become clickable, never javascript: or other schemes"
    assert ("status", "Searching the web") in results

    # The plan's limit stops further tool calls, and the model still answers
    search.reset_mock()
    groq = FakeGroq([tool_chunk("web_search", "a")], [tool_chunk("web_search", "b")], [word_chunk("Done")])
    text, chat, _, _ = turn(groq, limit=2)
    assert text == "Done" and search.call_count == 2
    assert ["tools" in call for call in groq.calls] == [True, True, False]
    assert any("no more tool calls" in str(m.get("content")) for m in chat), "the model is told to finish"

    # Tool markers some models add are removed from saved replies
    assert _chat.clean_reply("Python 3.15.2 【web_search】 is out.") == "Python 3.15.2 is out."
    assert _chat.clean_reply("Plain 【brackets】 too") == "Plain too"
    response, _ = send(alice, "marker check", FakeGroq([word_chunk("Done 【web_search】")]))
    chat_id = events(response)[0]["id"]
    assert alice.get(f"/api/conversations/{chat_id}").json()[-1]["content"] == "Done"

    # Several calls in one response can't get past the limit, and every call gets an answer
    search.reset_mock()
    many = [tool_chunk("web_search", str(n), f"c{n}", n) for n in range(5)]
    text, chat, _, _ = turn(FakeGroq(many, [word_chunk("Done")]), limit=2)
    assert search.call_count == 2 and text == "Done" and len(tool_results(chat)) == 5

    # A failing tool is reported to the model, without details
    search.side_effect = RuntimeError("secret-search-detail")
    text, chat, _, _ = turn(FakeGroq([tool_chunk("web_search", "x")], [word_chunk("Sorry")]))
    assert text == "Sorry" and "secret-search-detail" not in tool_results(chat)[0]

# A made-up tool doesn't crash the reply, and tools only appear when they should
text, chat, _, _ = turn(FakeGroq([tool_chunk("delete_everything", "x")], [word_chunk("Fine")]))
assert text == "Fine" and "no tool called" in tool_results(chat)[0]
_, chat, _, _ = turn(FakeGroq([tool_chunk("search_documents", "x")], [word_chunk("Fine")]))
assert "no tool called" in tool_results(chat)[0], "document search needs documents"
text, _, sources, _ = turn(
    FakeGroq([tool_chunk("search_documents", "refund")], [word_chunk("Fine")]),
    search=lambda query: ("Refund text", [{"label": "policy.pdf (page 1)"}]),
)
assert text == "Fine" and sources == [{"label": "policy.pdf (page 1)"}]

# A bad tool request falls back to answering without tools, and real failures still surface
groq = FakeGroq(RuntimeError("tool_use_failed"), [word_chunk("No tools needed")])
assert turn(groq)[0] == "No tools needed" and ["tools" in call for call in groq.calls] == [True, False]
try:
    turn(FakeGroq(RuntimeError("down"), RuntimeError("still down")))
    raise AssertionError("the failure was swallowed")
except RuntimeError:
    pass

# --- Status ---
with patch.object(_chat, "client", lambda: NS(models=NS(list=lambda: NS(data=[NS(id=m) for m in store.MODELS])))):
    assert alice.get("/api/status").json() == {"Database": True, "AI service": True}
with patch.object(_chat, "client", lambda: NS(models=NS(list=lambda: NS(data=[])))):
    assert alice.get("/api/status").json()["AI service"] is False, "a retired model shows as offline"

# --- Export and deletion ---
kim = sign_up("kim")
send(kim, "kim private")
upload(kim, ("kim.pdf", pdf_with("Kim private document")))
exported = kim.get("/api/export").json()
assert exported["chats"][0]["messages"][0]["content"] == "kim private" and exported["documents"] == ["kim.pdf"]
assert "password" not in json.dumps(exported) and "salt" not in exported
assert "kim private" not in alice.get("/api/export").text, "exports hold one user's data"
with store.engine().connect() as db:
    kim_id = db.execute(select(store.users.c.id).where(store.users.c.username == "kim")).scalar_one()
assert kim.post("/api/delete-account", json={"password": "wrong-password"}).status_code == 403
assert kim.get("/api/me").status_code == 200, "a wrong password deletes nothing"
assert kim.post("/api/delete-account", json={"password": PASSWORD}).status_code == 200
assert browser().post("/api/login", json={"username": "kim", "password": PASSWORD}).status_code == 401
with store.engine().connect() as db:
    for table in (store.conversations, store.sessions, store.recovery_codes, store.plans, store.usage, store.doc_chunks):
        left = db.execute(select(func.count()).select_from(table).where(table.c.user_id == kim_id)).scalar()
        assert left == 0, f"{table.name} kept data after deletion"
assert alice.get("/api/me").status_code == 200, "deleting one account doesn't touch another"

print("ok")
