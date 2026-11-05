# Run with: python test_app.py  (no Groq calls or model downloads happen here)
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

os.environ["GROQ_API_KEY"] = "test-key-not-real"

import streamlit as st
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

from ci_setup_tools import build_index, read_pdf


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
    st.cache_resource.clear()
    FakeChatGroq.calls.clear()
    FakeChatGroq.fail = False
    return AppTest.from_file("Ci_app.py", default_timeout=60).run()


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def send(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception


with patch("langchain_groq.ChatGroq", FakeChatGroq):
    # Fresh start shows the welcome screen, and there are no chats in the sidebar
    at = start()
    assert not at.exception, at.exception
    assert at.title[0].value == "Hi, I'm Paradigm"
    assert not at.session_state["conversations"]

    # The first message creates a chat straight away, titled from the message
    send(at, "What is torque?")
    assert len(at.session_state["conversations"]) == 1
    assert at.session_state["conversations"][0]["title"] == "What is torque?"
    assert "Hello there" in page_text(at)
    assert not at.title

    # The current message is sent once, and earlier messages go in order
    send(at, "Second question")
    sent = [m.content for m in FakeChatGroq.calls[-1][1:]]
    assert sent == ["What is torque?", "Hello there", "Second question"], sent

    # Failures show a friendly message, hide details, and stay out of later requests
    FakeChatGroq.fail = True
    send(at, "This one fails")
    assert "couldn't answer" in page_text(at)
    assert "secret-upstream-detail" not in page_text(at)
    FakeChatGroq.fail = False
    send(at, "Back again")
    sent = [m.content for m in FakeChatGroq.calls[-1][1:]]
    assert "Sorry, I couldn't answer that. Please try again." not in sent, sent

    # New chat returns to the welcome screen and keeps the old chat
    button(at, "New chat").click().run()
    assert at.title and len(at.session_state["conversations"]) == 1

    # A suggestion sends a message
    button(at, "Help me plan my week").click().run()
    assert not at.exception, at.exception
    assert FakeChatGroq.calls[-1][-1].content == "Help me plan my week"

    # Setup values reach the prompt, and one session's setup doesn't reach another's
    custom = start()
    custom.session_state["identity"] = {
        "ci_name": "LogiBot",
        "business_name": "LogiCorp",
        "base_personality": "Calm",
        "core_directive": "Answer shipping questions",
    }
    custom.run()
    send(custom, "hi")
    system = FakeChatGroq.calls[-1][0].content
    for expected in ["LogiBot", "LogiCorp", "Calm", "Answer shipping questions"]:
        assert expected in system, system
    assert start().title[0].value == "Hi, I'm Paradigm"

# Uploaded PDFs become searchable and cite the original file name
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

# Documents with no readable text are reported instead of crashing
assert build_index([], embeddings) is None

print("ok")
