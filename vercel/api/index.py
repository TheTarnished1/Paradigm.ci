"""Paradigm's API. Vercel sends every /api request here."""
import io
import json
import logging
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

import _chat
import _store as store

COOKIE = "paradigm_session"
MAX_PDFS = 5
MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # Vercel rejects request bodies over about 4.5 MB
CHUNK_CHARS = 1000
HISTORY_MESSAGES = 10
DEFAULT_IDENTITY = {
    "ci_name": "Paradigm",
    "business_name": "",
    "base_personality": "Kind and friendly.",
    "core_directive": "Give useful, accurate information.",
}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def same_site_only(request: Request, call_next):
    # The login cookie is SameSite=Lax. This also refuses changes from other sites as a second layer.
    if request.method not in ("GET", "HEAD"):
        origin = request.headers.get("origin")
        # Proxies change Host and put the original in X-Forwarded-Host. Browsers can't set that from another site.
        hosts = {request.headers.get("host"), request.headers.get("x-forwarded-host")}
        if origin and urlparse(origin).netloc not in hosts:
            return JSONResponse({"detail": "This request came from another site."}, status_code=403)
    return await call_next(request)


class Credentials(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=500)


class Recovery(BaseModel):
    username: str = Field(max_length=200)
    code: str = Field(max_length=200)
    password: str = Field(max_length=500)


class Password(BaseModel):
    password: str = Field(max_length=500)


class Setup(BaseModel):
    ci_name: str = Field(min_length=1, max_length=40)
    business_name: str = Field(default="", max_length=80)
    base_personality: str = Field(default="", max_length=300)
    core_directive: str = Field(default="", max_length=1000)


class PastMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    mode: str = Field(pattern="^(assistant|general)$")
    conversation_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=60)
    # Editing or regenerating deletes every saved message after the first `keep`, then answers again
    keep: int | None = Field(default=None, ge=0, le=10000)
    research: bool = False
    # Temporary chats save nothing, so the browser sends the earlier messages itself
    temporary: bool = False
    history: list[PastMessage] = Field(default=[], max_length=HISTORY_MESSAGES)


class ChatUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    pinned: bool | None = None


def start_session(response, request, user_id):
    token = store.create_session(user_id)
    response.set_cookie(
        COOKIE,
        token,
        max_age=store.SESSION_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        path="/",
    )


def current_user(request: Request):
    """Returns (user id, username), or refuses the request."""
    token = request.cookies.get(COOKIE)
    user = store.user_from_session(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Log in to continue.")
    return user


def too_many(message):
    return HTTPException(status_code=429, detail=message)


@app.post("/api/signup")
def signup(body: Credentials, request: Request, response: Response):
    # Only successful sign-ups count, so bad attempts can't block real people
    blocked = store.limit_message("signup")
    if blocked:
        raise too_many(blocked)
    username = body.username.strip()
    user_id, problem = store.create_user(username, body.password)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    store.record_usage("signup")
    start_session(response, request, user_id)
    return {"username": username, "recovery_code": store.new_recovery_code(user_id)}


@app.post("/api/login")
def login(body: Credentials, request: Request, response: Response):
    user = store.verify_user(body.username.strip(), body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Wrong username or password. After 5 failed tries, wait 5 minutes.")
    start_session(response, request, user[0])
    return {"username": user[1]}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE)
    if token:
        store.end_session(token)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.post("/api/recover")
def recover(body: Recovery):
    blocked = store.limit_message("recovery")
    if blocked:
        raise too_many(blocked)
    store.record_usage("recovery")
    problem = store.reset_password(body.username.strip(), body.code.strip(), body.password)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(current_user)):
    identity, documents = store.load_setup(user[0])
    plan = store.PLANS[store.get_plan(user[0])]
    return {
        "username": user[1],
        "setup": identity or DEFAULT_IDENTITY,
        "documents": documents,
        "plan": {
            "label": plan["label"],
            "messages_per_day": plan["messages_per_day"],
            "tool_calls": plan["tool_calls"],
        },
        "models": [{"id": model, "label": store.MODELS[model]} for model in plan["models"]],
        "more_models": len(plan["models"]) < len(store.MODELS),
    }


@app.post("/api/recovery-code")
def recovery_code(user=Depends(current_user)):
    return {"recovery_code": store.new_recovery_code(user[0])}


@app.put("/api/setup")
def save_setup(body: Setup, user=Depends(current_user)):
    store.save_identity(user[0], body.model_dump())
    return {"ok": True}


def pdf_chunks(name, data):
    """Splits a PDF's text into (source, page, text) pieces."""
    chunks = []
    for number, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        # NUL characters can't be stored in Postgres text
        text = text.replace("\x00", "")
        for start in range(0, len(text), CHUNK_CHARS - 100):
            piece = text[start : start + CHUNK_CHARS]
            if piece.strip():
                chunks.append((name, number, piece))
    return chunks


@app.post("/api/documents")
def upload_documents(files: list[UploadFile] = File(), user=Depends(current_user)):
    if len(files) > MAX_PDFS:
        raise HTTPException(status_code=400, detail=f"Upload {MAX_PDFS} PDFs or fewer.")
    blocked = store.limit_message("upload", user[0])
    if blocked:
        raise too_many(blocked)
    store.record_usage("upload", user[0])

    names, chunks, total = [], [], 0
    for file in files:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Those files are too large. The limit is 4 MB in total.")
        name = os.path.basename(file.filename or "document.pdf")[:120]
        try:
            chunks.extend(pdf_chunks(name, data))
        except Exception:
            logging.warning("Could not read an uploaded PDF")
            raise HTTPException(status_code=400, detail=f"Couldn't read {name}. Check that it's a PDF.")
        names.append(name)
    if not chunks:
        raise HTTPException(status_code=400, detail="Couldn't find readable text in those PDFs. Scanned pages won't work yet.")
    store.replace_documents(user[0], names, chunks)
    return {"documents": names}


@app.get("/api/conversations")
def conversations(mode: str, user=Depends(current_user)):
    return store.list_conversations(user[0], mode)


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str, user=Depends(current_user)):
    return store.load_messages(user[0], conversation_id)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user=Depends(current_user)):
    store.delete_conversation(user[0], conversation_id)
    return {"ok": True}


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(conversation_id: str, body: ChatUpdate, user=Depends(current_user)):
    title = " ".join(body.title.split()) if body.title else None
    if not store.update_conversation(user[0], conversation_id, title=title, pinned=body.pinned):
        raise HTTPException(status_code=404, detail="That chat doesn't exist.")
    return {"ok": True}


@app.post("/api/conversations/{conversation_id}/share")
def share_conversation(conversation_id: str, user=Depends(current_user)):
    blocked = store.limit_message("share", user[0])
    if blocked:
        raise too_many(blocked)
    token = store.share_conversation(user[0], conversation_id)
    if token is None:
        raise HTTPException(status_code=404, detail="That chat doesn't exist.")
    store.record_usage("share", user[0])
    return {"token": token}


@app.delete("/api/conversations/{conversation_id}/share")
def unshare_conversation(conversation_id: str, user=Depends(current_user)):
    store.unshare_conversation(user[0], conversation_id)
    return {"ok": True}


@app.get("/api/shared/{token}")
def shared(token: str):
    """Public: anyone with the link can read the snapshot."""
    snapshot = store.get_share(token[:64])
    if snapshot is None:
        raise HTTPException(status_code=404, detail="This link doesn't work. It may have been turned off.")
    return snapshot


@app.get("/api/memories")
def memories(user=Depends(current_user)):
    return store.list_memories(user[0])


@app.delete("/api/memories/{memory_id}")
def forget(memory_id: int, user=Depends(current_user)):
    store.delete_memory(user[0], memory_id)
    return {"ok": True}


@app.delete("/api/memories")
def forget_all(user=Depends(current_user)):
    store.delete_memory(user[0])
    return {"ok": True}


def reply_stream(user_id, body, plan_name, model, conversation_id):
    """Streams newline-separated JSON events to the browser, and saves the reply when it's done."""

    def event(kind, **fields):
        return json.dumps({"type": kind, **fields}) + "\n"

    if conversation_id:
        yield event("conversation", id=conversation_id)
    reply = ""
    try:
        if body.temporary:
            history, saved_memories, remember = body.history, [], None
        else:
            history = [
                PastMessage(role=m["role"], content=m["content"])
                for m in store.load_messages(user_id, conversation_id)
                if not m["error"]
            ][-HISTORY_MESSAGES:]
            store.add_message(user_id, conversation_id, "user", body.message)
            saved_memories = [memory["content"] for memory in store.list_memories(user_id)]
            remember = lambda fact: store.add_memory(user_id, fact)
        identity, documents = store.load_setup(user_id)
        has_documents = body.mode == "assistant" and bool(documents)
        context, sources = store.search_documents(user_id, body.message) if has_documents else ("", [])
        prompt = _chat.system_prompt(body.mode, identity or DEFAULT_IDENTITY, context, saved_memories, body.research)
        chat = [
            {"role": "system", "content": prompt},
            *({"role": past.role, "content": past.content} for past in history),
            {"role": "user", "content": body.message},
        ]
        search = (lambda query: store.search_documents(user_id, query)) if has_documents else None
        plan = store.PLANS[plan_name]
        for kind, value in _chat.run_turn(
            _chat.client(),
            model,
            chat,
            plan["research_tool_calls" if body.research else "tool_calls"],
            search,
            remember,
            before_model_call=lambda: store.record_usage("model_call"),
        ):
            if kind == "text":
                reply += value
                yield event("text", text=value)
            elif kind == "status":
                yield event("status", text=value)
            else:
                sources += [source for source in value if source not in sources]
        reply, followups = _chat.split_followups(_chat.clean_reply(reply))
        if conversation_id:
            store.add_message(user_id, conversation_id, "assistant", reply, sources=sources)
        yield event("done", sources=sources, followups=followups)
    except Exception as error:
        # Only the type is logged, because details can include user content
        logging.error("Chat request failed: %s", type(error).__name__)
        message = "Sorry, I couldn't answer that. Try again."
        try:
            if conversation_id:
                store.add_message(user_id, conversation_id, "assistant", message, error=True)
        finally:
            yield event("error", text=message)


@app.post("/api/chat")
def chat(body: ChatRequest, user=Depends(current_user)):
    plan_name = store.get_plan(user[0])
    blocked = (
        store.limit_message("message", user[0], plan=plan_name)
        or (body.research and store.limit_message("research", user[0]))
        or store.limit_message("model_call")
    )
    if blocked:
        raise too_many(blocked)
    allowed = store.PLANS[plan_name]["models"]
    if body.model and body.model not in allowed:
        raise HTTPException(status_code=403, detail="That model isn't on your plan.")
    conversation_id = None if body.temporary else body.conversation_id
    if conversation_id and not store.conversation_exists(user[0], conversation_id):
        raise HTTPException(status_code=404, detail="That chat doesn't exist.")
    store.record_usage("message", user[0])
    if body.research:
        store.record_usage("research", user[0])
    if conversation_id and body.keep is not None:
        store.truncate_messages(user[0], conversation_id, body.keep)
    elif not conversation_id and not body.temporary:
        conversation_id = store.create_conversation(user[0], body.mode, body.message[:60])
    return StreamingResponse(
        reply_stream(user[0], body, plan_name, body.model or allowed[0], conversation_id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status")
def status(user=Depends(current_user)):
    """Checks that the database and the AI service are working."""
    checks = {}
    try:
        store.get_plan(user[0])
        checks["Database"] = True
    except Exception:
        checks["Database"] = False
    try:
        available = {model.id for model in _chat.client().models.list().data}
        allowed = store.PLANS[store.get_plan(user[0])]["models"]
        checks["AI service"] = all(model in available for model in allowed)
    except Exception:
        checks["AI service"] = False
    return checks


@app.get("/api/export")
def export(user=Depends(current_user)):
    return JSONResponse(
        store.export_user_data(user[0]),
        headers={"Content-Disposition": 'attachment; filename="paradigm-data.json"'},
    )


@app.post("/api/delete-account")
def delete_account(body: Password, response: Response, user=Depends(current_user)):
    if not store.delete_account(user[0], body.password):
        raise HTTPException(status_code=403, detail="That password isn't right.")
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}
