"""Lets the model call tools during a reply, up to the limit set by the user's plan."""
import logging
import re
from urllib.parse import urlparse

from ddgs import DDGS
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool

MAX_RESULT_CHARS = 3000
WEB_RESULTS = 4
FINISH_NOTE = (
    "You have no more tool calls for this reply. Answer now from the tool results above. "
    "Don't say you can't search, and don't mention tool names."
)


def search_web(query):
    """Returns (text for the model, sources to show the user)."""
    results = DDGS(timeout=10).text(query[:200], max_results=WEB_RESULTS)
    lines, sources = [], []
    for result in results:
        title = re.sub(r"[\[\]()`*_]", "", result.get("title", ""))[:60]
        site = urlparse(result.get("href", "")).netloc
        lines.append(f"{title} ({site}): {result.get('body', '')}")
        sources.append(f"{title} ({site})")
    return "\n".join(lines)[:MAX_RESULT_CHARS] or "No results found.", sources


def make_tools(search_documents=None):
    """Returns (tools for the model, a list that collects sources as tools run).

    search_documents takes a query and returns (text, sources). Leave it out when there are no documents.
    """
    sources = []

    def keep(found):
        sources.extend(source for source in found if source not in sources)

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information or facts you aren't sure about."""
        text, found = search_web(query)
        keep(found)
        return text

    tools = [web_search]

    if search_documents:

        @tool(name_or_callable="search_documents")
        def search_user_documents(query: str) -> str:
            """Search the user's uploaded documents for more detail."""
            text, found = search_documents(query)
            keep(found)
            return text[:MAX_RESULT_CHARS] or "Nothing relevant in the documents."

        tools.append(search_user_documents)

    return tools, sources


def _run_tool(call, tools_by_name):
    # Tool errors go back to the model as text, so it can still answer
    try:
        return str(tools_by_name[call["name"]].invoke(call["args"]))
    except KeyError:
        return f"There is no tool called {call['name']}."
    except Exception as error:
        logging.warning("Tool %s failed: %s", call["name"], type(error).__name__)
        return "That tool isn't working right now. Answer without it, and say so if that matters."


def run_turn(llm, chat, tools, max_tool_calls, write, before_model_call=lambda: None, on_tool=lambda name: None):
    """Runs a reply and returns its text. write shows a stream of text and returns what it showed."""
    tools_by_name = {t.name: t for t in tools}
    calls_used = 0
    reply = []

    while True:
        use_tools = bool(tools_by_name) and calls_used < max_tool_calls
        model = llm.bind_tools(list(tools_by_name.values())) if use_tools else llm
        chunks = []
        if not use_tools and calls_used:
            chat.append(SystemMessage(content=FINISH_NOTE))

        def stream():
            for chunk in model.stream(chat):
                chunks.append(chunk)
                yield chunk.content or ""

        before_model_call()
        try:
            text = write(stream())
        except Exception:
            # Models sometimes produce a bad tool call, so answer without tools before giving up
            if not use_tools or chunks:
                raise
            logging.warning("Tool request failed, so retrying without tools")
            tools_by_name = {}
            continue

        if text:
            reply.append(text)
        message = sum(chunks[1:], chunks[0]) if chunks else None
        if not use_tools or message is None or not message.tool_calls:
            return "\n\n".join(reply)

        chat.append(message)
        for call in message.tool_calls:
            if calls_used < max_tool_calls:
                calls_used += 1
                on_tool(call["name"])
                result = _run_tool(call, tools_by_name)
            else:
                result = "Tool limit reached. Answer with what you have."
            chat.append(ToolMessage(content=result, tool_call_id=call["id"]))
