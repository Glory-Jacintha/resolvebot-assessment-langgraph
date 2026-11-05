"""The conversation, built as a LangGraph graph.

                  +--> confirm ---------------------------------------------> END
    START -> entry|                                                (clear "yes" and a proposal waiting)
                  +--> start_turn -> agent --tool calls--> tools -> agent (loop)
                                       |--> finalize / too_many / fail ------> END

LangGraph pieces used:
* StateGraph + ChatState      the state of one chat (messages, waiting proposal, per-turn log)
* add_messages (reducer)      appends new messages to the history; RemoveMessage deletes old ones
* accumulate (our reducer)    lets nodes ADD to the per-turn tool log; None resets it
* MemorySaver (checkpointer)  remembers each chat's state by thread_id = session_id
* conditional edges           the "yes" gate and the tool loop

Python still makes every decision (tools.py / rules.py). The LLM only picks tools and words the reply.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import tools
from config import get_settings
from llm import create_llm
from prompts import build_system_prompt
from tools import ORDER_NOT_FOUND_REPLY, WRITE_TOOLS, ToolContext

log = logging.getLogger("resolvebot.agent")
_debug: dict = {"error": None}  # last LLM error, shown in the reply only when DEBUG_ERRORS=1

ROUTES = ("answer", "clarify", "no_answer", "out_of_scope")
MAX_HISTORY = 40  # messages kept per chat
CONFIRM_QUESTION = "Shall I go ahead?"

FALLBACK_REPLY = "Sorry, I'm having trouble right now. Please try again in a moment."
TOO_MANY_CALLS_REPLY = (
    "That request needs more steps than I can do in one go. "
    "Could you ask about one thing at a time, for example one order?"
)

# ------------------------------------------------------------ "yes" detection
_CORE = r"(?:yes|yeah|yea|yep|yup|sure|ok|okay|confirm|confirmed|proceed|go ahead|please do|do it|absolutely|definitely|y)"
_FILLER = r"(?:please|thanks|thank you|thx|and)"


def is_clear_yes(message: str) -> bool:
    """True for 'yes', 'confirm', 'go ahead', 'please do', 'yes, go ahead' ...
    False as soon as anything else is in the message ('yes but only the cable')."""
    text = " ".join(re.sub(r"[^a-z\s]", " ", message.lower()).split())
    if not text or not re.search(rf"\b{_CORE}\b", text):
        return False
    rest = re.sub(rf"\b(?:{_CORE}|{_FILLER})\b", " ", text)
    return not rest.strip()


# -------------------------------------------------------------------- state
def accumulate(old: Optional[list], new: Optional[list]) -> list:
    """Reducer: nodes return a list to ADD to the field; None clears it (start of a new turn)."""
    if new is None:
        return []
    return (old or []) + list(new)


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: append / replace / remove by id
    customer_id: str  # from the X-Customer-Id header only; the LLM never sees or sets it
    pending: Optional[dict[str, Any]]  # the waiting proposal: {"name": ..., "args": {...}}
    # ---- per turn (reset by the input of every request)
    tool_log: Annotated[list[dict[str, Any]], accumulate]
    retrieved: Annotated[list[list[str]], accumulate]  # [doc_id, section] pairs search_policy returned
    order_not_found: bool
    tool_count: int
    started: float
    failed: bool
    reply: Optional[dict[str, Any]]


class SessionOwnerError(Exception):
    """The session_id belongs to another customer (HTTP 403)."""


# ------------------------------------------------------------------ helpers
def _reply(
    answer: str,
    route: str = "answer",
    citations: Optional[list[dict[str, str]]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    pending: bool = False,
) -> dict[str, Any]:
    return {
        "request_id": str(uuid.uuid4()),
        "answer": answer,
        "route": route,
        "citations": citations or [],
        "tool_calls": tool_calls or [],
        "pending_confirmation": pending,
    }


def parse_final(text: str) -> dict[str, Any]:
    """Read the LLM's final JSON object; fall back to plain text."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    return {"answer": text.strip(), "route": "answer", "citations": []}


def _text(message: Any) -> str:
    """The text of a message (Anthropic may return a list of content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _text(m)
    return ""


def _tidy(messages: list[AnyMessage], answer: str) -> list[Any]:
    """Message updates that end a turn.

    Keeps the history clean and small: the user's message plus ONE plain-text assistant
    reply. Tool calls and tool results of this turn are removed, and the oldest messages
    are dropped beyond MAX_HISTORY (always starting on a user message).
    """
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    updates: list[Any] = [RemoveMessage(id=m.id) for m in messages[last_human + 1 :]]
    kept = messages[: last_human + 1]
    drop = max(0, len(kept) + 1 - MAX_HISTORY)
    while drop < len(kept) and not isinstance(kept[drop], HumanMessage):
        drop += 1
    updates = [RemoveMessage(id=m.id) for m in kept[:drop]] + updates
    updates.append(AIMessage(content=answer))
    return updates


# ------------------------------------------------------------------- agent
class Agent:
    def __init__(self, llm: Any = None, checkpointer: Any = None):
        self._llm = llm  # anything with .invoke(list_of_messages) -> AIMessage
        self._owners: dict[str, str] = {}  # session_id -> customer_id
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self._graph = self._build_graph(checkpointer or MemorySaver())

    def _get_llm(self):
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    # ---------------------------------------------------------- entry point
    def handle(self, customer_id: str, session_id: str, message: str) -> dict[str, Any]:
        with self._registry_lock:
            owner = self._owners.setdefault(session_id, customer_id)
            if owner != customer_id:
                raise SessionOwnerError(session_id)
            lock = self._locks.setdefault(session_id, threading.Lock())

        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 40}
        turn_input = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
            "tool_log": None,  # None clears the per-turn fields (see `accumulate`)
            "retrieved": None,
            "order_not_found": False,
            "tool_count": 0,
            "started": time.monotonic(),
            "failed": False,
            "reply": None,
        }
        with lock:
            try:
                _debug["error"] = None
                result = self._graph.invoke(turn_input, config)
                reply = result["reply"] or _reply(FALLBACK_REPLY)
                if os.getenv("DEBUG_ERRORS") == "1" and _debug["error"]:  # temporary: show the real error
                    reply = {**reply, "debug_error": _debug["error"]}
                return reply
            except Exception:
                log.exception("chat failed")
                self._repair(config)
                return _reply(FALLBACK_REPLY)

    def _repair(self, config: dict[str, Any]) -> None:
        """After a crash: drop half-finished tool messages and any waiting proposal."""
        try:
            messages = self._graph.get_state(config).values.get("messages", [])
            broken = [
                RemoveMessage(id=m.id)
                for m in messages
                if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
            ]
            self._graph.update_state(config, {"messages": broken, "pending": None})
        except Exception:
            log.exception("could not repair chat state")

    # ------------------------------------------------------------- the graph
    def _build_graph(self, checkpointer: Any):
        g = StateGraph(ChatState)
        g.add_node("start_turn", self._start_turn)
        g.add_node("confirm", self._confirm)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("finalize", self._finalize)
        g.add_node("too_many", self._too_many)
        g.add_node("fail", self._fail)

        g.add_conditional_edges(START, self._entry, {"confirm": "confirm", "start_turn": "start_turn"})
        g.add_edge("start_turn", "agent")
        g.add_conditional_edges(
            "agent",
            self._after_agent,
            {"tools": "tools", "finalize": "finalize", "too_many": "too_many", "fail": "fail"},
        )
        g.add_conditional_edges("tools", self._after_tools, {"agent": "agent", "finalize": "finalize"})
        for node in ("confirm", "finalize", "too_many", "fail"):
            g.add_edge(node, END)
        return g.compile(checkpointer=checkpointer)

    # ---- edges
    @staticmethod
    def _entry(state: ChatState) -> str:
        """Only a clear yes in the very next message confirms a waiting proposal."""
        if state.get("pending") and is_clear_yes(_last_human_text(state["messages"])):
            return "confirm"
        return "start_turn"

    @staticmethod
    def _after_agent(state: ChatState) -> str:
        if state.get("failed"):
            return "fail"
        calls = getattr(state["messages"][-1], "tool_calls", None) or []
        if not calls:
            return "finalize"
        if state.get("tool_count", 0) + len(calls) > get_settings().max_tool_calls:
            return "too_many"
        return "tools"

    @staticmethod
    def _after_tools(state: ChatState) -> str:
        """An order that is missing or not the customer's has a fixed reply: no more LLM calls."""
        return "finalize" if state.get("order_not_found") else "agent"

    # ---- nodes
    @staticmethod
    def _start_turn(state: ChatState) -> dict[str, Any]:
        return {"pending": None}  # anything other than a clear yes forgets the proposal

    def _confirm(self, state: ChatState) -> dict[str, Any]:
        ctx = ToolContext(state["customer_id"])
        pending = state["pending"]
        name, args = pending["name"], pending["args"]
        result = tools.dispatch(ctx, name, args)  # re-checks the rules, then saves
        failed = "error_code" in result
        entry = {"name": name, "args": args, "status": "error" if failed else "success"}

        if failed:
            default = f"Sorry, I couldn't save that, so nothing was changed. {result.get('message', '')}".strip()
        elif name == "create_return_request":
            names = ", ".join(i["name"] for i in result["items"])
            default = (
                f"Done. Your return request {result['request_id']} for {names} "
                f"(order {result['order_id']}) has been created."
            )
        else:
            kind = "cancellation" if result["ticket_type"] == "cancellation" else "warranty"
            default = (
                f"Done. I've opened {kind} ticket {result['ticket_id']} for order {result['order_id']}. "
                "Our team will follow up."
            )
        answer = self._phrase_outcome(name, result, failed, default)
        return {
            "pending": None,
            "reply": _reply(answer, "answer", tool_calls=[entry]),
            "messages": _tidy(state["messages"], answer),
        }

    def _phrase_outcome(self, name: str, result: dict[str, Any], failed: bool, default: str) -> str:
        """Let the LLM word the outcome of a confirmed action. Python already did the save and knows if it
        worked; the model only gets those facts. If it fails, or leaves out the ID, the fixed text is used."""
        try:
            system = SystemMessage(
                content=(
                    "You are ResolveBot, ShopSphere's customer support assistant. The customer said yes to an "
                    "action and our system has already carried it out (or refused it). Tell the customer the "
                    "outcome in 1 to 3 warm, plain-text sentences. Use ONLY the facts you are given: never invent "
                    "dates, amounts or promises. If it was saved, mention what was done (the item names or the "
                    "ticket type), the order ID, and the request or ticket ID exactly as given. If it failed, say "
                    "that nothing was changed and explain the reason from the message. Do not call tools. Plain "
                    "text only, no JSON, no markdown."
                )
            )
            facts = {"outcome": "failed, nothing was saved" if failed else "saved", "action": name, "details": result}
            msg = self._get_llm().invoke([system, HumanMessage(content=json.dumps(facts, ensure_ascii=False, default=str))])
            text = _text(msg).strip()
            if text.startswith("{"):
                text = str(parse_final(text).get("answer") or "").strip()
            key = str(result.get("request_id") or result.get("ticket_id") or "")
            if not text or getattr(msg, "tool_calls", None) or (not failed and key not in text):
                return default
            return text
        except Exception:
            log.exception("could not word the outcome with the LLM; using the fixed text")
            return default

    def _agent(self, state: ChatState) -> dict[str, Any]:
        settings = get_settings()
        elapsed = time.monotonic() - state["started"]
        if elapsed > settings.reply_time_budget:
            log.warning("time budget used up (%.1fs so far): giving the fallback reply", elapsed)
            _debug["error"] = f"time budget used up after {elapsed:.1f}s (REPLY_TIME_BUDGET_SECONDS)"
            return {"failed": True}
        try:
            system = SystemMessage(content=build_system_prompt(settings.app_today))
            history = [system] + list(state["messages"])
            began = time.monotonic()
            msg = self._get_llm().invoke(history)
            text = _text(msg).strip()
            wrote_a_call = bool(re.search(r"\b(?:get_orders|search_policy|check_return_eligibility|create_return_request|create_support_ticket)\s*\(", text))
            if not getattr(msg, "tool_calls", None) and (not text or wrote_a_call):
                log.warning("LLM gave an empty reply or wrote a tool call as text: asking once more")
                nudge = HumanMessage(
                    content="Use the real tool-calling mechanism if you need a tool. Otherwise reply to the "
                    "customer with the JSON object only."
                )
                msg = self._get_llm().invoke(history + [nudge])
            log.info("LLM step took %.1fs, tool calls: %s", time.monotonic() - began,
                     [c["name"] for c in (getattr(msg, "tool_calls", None) or [])])
            return {"messages": [msg]}
        except Exception as exc:
            log.exception("LLM call failed")
            cause = exc.__cause__
            detail = getattr(cause, "body", None) or getattr(getattr(cause, "response", None), "text", "")
            _debug["error"] = f"{type(exc).__name__}: {str(exc)[:500]} | {str(detail)[:1500]}"
            return {"failed": True}

    def _tools(self, state: ChatState) -> dict[str, Any]:
        ctx = ToolContext(state["customer_id"])
        calls = state["messages"][-1].tool_calls
        pending = state.get("pending")
        order_not_found = state.get("order_not_found", False)
        results: list[ToolMessage] = []
        log_entries: list[dict[str, Any]] = []
        retrieved: list[list[str]] = []

        for call in calls:
            name, args = call["name"], call["args"] if isinstance(call["args"], dict) else {}
            if name in WRITE_TOOLS:  # propose only; saving happens after a "yes"
                if pending is not None:
                    result, status = (
                        {"error_code": "ONE_PROPOSAL_ONLY", "message": "Only one proposal can wait at a time."},
                        "error",
                    )
                else:
                    error = tools.validate_write(ctx, name, args)
                    if error:
                        result, status = error, "error"
                    else:
                        pending = {"name": name, "args": args}
                        result, status = (
                            {
                                "status": "pending_confirmation",
                                "message": (
                                    "Nothing has been saved yet. Describe what you will do and ask "
                                    f"'{CONFIRM_QUESTION}'. The customer's next message decides."
                                ),
                            },
                            "pending_confirmation",
                        )
            else:
                result = tools.dispatch(ctx, name, args)
                status = "error" if "error_code" in result else "success"
                if status == "success" and name == "search_policy":
                    retrieved += [[r["doc_id"], r["section"]] for r in result.get("results", [])]

            order_not_found |= result.get("error_code") == "ORDER_NOT_FOUND"
            log_entries.append({"name": name, "args": args, "status": status})
            results.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=call["id"],
                    name=name,
                )
            )
        return {
            "messages": results,
            "tool_log": log_entries,
            "retrieved": retrieved,
            "pending": pending,
            "order_not_found": order_not_found,
            "tool_count": state.get("tool_count", 0) + len(calls),
        }

    def _finalize(self, state: ChatState) -> dict[str, Any]:
        log_entries = state.get("tool_log", [])

        # An order that is missing or not the customer's always gets the same reply.
        if state.get("order_not_found"):
            reply = _reply(ORDER_NOT_FOUND_REPLY, "no_answer", tool_calls=log_entries)
        else:
            raw = _text(state["messages"][-1])
            data = parse_final(raw)
            answer = str(data.get("answer") or "").strip()
            if not answer:
                log.warning("final reply had no answer; raw LLM text: %r", raw[:800])
                _debug["error"] = f"empty answer; raw LLM text: {raw[:800]!r}"
                answer = FALLBACK_REPLY
            route = data.get("route") if data.get("route") in ROUTES else "answer"
            pending = state.get("pending")

            if pending:
                route = "answer"
                if "go ahead" not in answer.lower():
                    answer = f"{answer} {CONFIRM_QUESTION}"

            retrieved = {tuple(pair) for pair in state.get("retrieved", [])}
            citations: list[dict[str, str]] = []
            if route in ("answer", "clarify"):
                for c in data.get("citations") or []:
                    if not isinstance(c, dict):
                        continue
                    key = (str(c.get("doc_id", "")), str(c.get("section", "")))
                    entry = {"doc_id": key[0], "section": key[1]}
                    if key in retrieved and entry not in citations:  # only paragraphs we really retrieved
                        citations.append(entry)

            if route == "out_of_scope":
                reply = _reply(answer, "out_of_scope")  # no tool_calls, no citations
            else:
                reply = _reply(answer, route, citations, log_entries, pending=bool(pending))

        return {
            "reply": reply,
            "pending": state.get("pending") if reply["pending_confirmation"] else None,
            "messages": _tidy(state["messages"], reply["answer"]),
        }

    @staticmethod
    def _too_many(state: ChatState) -> dict[str, Any]:
        return {
            "pending": None,
            "reply": _reply(TOO_MANY_CALLS_REPLY, tool_calls=state.get("tool_log", [])),
            "messages": _tidy(state["messages"], TOO_MANY_CALLS_REPLY),
        }

    @staticmethod
    def _fail(state: ChatState) -> dict[str, Any]:
        return {
            "pending": None,
            "reply": _reply(FALLBACK_REPLY, tool_calls=state.get("tool_log", [])),
            "messages": _tidy(state["messages"], FALLBACK_REPLY),
        }
