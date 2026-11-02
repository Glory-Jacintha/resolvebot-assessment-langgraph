"""HTTP API: POST /chat on port 8000.

Run:  uvicorn main:app --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from agent import Agent, SessionOwnerError
from db import customer_exists

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ResolveBot")
agent = Agent()


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    # 401: the customer ID header must be present and known.
    customer_id = (request.headers.get("x-customer-id") or "").strip()
    if not customer_id or not await run_in_threadpool(customer_exists, customer_id):
        return _error(401, "X-Customer-Id is missing or unknown")

    # 400: the body must be JSON with session_id and message.
    try:
        payload = await request.json()
    except Exception:
        return _error(400, "Body must be valid JSON")
    if not isinstance(payload, dict):
        return _error(400, "Body must be a JSON object")
    session_id, message = payload.get("session_id"), payload.get("message")
    if not isinstance(session_id, str) or not session_id.strip():
        return _error(400, "session_id is required")
    if not isinstance(message, str) or not message.strip():
        return _error(400, "message is required")

    try:
        reply = await run_in_threadpool(agent.handle, customer_id, session_id, message)
    except SessionOwnerError:
        return _error(403, "This session belongs to another customer")
    return JSONResponse(reply)
