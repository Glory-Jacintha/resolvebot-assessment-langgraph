"""HTTP API: POST /chat on port 8000.

Run:  uvicorn main:app --port 8000
"""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from agent import Agent, SessionOwnerError
from db import customer_exists

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ResolveBot")
agent = Agent()

# Swagger only: adds an "Authorize" button where the customer ID is entered once and then sent as the
# X-Customer-Id header with every request, so the request forms show only the body fields.
# The handlers below still read and check the header by hand (auto_error=False),
# so the status codes stay 400 / 401 / 403.
_customer_header = APIKeyHeader(
    name="X-Customer-Id",
    auto_error=False,
    description="The logged-in customer, for example C1001. Missing or unknown -> 401.",
)

# Swagger only: the body shown for POST /chat (parsing is still done by hand below).
_CHAT_DOCS = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["session_id", "message"],
                },
                "example": {"session_id": "s1", "message": "Hi, what can you help me with?"},
            }
        },
    },
}


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


@app.post("/chat", openapi_extra=_CHAT_DOCS, dependencies=[Depends(_customer_header)])
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

