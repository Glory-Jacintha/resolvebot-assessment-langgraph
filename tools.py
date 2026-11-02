"""The five tools the agent can call, plus the schemas the LLM sees.

Safety rules enforced here, in Python (never by the LLM):
* The customer ID always comes from the X-Customer-Id header via ToolContext.
* Every order lookup is scoped to that customer. An order that does not exist and
  an order that belongs to someone else give the identical ORDER_NOT_FOUND error.
* Writes re-run their checks right before saving.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import policy_search
import rules
from config import get_settings
from db import connect, load_policy_parameters

log = logging.getLogger("resolvebot.tools")

ORDER_NOT_FOUND_REPLY = "I couldn't find that order on your account."
RETURN_REASONS = ("changed_mind", "defective")
TICKET_TYPES = ("warranty", "cancellation")
ORDER_STATUSES = ("processing", "shipped", "delivered", "cancelled")
WRITE_TOOLS = ("create_return_request", "create_support_ticket")


@dataclass(frozen=True)
class ToolContext:
    customer_id: str  # from the X-Customer-Id header only


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error_code": code, "message": message, **extra}


def _order_not_found() -> dict[str, Any]:
    # Same text for "does not exist" and "belongs to someone else".
    return _err("ORDER_NOT_FOUND", ORDER_NOT_FOUND_REPLY)


# ------------------------------------------------------------- id helpers
def normalize_order_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    m = re.fullmatch(r"SS[-\s]?(\d+)", text)
    return f"SS-{m.group(1)}" if m else text


def normalize_item_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    m = re.fullmatch(r"SS[-\s]?(\d+)[-\s](\d+)", text)
    return f"SS-{m.group(1)}-{m.group(2)}" if m else text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


# ------------------------------------------------------------ data access
def _load_order(conn: sqlite3.Connection, customer_id: str, order_id: str) -> Optional[dict[str, Any]]:
    """The order, its items and its shipment, only if it belongs to this customer."""
    row = conn.execute(
        "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?", (order_id, customer_id)
    ).fetchone()
    if row is None:
        return None
    items = conn.execute(
        """SELECT oi.item_id, oi.product_id, oi.qty, oi.unit_price, oi.opened,
                  p.name, p.category, p.returnable, p.warranty_class
             FROM order_items oi JOIN products p ON p.product_id = oi.product_id
            WHERE oi.order_id = ? ORDER BY oi.item_id""",
        (order_id,),
    ).fetchall()
    shipment = conn.execute(
        "SELECT * FROM shipments WHERE order_id = ? ORDER BY rowid LIMIT 1", (order_id,)
    ).fetchone()
    return {
        "order": dict(row),
        "items": [dict(i) for i in items],
        "shipment": dict(shipment) if shipment else None,
    }


def _delivered_on(loaded: dict[str, Any], today: date) -> Optional[date]:
    shipment = loaded["shipment"] or {}
    return rules.delivery_date(loaded["order"].get("status"), shipment.get("delivered_at"), today)


def _shape_order(loaded: dict[str, Any], params: dict[str, Any], today: date) -> dict[str, Any]:
    order, shipment = loaded["order"], loaded["shipment"]
    delivered = _delivered_on(loaded, today)
    window = rules.int_param(params, "return_window_days")
    days = rules.days_since(delivered, today) if delivered else None
    last_return_day = delivered + timedelta(days=window) if delivered else None

    items = []
    for it in loaded["items"]:
        months = rules.warranty_months(it["warranty_class"], params)
        end = rules.warranty_end(delivered, months) if (delivered and months) else None
        items.append(
            {
                "item_id": it["item_id"],
                "product_id": it["product_id"],
                "name": it["name"],
                "category": it["category"],
                "qty": it["qty"],
                "unit_price": it["unit_price"],
                "opened": bool(it["opened"]),
                "returnable": it["returnable"],
                "warranty_class": it["warranty_class"],
                "warranty_months": months,
                "warranty_end_date": end.isoformat() if end else None,
                "warranty_end_text": rules.fmt_date(end),
                "warranty_active": bool(end and rules.warranty_is_active(end, today)),
            }
        )

    order_date = rules.parse_date(order.get("order_date"))
    ship_out = None
    if shipment:
        eta = rules.parse_date(shipment.get("eta"))
        ship_out = {
            "carrier": shipment.get("carrier"),
            "status": shipment.get("status"),
            "eta": eta.isoformat() if eta else None,
            "eta_text": rules.fmt_date(eta),
        }
    return {
        "order_id": order["order_id"],
        "order_date": order_date.isoformat() if order_date else order.get("order_date"),
        "order_date_text": rules.fmt_date(order_date),
        "status": order["status"],
        "total": order.get("total"),
        "shipment": ship_out,
        "delivery_date": delivered.isoformat() if delivered else None,
        "delivery_date_text": rules.fmt_date(delivered),
        "days_since_delivery": days,
        "return_window_days": window,
        "return_window_open": (days <= window) if days is not None else None,
        "last_return_date_text": rules.fmt_date(last_return_day),
        "items": items,
    }


# ============================================================ tool 1: reads
def get_orders(ctx: ToolContext, order_id: Any = None, status: Any = None) -> dict[str, Any]:
    today = get_settings().app_today
    wanted_status = str(status).strip().lower() if status not in (None, "") else None
    if wanted_status and wanted_status not in ORDER_STATUSES:
        return _err("INVALID_STATUS", f"status must be one of: {', '.join(ORDER_STATUSES)}")

    with connect() as conn:
        params = load_policy_parameters(conn)
        loaded_orders = []
        if order_id not in (None, ""):
            loaded = _load_order(conn, ctx.customer_id, normalize_order_id(order_id))
            if loaded is None:
                return _order_not_found()
            loaded_orders = [loaded]
        else:
            ids = conn.execute(
                "SELECT order_id FROM orders WHERE customer_id = ? ORDER BY order_date DESC, order_id DESC",
                (ctx.customer_id,),
            ).fetchall()
            for row in ids:
                loaded = _load_order(conn, ctx.customer_id, row["order_id"])
                if loaded:
                    loaded_orders.append(loaded)

        if wanted_status:
            loaded_orders = [o for o in loaded_orders if str(o["order"]["status"]).lower() == wanted_status]
        orders = [_shape_order(o, params, today) for o in loaded_orders]
    return {"count": len(orders), "orders": orders}


def search_policy(ctx: ToolContext, query: Any = None) -> dict[str, Any]:
    text = str(query or "").strip()
    if not text:
        return _err("INVALID_QUERY", "query must not be empty")
    results = policy_search.search(text)
    if not results:
        return {
            "count": 0,
            "results": [],
            "message": "No current policy paragraph matches this query.",
        }
    return {"count": len(results), "results": results}


def check_return_eligibility(
    ctx: ToolContext, order_id: Any = None, item_ids: Any = None, reason: Any = None
) -> dict[str, Any]:
    today = get_settings().app_today
    with connect() as conn:
        params = load_policy_parameters(conn)
        loaded = _load_order(conn, ctx.customer_id, normalize_order_id(order_id))
        if loaded is None:
            return _order_not_found()

        reason_text = str(reason or "").strip().lower()
        if reason_text not in RETURN_REASONS:
            return _err("INVALID_REASON", f"reason must be one of: {', '.join(RETURN_REASONS)}")

        wanted: list[str] = []
        for raw in _as_list(item_ids):
            item_id = normalize_item_id(raw)
            if item_id not in wanted:
                wanted.append(item_id)
        if not wanted:
            return _err("INVALID_ITEM", "At least one item_id is required")
        by_id = {i["item_id"]: i for i in loaded["items"]}
        for item_id in wanted:
            if item_id not in by_id:
                return _err("INVALID_ITEM", f"Item {item_id} is not in this order")
        selected = [by_id[i] for i in wanted]

        shipment = loaded["shipment"] or {}
        window = rules.int_param(params, "return_window_days")
        error = rules.check_return(
            order_status=loaded["order"].get("status"),
            delivered_at=shipment.get("delivered_at"),
            items=selected,
            reason=reason_text,
            today=today,
            window_days=window,
        )
        if error:
            return error

        delivered = _delivered_on(loaded, today)
        days = rules.days_since(delivered, today)
        return {
            "result": "eligible",
            "order_id": loaded["order"]["order_id"],
            "items": [{"item_id": i["item_id"], "name": i["name"]} for i in selected],
            "reason": reason_text,
            "delivery_date_text": rules.fmt_date(delivered),
            "days_since_delivery": days,
            "return_window_days": window,
        }


# ========================================================== tool 2: writes
def _validate_ticket(
    conn: sqlite3.Connection, ctx: ToolContext, order_id: Any, ticket_type: Any, summary: Any
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], str]:
    """Returns (error, loaded_order, normalized_ticket_type)."""
    today = get_settings().app_today
    params = load_policy_parameters(conn)
    loaded = _load_order(conn, ctx.customer_id, normalize_order_id(order_id))
    if loaded is None:
        return _order_not_found(), None, ""

    kind = str(ticket_type or "").strip().lower()
    if kind not in TICKET_TYPES:
        return _err("INVALID_TICKET_TYPE", f"ticket_type must be one of: {', '.join(TICKET_TYPES)}"), None, ""
    if not str(summary or "").strip():
        return _err("INVALID_ARGUMENTS", "summary must not be empty"), None, ""

    status = str(loaded["order"].get("status") or "").strip().lower()
    if kind == "cancellation":
        if status == "processing":
            return None, loaded, kind
        if status == "cancelled":
            return _err("ORDER_ALREADY_CANCELLED", "This order has already been cancelled."), None, ""
        return (
            _err("ORDER_ALREADY_SHIPPED", "This order has already shipped, so it can no longer be cancelled."),
            None,
            "",
        )

    # warranty
    delivered = _delivered_on(loaded, today)
    if delivered is None:
        return _err("NOT_DELIVERED", "This order has not been delivered yet, so no warranty applies."), None, ""
    months = [rules.warranty_months(i["warranty_class"], params) for i in loaded["items"]]
    if not any(months):
        return _err("NO_WARRANTY", "None of the items in this order have a warranty."), None, ""
    active = [
        m for m in months if m and rules.warranty_is_active(rules.warranty_end(delivered, m), today)
    ]
    if not active:
        return _err("WARRANTY_EXPIRED", "The warranty has ended for every item in this order."), None, ""
    return None, loaded, kind


def create_return_request(
    ctx: ToolContext, order_id: Any = None, item_ids: Any = None, reason: Any = None
) -> dict[str, Any]:
    # Check again right before saving.
    check = check_return_eligibility(ctx, order_id, item_ids, reason)
    if "error_code" in check:
        return check

    request_id = str(uuid.uuid4())
    stored_items = [i["item_id"] for i in check["items"]]
    with connect() as conn:
        conn.execute(
            """INSERT INTO return_requests
                   (request_id, order_id, customer_id, item_ids, reason, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'requested', ?)""",
            (
                request_id,
                check["order_id"],
                ctx.customer_id,
                json.dumps(stored_items),
                check["reason"],
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return {
        "result": "created",
        "request_id": request_id,
        "order_id": check["order_id"],
        "items": check["items"],
        "reason": check["reason"],
        "request_status": "requested",
    }


def create_support_ticket(
    ctx: ToolContext, order_id: Any = None, ticket_type: Any = None, summary: Any = None
) -> dict[str, Any]:
    with connect() as conn:
        error, loaded, kind = _validate_ticket(conn, ctx, order_id, ticket_type, summary)
        if error:
            return error
        ticket_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO support_tickets
                   (ticket_id, order_id, customer_id, ticket_type, summary, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'open', ?)""",
            (
                ticket_id,
                loaded["order"]["order_id"],
                ctx.customer_id,
                kind,
                str(summary).strip(),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return {
        "result": "created",
        "ticket_id": ticket_id,
        "order_id": loaded["order"]["order_id"],
        "ticket_type": kind,
        "ticket_status": "open",
    }


def validate_write(ctx: ToolContext, name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Dry run for a proposal: the same checks as the write, but saves nothing.
    Returns an error dict, or None if the write would be allowed."""
    try:
        if name == "create_return_request":
            result = check_return_eligibility(
                ctx, args.get("order_id"), args.get("item_ids"), args.get("reason")
            )
            return result if "error_code" in result else None
        if name == "create_support_ticket":
            with connect() as conn:
                error, _, _ = _validate_ticket(
                    conn, ctx, args.get("order_id"), args.get("ticket_type"), args.get("summary")
                )
            return error
    except Exception:  # never crash on a bad proposal
        log.exception("validate_write failed")
        return _err("INTERNAL_ERROR", "Something went wrong while checking this request.")
    return _err("UNKNOWN_TOOL", f"{name} is not a write tool")


def item_names(ctx: ToolContext, order_id: Any, item_ids: Any) -> list[str]:
    """Product names for a confirmation message (scoped to this customer)."""
    with connect() as conn:
        loaded = _load_order(conn, ctx.customer_id, normalize_order_id(order_id))
    if not loaded:
        return []
    wanted = {normalize_item_id(i) for i in _as_list(item_ids)}
    return [i["name"] for i in loaded["items"] if i["item_id"] in wanted]


# ================================================================ dispatch
_FUNCTIONS = {
    "get_orders": get_orders,
    "search_policy": search_policy,
    "check_return_eligibility": check_return_eligibility,
    "create_return_request": create_return_request,
    "create_support_ticket": create_support_ticket,
}


def dispatch(ctx: ToolContext, name: str, args: Any) -> dict[str, Any]:
    """Run a tool by name. Never raises."""
    func = _FUNCTIONS.get(name)
    if func is None:
        return _err("UNKNOWN_TOOL", f"There is no tool called {name}")
    if not isinstance(args, dict):
        return _err("INVALID_ARGUMENTS", "Tool arguments must be a JSON object")
    try:
        return func(ctx, **args)
    except TypeError as exc:
        return _err("INVALID_ARGUMENTS", f"Bad arguments for {name}: {exc}")
    except Exception:
        log.exception("tool %s crashed", name)
        return _err("INTERNAL_ERROR", "Something went wrong while running this tool.")


# =================================================================== schemas
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_orders",
        "description": (
            "Look up the logged-in customer's own orders. Each order comes with its items "
            "(name, opened or not, returnable rule, warranty end date and whether the warranty "
            "is still active), shipment status and ETA, delivery date, and days since delivery. "
            "Pass order_id to fetch one order, or leave it out to list all of the customer's "
            "orders (optionally filtered by status). Use this to track an order, to find which "
            "order or item the customer means, and before any return, warranty or cancellation. "
            "Never needed to be told the customer ID: it is already known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID like SS-10458 (optional)."},
                "status": {
                    "type": "string",
                    "enum": list(ORDER_STATUSES),
                    "description": "Only return orders with this status (optional).",
                },
            },
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search ShopSphere's policy documents (returns, refunds, warranty, shipping, damaged "
            "items, exchanges, cancellation, FAQ). Returns the best matching paragraphs, each with "
            "doc_id and section so you can cite them. Use this whenever you state a rule, deadline, "
            "fee or process, and for any general policy question. Use short keyword queries such as "
            "'return window' or 'cancel order after shipment'. Paragraphs marked in_effect=false "
            "are future changes, not current rules."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Short keyword query."}},
            "required": ["query"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": (
            "Check whether items from one order can be returned. Read-only; saves nothing. "
            "Returns result 'eligible' or an error_code (e.g. RETURN_WINDOW_EXPIRED, "
            "OPENED_NOT_RETURNABLE, NON_RETURNABLE, NOT_DELIVERED). Always call this before "
            "telling the customer a return is possible or proposing one. reason is 'changed_mind' "
            "when the customer just does not want the item, or 'defective' when it is broken, "
            "damaged or the wrong item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item IDs like SS-10458-1 from get_orders.",
                },
                "reason": {"type": "string", "enum": list(RETURN_REASONS)},
            },
            "required": ["order_id", "item_ids", "reason"],
        },
    },
    {
        "name": "create_return_request",
        "description": (
            "Propose a return request for the customer. Call this ONLY when the customer has asked "
            "you to return the item(s), and only after check_return_eligibility returned 'eligible' "
            "for the same order, items and reason. This does NOT save anything yet: it records a "
            "proposal that the customer must confirm with a 'yes' in their next message. After "
            "calling it, describe what you will do and ask 'Shall I go ahead?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string", "enum": list(RETURN_REASONS)},
            },
            "required": ["order_id", "item_ids", "reason"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Propose a support ticket. ticket_type 'warranty': for a fault covered by warranty "
            "(the item must be delivered and still under warranty, check get_orders first). "
            "ticket_type 'cancellation': to cancel an order that is still 'processing'. Call this "
            "ONLY when the customer asked you to open the ticket / cancel the order. This does NOT "
            "save anything yet: the customer must confirm with a 'yes' in their next message. "
            "After calling it, describe what you will do and ask 'Shall I go ahead?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "ticket_type": {"type": "string", "enum": list(TICKET_TYPES)},
                "summary": {
                    "type": "string",
                    "description": "One or two sentences: which item and what the problem or request is.",
                },
            },
            "required": ["order_id", "ticket_type", "summary"],
        },
    },
]
