"""Business rules. Pure Python, no database and no LLM.

Every number comes in through ``params`` / ``window_days`` (read from the
policy_parameters table by the caller). Nothing business-related is typed here.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from dateutil.relativedelta import relativedelta


# ----------------------------------------------------------------- dates
def parse_date(value: Any) -> Optional[date]:
    """Accepts '2026-06-21', '2026-06-21 10:30:00', datetime or date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def fmt_date(d: Optional[date]) -> Optional[str]:
    """3 July 2026"""
    if d is None:
        return None
    return f"{d.day} {d.strftime('%B %Y')}"


def int_param(params: dict[str, Any], key: str) -> int:
    try:
        return int(float(params[key]))
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(f"policy_parameters has no numeric value for '{key}'")


# -------------------------------------------------------------- delivery
def delivery_date(order_status: Any, delivered_at: Any, today: date) -> Optional[date]:
    """The delivery date, or None if the order has not been delivered."""
    if str(order_status or "").strip().lower() != "delivered":
        return None
    delivered = parse_date(delivered_at)
    if delivered is None or delivered > today:
        return None
    return delivered


def days_since(delivered: date, today: date) -> int:
    return (today - delivered).days


# --------------------------------------------------------------- returns
def check_return(
    *,
    order_status: Any,
    delivered_at: Any,
    items: list[dict[str, Any]],
    reason: str,
    today: date,
    window_days: int,
) -> Optional[dict[str, Any]]:
    """Checks 1 (delivered), 2, 3 and 4 from the spec, in order.

    ``items`` are the selected order items: dicts with item_id, name,
    returnable (yes / unopened_only / no) and opened (truthy if opened).
    Returns an error dict, or None when the items are eligible.
    (ORDER_NOT_FOUND, INVALID_REASON and INVALID_ITEM are handled by the caller.)
    """
    delivered = delivery_date(order_status, delivered_at, today)
    if delivered is None:
        return {
            "error_code": "NOT_DELIVERED",
            "message": "This order has not been delivered yet, so it cannot be returned.",
        }

    for item in items:  # check 2
        if _returnable(item) == "no":
            return {
                "error_code": "NON_RETURNABLE",
                "message": f"{item['name']} cannot be returned.",
                "item_id": item["item_id"],
            }

    days = days_since(delivered, today)  # check 3: 30 days or fewer is fine
    if days > window_days:
        return {
            "error_code": "RETURN_WINDOW_EXPIRED",
            "message": (
                f"The return window is {window_days} days and this order was delivered "
                f"{days} days ago."
            ),
            "days_since_delivery": days,
            "return_window_days": window_days,
        }

    if reason == "changed_mind":  # check 4
        for item in items:
            if item.get("opened") and _returnable(item) == "unopened_only":
                return {
                    "error_code": "OPENED_NOT_RETURNABLE",
                    "message": (
                        f"{item['name']} has been opened and can only be returned "
                        "for a change of mind while unopened."
                    ),
                    "item_id": item["item_id"],
                }
    return None


def _returnable(item: dict[str, Any]) -> str:
    return str(item.get("returnable") or "").strip().lower()


# -------------------------------------------------------------- warranty
def warranty_months(warranty_class: Any, params: dict[str, Any]) -> int:
    cls = str(warranty_class or "").strip().lower()
    if cls == "standard":
        return int_param(params, "standard_warranty_months")
    if cls == "extended":
        return int_param(params, "extended_warranty_months")
    return 0  # "none" or unknown


def warranty_end(delivered: date, months: int) -> date:
    """Same date N months later; clipped to month end (31 Jan + 1 month = 28 Feb)."""
    return delivered + relativedelta(months=months)


def warranty_is_active(end: date, today: date) -> bool:
    return today <= end  # the last day is still covered
