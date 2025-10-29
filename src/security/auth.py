"""
Basic security / access control helpers.

This is where we enforce:
- The user can ONLY access their own orders.
- We strip obvious SQL/prompt-injection attempts from free text (lightweight).

In a real production system, you'd also:
- verify the session / JWT
- check rate limiting
- log suspicious attempts
"""


def verify_order_ownership(user_id: int, order_info: dict | None) -> bool:
    """
    Check that the order belongs to the given user.
    If order_info is None, return False.
    """
    if order_info is None:
        return False
    return order_info.get("user_id") == user_id


def sanitize_user_input(user_input: str) -> str:
    """
    Very naive sanitation step. This is mostly illustrative for the report.

    We remove obvious attack patterns like 'DROP TABLE', 'SELECT *', etc.
    NOTE: This is NOT bulletproof, it's mostly here to show intent.
    """
    forbidden_tokens = [
        "SELECT", "DROP", "DELETE", "UPDATE",
        "--", ";", "/*", "*/"
    ]

    cleaned = user_input
    for token in forbidden_tokens:
        cleaned = cleaned.replace(token, "")

    return cleaned