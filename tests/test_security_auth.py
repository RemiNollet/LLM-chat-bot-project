# tests/test_security_auth.py
from src.security.auth import verify_order_ownership, sanitize_user_input

def test_verify_order_ownership_valid():
    """Should return True if order belongs to user."""
    user_id = 6
    order = {"order_id": 5, "user_id": 6, "status": "delivered"}
    assert verify_order_ownership(user_id, order) is True


def test_verify_order_ownership_invalid():
    """Should return False if order belongs to another user."""
    user_id = 6
    order = {"order_id": 5, "user_id": 12, "status": "delivered"}
    assert verify_order_ownership(user_id, order) is False

def test_sanitize_user_input_removes_sql_keywords():
    text = "1; DROP TABLE users; --"
    cleaned = sanitize_user_input(text)
    assert "DROP TABLE" not in cleaned.upper()