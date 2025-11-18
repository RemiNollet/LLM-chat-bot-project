# tests/test_db_queries.py
from src.db.queries import fetch_orders_for_user, fetch_order_status


def test_fetch_orders_for_user_returns_list():
    user_id = 6  # adapt to your dev DB
    orders = fetch_orders_for_user(user_id)

    assert isinstance(orders, list)
    if orders:
        assert isinstance(orders[0], dict)
        assert "order_id" in orders[0]


def test_fetch_order_status_for_existing_order():
    user_id = 6
    order_id = 5  # adapt to known test order
    order = fetch_order_status(user_id, order_id)

    if order is not None:
        assert order["order_id"] == order_id
        assert order["user_id"] == user_id