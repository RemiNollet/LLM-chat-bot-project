"""
This module contains safe, parameterized queries.

IMPORTANT:
- We ALWAYS filter by user_id in SQL to prevent data leakage.
- We never allow the LLM to directly build SQL strings.
"""

from typing import List, Dict, Any, Optional
from .connection import get_db_connection


def fetch_orders_for_user(user_id: int) -> List[Dict[str, Any]]:
    """
    Return the recent orders for a specific user, most recent first.

    This is used both to:
    - show potential options to the LLM (so it knows which order IDs exist),
    - let the LLM choose which order the user is referring to.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT order_id, status, date_purchase, date_shipped, date_delivered
        FROM orders
        WHERE user_id = ?
        ORDER BY date_purchase DESC
        LIMIT 5
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_order_status(user_id: int, order_id: int) -> Optional[Dict[str, Any]]:
    """
    Return details of one specific order belonging to this user.
    If no order is found for that user/order pair, return None.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT order_id, user_id, status,
               date_purchase, date_shipped, date_delivered
        FROM orders
        WHERE user_id = ? AND order_id = ?
        LIMIT 1
    """, (user_id, order_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None