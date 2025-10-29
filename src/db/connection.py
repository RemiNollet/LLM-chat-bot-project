"""
This module centralizes the SQLite connection logic.
If we later switch to Postgres or an API microservice, we only change here.
Database can be dowloaded at: https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/projects/e23c6b/data/orders.db
"""

import sqlite3
from config import DB_PATH


def get_db_connection():
    """
    Open a SQLite connection to the orders database.
    Row factory is set to sqlite3.Row so that we can access columns by name.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

