"""Simple DB connection tester using psycopg2 or PostgresDB helper.

This script will attempt to connect to the database using the
`DATABASE_URL` environment variable and execute a simple `SELECT 1`.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

try:
    from src.utils.db_helper import PostgresDB
except Exception:
    PostgresDB = None

import psycopg2


def test_with_psycopg2():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL not set")
    parsed = urlparse(database_url)
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port,
        dbname=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1")
    val = cur.fetchone()
    cur.close()
    conn.close()
    return val


def test_with_helper():
    if not PostgresDB:
        return None
    db = PostgresDB()
    db.init_pool()
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()
    finally:
        db.close_pool()


if __name__ == "__main__":
    print("Testing DB connection using DATABASE_URL")
    try:
        v = test_with_psycopg2()
        print("psycopg2: SELECT 1 ->", v)
    except Exception as e:
        print("psycopg2 test failed:", e)

    if PostgresDB:
        try:
            v2 = test_with_helper()
            print("PostgresDB helper: SELECT 1 ->", v2)
        except Exception as e:
            print("PostgresDB helper test failed:", e)
