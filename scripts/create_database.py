"""Create the target Postgres database if it doesn't exist.

This script reads `DATABASE_URL` from the environment, connects to the
server (using the maintenance DB `postgres`), checks if the target
database exists, and creates it if necessary.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import psycopg2


def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL not set")

    parsed = urlparse(database_url)
    target_db = parsed.path.lstrip("/")
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = parsed.username or "postgres"
    password = parsed.password or ""

    # Connect to the maintenance DB 'postgres' to run CREATE DATABASE
    conn = None
    try:
        conn = psycopg2.connect(host=host, port=port, dbname="postgres", user=user, password=password)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        exists = cur.fetchone() is not None
        if exists:
            print(f"Database '{target_db}' already exists.")
        else:
            print(f"Creating database '{target_db}'...")
            cur.execute(f"CREATE DATABASE \"{target_db}\"")
            print("Database created.")
        cur.close()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
