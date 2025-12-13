"""Run SQL migrations (simple runner).

This script reads `sql/schema.sql` and executes the statements against the
database specified by the `DATABASE_URL` environment variable. If not set,
it will fallback to the default used by `PostgresDB`.
"""
from __future__ import annotations

import os
from pathlib import Path
import psycopg2
from urllib.parse import urlparse


def get_conn_params_from_env():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    parsed = urlparse(database_url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or "",
        "user": parsed.username or "",
        "password": parsed.password or "",
    }


def run_migrations(sql_path: str = "sql/schema.sql"):
    p = Path(sql_path)
    if not p.exists():
        raise FileNotFoundError(f"Schema file not found: {p}")

    params = get_conn_params_from_env()
    if params is None:
        raise EnvironmentError("DATABASE_URL not set; please set it to run migrations.")

    conn = None
    try:
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            dbname=params["database"],
            user=params["user"],
            password=params["password"],
        )
        cur = conn.cursor()
        sql = p.read_text()
        cur.execute(sql)
        conn.commit()
        cur.close()
        print("Migrations applied successfully.")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SQL migrations from sql/schema.sql")
    parser.add_argument("--sql", default="sql/schema.sql", help="Path to SQL schema file")
    args = parser.parse_args()
    run_migrations(args.sql)
