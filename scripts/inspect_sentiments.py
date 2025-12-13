"""Inspect sentiment_label values stored in the database.

Prints distinct sentiment labels and a few sample rows where sentiment_label
is NULL or present to help debug insertion issues.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.db_helper import PostgresDB


def main():
    db = PostgresDB()
    db.init_pool()
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT sentiment_label FROM reviews ORDER BY sentiment_label NULLS FIRST")
                labels = cur.fetchall()
                print('Distinct sentiment_label values:')
                for l in labels:
                    print(' ', l[0])

                print('\nCounts by sentiment_label:')
                cur.execute("SELECT sentiment_label, COUNT(*) FROM reviews GROUP BY sentiment_label ORDER BY COUNT(*) DESC")
                for lbl, cnt in cur.fetchall():
                    print(f'  {lbl}: {cnt}')

                print('\nSample rows with NULL sentiment_label:')
                cur.execute("SELECT review_id, orig_review_id, bank_id, rating, review_text FROM reviews WHERE sentiment_label IS NULL LIMIT 5")
                for row in cur.fetchall():
                    print(row)

                print('\nSample rows with non-NULL sentiment_label:')
                cur.execute("SELECT review_id, orig_review_id, bank_id, rating, sentiment_label, substring(review_text for 80) FROM reviews WHERE sentiment_label IS NOT NULL LIMIT 5")
                for row in cur.fetchall():
                    print(row)
                # Debug a known orig id from CSV
                cur.execute("SELECT sentiment_label, sentiment_score, pg_typeof(sentiment_label) FROM reviews WHERE orig_review_id = %s", ('3463230e-f9f7-4be3-a632-fdd8d017ce84',))
                row = cur.fetchone()
                print('\nDebug specific orig id ->', row)
    finally:
        db.close_pool()


if __name__ == '__main__':
    main()
