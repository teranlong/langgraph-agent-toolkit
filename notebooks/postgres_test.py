# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: agent-service-toolkit
#     language: python
#     name: python3
# ---

# %%
"""
Simple connectivity and sanity checks for the Postgres card data.
"""

# %%

# %%
import os
from pprint import pprint

import psycopg

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "56432"))
DB_NAME = os.getenv("POSTGRES_DB", "cuecards")
DB_USER = os.getenv("POSTGRES_USER", "cuecards")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "cuecards")

conn = psycopg.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)

# %%
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM cards;")
    (row_count,) = cur.fetchone()
    expected_rows = 4413
    print(f"Row count: {row_count}")
    assert row_count == expected_rows, f"Expected {expected_rows} rows, found {row_count}"

    cur.execute(
        """
        SELECT id, name, album, collection, rarity, energy, power, ppe
        FROM cards
        ORDER BY id
        LIMIT 5;
        """
    )
    preview_rows = cur.fetchall()

print("Preview rows:")
pprint(preview_rows)

# %%
query_text = "%worm%"
with conn.cursor() as cur:
    cur.execute(
        """
        SELECT id, name, ability_name
        FROM cards
        WHERE name ILIKE %s
        ORDER BY name
        LIMIT 5;
        """,
        (query_text,),
    )
    matches = cur.fetchall()

print(f"Example query for name ILIKE {query_text!r}:")
pprint(matches)

# %%
conn.close()
