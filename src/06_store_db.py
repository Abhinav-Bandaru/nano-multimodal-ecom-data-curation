# src/06_store_db.py
#
# Create a simple SQLite DB for curated items.
# - Reads data/curated/curated_metadata.csv
# - Adds an 'embed_index' column to map rows -> embeddings
# - Writes to db/curated.db in table 'products'

from pathlib import Path
import sqlite3

import pandas as pd


def main():
    curated_csv = Path("data/curated/curated_metadata.csv")
    db_path = Path("db/curated.db")

    print(f"[INFO] Loading curated metadata from: {curated_csv}")
    df = pd.read_csv(curated_csv)

    # This index will correspond 1:1 with rows in curated_embeddings.npz
    df = df.reset_index(drop=True)
    df["embed_index"] = df.index

    # Make sure DB directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing to SQLite DB: {db_path}")
    conn = sqlite3.connect(db_path)
    # Write to table 'products', replace if it already exists
    df.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()

    print("[INFO] Done. SQLite table 'products' created.")
    print(f"[INFO] Number of rows stored: {len(df)}")


if __name__ == "__main__":
    main()
