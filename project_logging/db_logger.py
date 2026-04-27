# Test the simple inventory
"""Simple database inventory tool for IVCorrelation project."""

import sqlite3
import pandas as pd
from pathlib import Path


def inventory_sqlite_db(db_path: str = None) -> pd.DataFrame:
    """
    Print and return inventory of all tables in the SQLite database.
    Shows table names, row counts, and all column names.

    Parameters:
    -----------
    db_path : str, optional
        Path to SQLite database. Defaults to data/iv_data.db

    Returns:
    --------
    pd.DataFrame
        Inventory with columns: table, row_count, columns
    """
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "iv_data.db"

    print("\n=== DATABASE INVENTORY ===")
    print(f"Database: {db_path}")
    print("=" * 50)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

    print(f"Found {len(tables)} tables:\n")

    inventory = []
    for (table_name,) in tables:
        try:
            # Get row count
            row_count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Get column names
            columns_info = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = [col[1] for col in columns_info]  # col[1] is the column name

            # Print table info
            print(f"Table: {table_name}")
            print(f"  Rows: {row_count:,}")
            print(f"  Columns ({len(columns)}): {', '.join(columns)}")
            print()

            inventory.append(
                {
                    "table": table_name,
                    "row_count": row_count,
                    "columns": ", ".join(columns),
                    "column_count": len(columns),
                }
            )

        except Exception as e:
            print(f"Table: {table_name}")
            print(f"  ERROR: {e}")
            print()

            inventory.append({"table": table_name, "row_count": "ERROR", "columns": f"ERROR: {e}", "column_count": 0})

    conn.close()

    # Create and return DataFrame
    df = pd.DataFrame(inventory)
    print("=" * 50)
    print(f"Total tables: {len(df)}")
    print(f"Total rows across all tables: {df[df['row_count'] != 'ERROR']['row_count'].sum():,}")
    print("=" * 50)

    return df


def print_table_schemas():
    """Print just the table schemas (names and columns) without row counts."""
    db_path = Path(__file__).parent.parent / "data" / "iv_data.db"

    print("\n=== TABLE SCHEMAS ===")
    print(f"Database: {db_path}")
    print("=" * 30)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

    for (table_name,) in tables:
        try:
            columns_info = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = [col[1] for col in columns_info]

            print(f"{table_name}:")
            print(f"  {', '.join(columns)}")
            print()

        except Exception as e:
            print(f"{table_name}: ERROR - {e}")
            print()

    conn.close()


if __name__ == "__main__":
    # Run inventory when script is executed directly
    df = inventory_sqlite_db()

    # Also show just schemas
    print_table_schemas()
