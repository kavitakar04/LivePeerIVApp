"""
Ticker Groups management for storing and retrieving preset ticker combinations.
Allows users to save commonly used target + peers combinations for quick access.
"""

from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional
from .db_utils import get_conn, check_db_health

logger = logging.getLogger(__name__)


def save_ticker_group(
    group_name: str,
    target_ticker: str,
    peer_tickers: List[str],
    description: str = "",
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """
    Save a ticker group preset to the database.

    Args:
        group_name: Unique name for this group
        target_ticker: The target ticker symbol
        peer_tickers: List of peer ticker symbols
        description: Optional description of the group
        conn: Database connection (if None, creates new one)

    Returns:
        True if saved successfully, False otherwise
    """
    if not group_name or not target_ticker or not peer_tickers:
        return False

    if conn is None:
        conn = get_conn()
        should_close = True
    else:
        should_close = False

    try:
        now = datetime.now(timezone.utc).isoformat()
        peer_tickers_json = json.dumps([t.upper().strip() for t in peer_tickers])
        target_ticker = target_ticker.upper().strip()

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ticker_groups
                (group_name, target_ticker, peer_tickers, description, created_at, updated_at)
                VALUES (?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM ticker_groups WHERE group_name = ?), ?),
                        ?)
            """,
                (group_name, target_ticker, peer_tickers_json, description, group_name, now, now),
            )
        check_db_health(conn)
        return True

    except Exception:
        logger.exception("error saving ticker group group_name=%s", group_name)
        return False
    finally:
        if should_close:
            conn.close()


def load_ticker_group(group_name: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict]:
    """
    Load a ticker group by name.

    Args:
        group_name: Name of the group to load
        conn: Database connection (if None, creates new one)

    Returns:
        Dict with keys: group_name, target_ticker, peer_tickers, description, created_at, updated_at
        None if group not found
    """
    if conn is None:
        conn = get_conn()
        should_close = True
    else:
        should_close = False

    try:
        cursor = conn.execute(
            """
            SELECT group_name, target_ticker, peer_tickers, description, created_at, updated_at
            FROM ticker_groups WHERE group_name = ?
        """,
            (group_name,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "group_name": row[0],
            "target_ticker": row[1],
            "peer_tickers": json.loads(row[2]),
            "description": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    except Exception:
        logger.exception("error loading ticker group group_name=%s", group_name)
        return None
    finally:
        if should_close:
            conn.close()


def list_ticker_groups(conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
    """
    Get all ticker groups.

    Args:
        conn: Database connection (if None, creates new one)

    Returns:
        List of dicts, each containing group info
    """
    if conn is None:
        conn = get_conn()
        should_close = True
    else:
        should_close = False

    try:
        cursor = conn.execute("""
            SELECT group_name, target_ticker, peer_tickers, description, created_at, updated_at
            FROM ticker_groups
            ORDER BY group_name
        """)

        groups = []
        for row in cursor.fetchall():
            groups.append(
                {
                    "group_name": row[0],
                    "target_ticker": row[1],
                    "peer_tickers": json.loads(row[2]),
                    "description": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                }
            )
        return groups

    except Exception:
        logger.exception("error listing ticker groups")
        return []
    finally:
        if should_close:
            conn.close()


def delete_ticker_group(group_name: str, conn: Optional[sqlite3.Connection] = None) -> bool:
    """
    Delete a ticker group by name.

    Args:
        group_name: Name of the group to delete
        conn: Database connection (if None, creates new one)

    Returns:
        True if deleted successfully, False otherwise
    """
    if conn is None:
        conn = get_conn()
        should_close = True
    else:
        should_close = False

    try:
        with conn:
            cursor = conn.execute("DELETE FROM ticker_groups WHERE group_name = ?", (group_name,))
            return cursor.rowcount > 0

    except Exception:
        logger.exception("error deleting ticker group group_name=%s", group_name)
        return False
    finally:
        if should_close:
            conn.close()


def get_groups_for_target(target_ticker: str, conn: Optional[sqlite3.Connection] = None) -> List[str]:
    """
    Get all group names that have the specified ticker as target.

    Args:
        target_ticker: Target ticker to search for
        conn: Database connection (if None, creates new one)

    Returns:
        List of group names
    """
    if conn is None:
        conn = get_conn()
        should_close = True
    else:
        should_close = False

    try:
        cursor = conn.execute(
            """
            SELECT group_name FROM ticker_groups
            WHERE UPPER(target_ticker) = UPPER(?)
            ORDER BY group_name
        """,
            (target_ticker,),
        )

        return [row[0] for row in cursor.fetchall()]

    except Exception:
        logger.exception("error getting groups for target target=%s", target_ticker)
        return []
    finally:
        if should_close:
            conn.close()


def create_default_groups(conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Create some default ticker groups for common analysis.
    """
    default_groups = [
        {
            "group_name": "Tech Giants vs SPY",
            "target_ticker": "SPY",
            "peer_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "description": "Major tech stocks vs S&P 500",
        },
        {
            "group_name": "Semiconductors vs SMH",
            "target_ticker": "SMH",
            "peer_tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
            "description": "Major semiconductor stocks vs SMH ETF",
        },
        {
            "group_name": "QQQ vs Tech",
            "target_ticker": "QQQ",
            "peer_tickers": ["SPY", "XLK", "TQQQ", "IWM"],
            "description": "QQQ vs other major indices and tech ETFs",
        },
        {
            "group_name": "Financials vs XLF",
            "target_ticker": "XLF",
            "peer_tickers": ["JPM", "BAC", "WFC", "GS", "MS"],
            "description": "Major banks vs Financial sector ETF",
        },
    ]

    for group in default_groups:
        save_ticker_group(
            group_name=group["group_name"],
            target_ticker=group["target_ticker"],
            peer_tickers=group["peer_tickers"],
            description=group["description"],
            conn=conn,
        )


if __name__ == "__main__":
    # Test the functionality
    from .db_utils import ensure_initialized

    conn = get_conn()
    ensure_initialized(conn)

    # Create default groups
    create_default_groups(conn)

    # List all groups
    groups = list_ticker_groups(conn)
    print("Available ticker groups:")
    for group in groups:
        print(f"  {group['group_name']}: {group['target_ticker']} vs {group['peer_tickers']}")

    conn.close()
