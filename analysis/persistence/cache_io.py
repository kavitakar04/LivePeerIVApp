"""Canonical persistent calculation cache.

Cache ownership:
- Backend: SQLite database at ``data/calculations.db`` by default.
- Key: SHA-256 of ``kind``, per-kind artifact version, and stable JSON payload.
- Value: pickled artifact compressed with zlib.
- Lifecycle: entries expire after ``TTL_SEC`` seconds and expired rows are
  pruned opportunistically on writes.

Legacy raw-marker helpers ``save_calc_cache`` and ``load_calc_cache`` remain
for tests and old callers, but new calculation artifacts should use
``compute_or_load``.
"""

import json
import pickle
import sqlite3
import zlib
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import queue
import threading
import time

from analysis.config.settings import DEFAULT_CALC_CACHE_TTL_SEC

TTL_SEC = DEFAULT_CALC_CACHE_TTL_SEC
ARTIFACT_VERSION: Dict[str, str] = {}
DEFAULT_CALC_CACHE_DB_PATH = "data/calculations.db"


def _latest_raw_marker(conn: sqlite3.Connection) -> str | None:
    """Return the latest ``asof_date`` across raw data tables if present."""
    dates = []
    for tbl in ("options_quotes", "underlying_prices"):
        try:
            row = conn.execute(f"SELECT MAX(asof_date) FROM {tbl}").fetchone()
            if row and row[0]:
                dates.append(row[0])
        except sqlite3.Error:
            pass
    return max(dates) if dates else None


def save_calc_cache(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Persist a calculation cache entry tied to current raw data state."""
    conn.execute("""CREATE TABLE IF NOT EXISTS calc_cache(
           key TEXT PRIMARY KEY,
           value BLOB,
           created_at INTEGER NOT NULL,
           raw_marker TEXT)""")
    blob = sqlite3.Binary(pickle.dumps(value))
    marker = _latest_raw_marker(conn)
    conn.execute(
        "REPLACE INTO calc_cache(key,value,created_at,raw_marker) VALUES(?,?,strftime('%s','now'),?)",
        (key, blob, marker),
    )
    conn.commit()


def load_calc_cache(conn: sqlite3.Connection, key: str) -> Any | None:
    """Load a cached artifact if raw data has not changed since saving."""
    row = conn.execute("SELECT value, raw_marker FROM calc_cache WHERE key=?", (key,)).fetchone()
    if not row:
        return None
    value_blob, marker = row
    if marker != _latest_raw_marker(conn):
        return None
    try:
        return pickle.loads(value_blob)
    except Exception:
        return None


def _hash_inputs(kind: str, payload: dict) -> str:
    """Create a stable hash for the inputs.

    The hash combines the kind, the artifact version for that kind, and the payload
    dictionary serialized in a deterministic manner.
    """
    version = ARTIFACT_VERSION.get(kind, "1")
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    hasher = hashlib.sha256()
    hasher.update(kind.encode())
    hasher.update(b"|")
    hasher.update(version.encode())
    hasher.update(b"|")
    hasher.update(payload_json.encode())
    return hasher.hexdigest()


def compute_or_load(
    kind: str,
    payload: dict,
    builder_fn: Callable[[], Any],
    db_path: str = DEFAULT_CALC_CACHE_DB_PATH,
) -> Any:
    """Compute an artifact or load it from cache.

    Parameters
    ----------
    kind : str
        Identifier for the artifact type.
    payload : dict
        Inputs used to build the artifact. Must be JSON serializable.
    builder_fn : Callable[[], Any]
        Function that builds the artifact when cache miss occurs.
    db_path : str
        Path to the SQLite database file used for caching. Defaults to the
        canonical calculations DB.
    """
    key = _hash_inputs(kind, payload)
    now = int(time.time())
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS calc_cache(
  key TEXT PRIMARY KEY,
  artifact BLOB,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  kind TEXT NOT NULL,
  version TEXT NOT NULL,
  meta_json TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS calc_cache_expires ON calc_cache(expires_at)")
        row = conn.execute("SELECT artifact, expires_at FROM calc_cache WHERE key=?", (key,)).fetchone()
        if row and row[1] > now:
            try:
                return pickle.loads(zlib.decompress(row[0]))
            except Exception:
                pass
        artifact = builder_fn()
        blob = zlib.compress(pickle.dumps(artifact))
        expires_at = now + TTL_SEC
        version = ARTIFACT_VERSION.get(kind, "1")
        meta_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with conn:
            conn.execute(
                "REPLACE INTO calc_cache"
                "(key, artifact, created_at, expires_at, kind, version, meta_json) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (key, blob, now, expires_at, kind, version, meta_json),
            )
            conn.execute("DELETE FROM calc_cache WHERE expires_at <= ?", (now,))
        return artifact
    finally:
        conn.close()


class WarmupWorker:
    """Simple background worker to warm caches lazily.

    Jobs are enqueued via :meth:`enqueue` and processed sequentially by a
    dedicated daemon thread. Each job is described by a ``kind`` string, an
    arbitrary ``payload`` object and a ``builder_fn`` callable. The callable is
    executed through :func:`compute_or_load` which is responsible for building
    and populating the cache.
    """

    def __init__(self, db_path: str = "data/calculations.db"):
        self.db_path = db_path
        self._queue: "queue.Queue[Tuple[str, Any, Callable[[], Any]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, kind: str, payload: Any, builder_fn: Callable[[], Any]) -> None:
        """Add a warmup job to the queue."""
        self._queue.put((kind, payload, builder_fn))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            kind, payload, builder_fn = self._queue.get()
            try:
                compute_or_load(kind, payload, builder_fn, self.db_path)
            except Exception:
                # Never let cache warmup failures impact the GUI thread.
                pass
            finally:
                # Mark job done and briefly yield control to keep UI responsive.
                self._queue.task_done()
                time.sleep(0.01)
