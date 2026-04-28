import sqlite3
import analysis.persistence.cache_io as cache_io
from analysis.persistence.cache_io import save_calc_cache, load_calc_cache, compute_or_load


def _setup_db():
    conn = sqlite3.connect(':memory:')
    conn.executescript(
        """
        CREATE TABLE options_quotes(asof_date TEXT);
        CREATE TABLE underlying_prices(asof_date TEXT);
        """
    )
    return conn


def test_cache_reuse_when_raw_unchanged():
    conn = _setup_db()
    conn.execute("INSERT INTO options_quotes(asof_date) VALUES('2000-01-01')")
    save_calc_cache(conn, 'k', {'v': 1})
    loaded = load_calc_cache(conn, 'k')
    assert loaded == {'v': 1}


def test_cache_invalidated_on_raw_update():
    conn = _setup_db()
    conn.execute("INSERT INTO options_quotes(asof_date) VALUES('1999-01-01')")
    save_calc_cache(conn, 'k', {'v': 1})
    # Force created_at to be old
    conn.execute("UPDATE calc_cache SET created_at = 0")
    # Later raw data arrives
    conn.execute("INSERT INTO options_quotes(asof_date) VALUES('2000-01-01')")
    conn.commit()
    assert load_calc_cache(conn, 'k') is None


def test_compute_or_load_reuses_unexpired_artifact(tmp_path):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return {"v": calls["n"]}

    db_path = tmp_path / "calculations.db"
    assert compute_or_load("kind", {"x": 1}, builder, db_path=str(db_path)) == {"v": 1}
    assert compute_or_load("kind", {"x": 1}, builder, db_path=str(db_path)) == {"v": 1}
    assert calls["n"] == 1


def test_compute_or_load_rebuilds_expired_artifact(tmp_path, monkeypatch):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return {"v": calls["n"]}

    db_path = tmp_path / "calculations.db"
    monkeypatch.setattr(cache_io, "TTL_SEC", 1)
    monkeypatch.setattr(cache_io.time, "time", lambda: 1000)
    assert compute_or_load("kind", {"x": 1}, builder, db_path=str(db_path)) == {"v": 1}

    monkeypatch.setattr(cache_io.time, "time", lambda: 1002)
    assert compute_or_load("kind", {"x": 1}, builder, db_path=str(db_path)) == {"v": 2}
    assert calls["n"] == 2
