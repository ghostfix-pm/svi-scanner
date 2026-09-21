"""Store layer: SQLite schema, snapshot windowing, alert lifecycle, stats."""

import time

import pytest

from sviscan.store import (
    alert_stats,
    fetch_alerts,
    init_store,
    load_latest_snapshot,
    log_alert,
    resolve_alerts,
    save_quotes,
)


def _quote(T=0.08, K=100.0, bid=0.55, ask=0.56, spot=100.0, mark=0.555):
    return {"T": T, "K": K, "iv_bid": bid, "iv_ask": ask, "spot": spot, "mark_iv": mark}


def test_init_store_creates_schema(tmp_path):
    db = tmp_path / "nested" / "scanner.db"
    con = init_store(str(db))
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"quotes", "alerts"} <= tables
    idx = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    assert "idx_quotes_ts" in idx
    con.close()


def test_save_quotes_roundtrip(tmp_path):
    con = init_store(str(tmp_path / "s.db"))
    rows = [_quote(T=0.08, K=100.0), _quote(T=0.08, K=110.0), _quote(T=0.16, K=100.0)]
    assert save_quotes(con, rows) == 3
    got = con.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    assert got == 3
    con.close()


def test_load_latest_snapshot_filters_by_age_and_keeps_spot(tmp_path):
    con = init_store(str(tmp_path / "s.db"))
    save_quotes(con, [_quote(T=0.08, K=100.0, spot=100.5)])
    con.execute("UPDATE quotes SET ts = ?", (time.time() - 7200,))
    con.commit()
    assert load_latest_snapshot(con) == []
    save_quotes(con, [_quote(T=0.08, K=110.0, spot=101.0, mark=0.56)])
    snap = load_latest_snapshot(con)
    assert len(snap) == 1
    assert snap[0]["spot"] == 101.0
    assert snap[0]["mark_iv"] == 0.56
    assert snap[0]["iv_bid"] == 0.55
    con.close()


def test_log_alert_returns_id_and_stores_fields(tmp_path):
    con = init_store(str(tmp_path / "s.db"))
    aid = log_alert(con, "butterfly", 0.08, 0.0, -0.2, 0.1, 0.0042)
    row = con.execute("SELECT * FROM alerts WHERE id=?", (aid,)).fetchone()
    assert row["kind"] == "butterfly"
    assert row["expiry_short"] == 0.08
    assert row["severity"] == pytest.approx(0.0042)
    assert row["resolved_at"] is None
    con.close()


def test_resolve_alerts_only_marks_stale(tmp_path):
    con = init_store(str(tmp_path / "s.db"))
    old = log_alert(con, "calendar", 0.08, 0.16, -0.1, 0.1, 0.001)
    con.execute("UPDATE alerts SET detected_at = ? WHERE id = ?",
                (time.time() - 2 * 86400, old))
    fresh = log_alert(con, "butterfly", 0.08, 0.0, -0.2, 0.2, 0.002)
    con.commit()
    assert resolve_alerts(con, older_than_sec=86400) == 1
    states = dict(con.execute("SELECT id, resolved_at IS NULL FROM alerts").fetchall())
    assert states[old] == 0  # stale -> resolved
    assert states[fresh] == 1  # fresh stays open
    con.close()


def test_alert_stats_grouping_and_empty_db(tmp_path):
    con = init_store(str(tmp_path / "s.db"))
    assert alert_stats(con) == {}
    log_alert(con, "butterfly", 0.08, 0.0, -0.2, 0.1, 0.01)
    log_alert(con, "butterfly", 0.08, 0.0, -0.2, 0.1, 0.03)
    log_alert(con, "calendar", 0.08, 0.16, 0.0, 0.1, 0.05)
    stats = alert_stats(con)
    assert stats["butterfly"]["count"] == 2
    assert stats["butterfly"]["open_share"] == 1.0
    assert stats["butterfly"]["avg_severity"] == pytest.approx(0.02)
    assert stats["calendar"]["count"] == 1
    con.close()


def test_fetch_alerts_order_filters_and_limit(tmp_path):
    con = init_store(str(tmp_path / "s.db"))
    b = log_alert(con, "butterfly", 0.08, 0.0, -0.2, 0.1, 0.01)
    c = log_alert(con, "calendar", 0.08, 0.16, 0.0, 0.1, 0.02)
    con.execute("UPDATE alerts SET resolved_at = ? WHERE id = ?", (time.time(), b))
    con.commit()
    all_ = fetch_alerts(con, limit=10)
    assert [a["id"] for a in all_] == [c, b]  # newest first
    kinds = fetch_alerts(con, kind="butterfly", limit=10)
    assert [a["id"] for a in kinds] == [b]
    open_only = fetch_alerts(con, unresolved_only=True, limit=10)
    assert [a["id"] for a in open_only] == [c]
    limited = fetch_alerts(con, limit=1)
    assert len(limited) == 1
    con.close()