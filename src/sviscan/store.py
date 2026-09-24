"""Deribit public chain fetcher + SQLite alert store."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import httpx
import numpy as np

__all__ = [
    "alert_stats",
    "fetch_alerts",
    "fetch_chain",
    "init_store",
    "load_latest_snapshot",
    "log_alert",
    "resolve_alerts",
    "save_quotes",
]


# ------------------------------------------------------------------ deribit
_BASE = "https://www.deribit.com/api/v2/public"


def fetch_chain(currency: str = "BTC", n_expiries: int = 3) -> list[dict]:
    """Live option quotes for the nearest `n_expiries` maturities.

    ponytail: one ticker call per instrument, serial; concurrent fan-out is a
    drop-in upgrade if scan latency ever matters.
    """
    now_ms = time.time() * 1000
    with httpx.Client(timeout=30) as c:
        instruments = c.get(f"{_BASE}/get_instruments", params={"currency": currency, "kind": "option"}).json()["result"]
        live = [i for i in instruments if i["expiration_timestamp"] >= now_ms + 6 * 3600e3]
        live.sort(key=lambda i: i["expiration_timestamp"])
        # take all strikes of each of the first n_expiries distinct dates
        seen_dates: list[int] = []
        selected = []
        for ins in live:
            d = ins["expiration_timestamp"]
            if len(seen_dates) < n_expiries or d in seen_dates:
                if d not in seen_dates:
                    seen_dates.append(d)
                selected.append(ins)
        out: dict[tuple, dict] = {}
        for ins in selected:
            ticker = c.get(f"{_BASE}/ticker", params={"instrument_name": ins["instrument_name"]}).json()["result"]
            bid_iv, ask_iv = ticker.get("bid_iv"), ticker.get("ask_iv")
            if bid_iv is None or ask_iv is None or bid_iv <= 0 or ask_iv <= 0:
                continue
            # Deribit timestamps are ms; T = years to maturity
            ttm_years = (ins["expiration_timestamp"] / 1000.0 - now_ms / 1000.0) / (365 * 24 * 3600)
            key = (ttm_years, float(ins["strike"]))
            rec = out.get(key)
            if rec is None:
                rec = out[key] = {"T": key[0], "K": key[1],
                                  "iv_bid": bid_iv / 100.0,
                                  "iv_ask": ask_iv / 100.0,
                                  "mark_iv": (ticker.get("mark_iv") or 0) / 100.0,
                                  "spot": float(ticker.get("underlying_price") or np.nan),
                                  "n": 0}
            else:
                # same (expiry, strike) appears for call AND put: average the
                # bid/ask IVs so the mid is a genuine cross-instrument mid
                rec["iv_bid"] += bid_iv / 100.0
                rec["iv_ask"] += ask_iv / 100.0
            rec["n"] += 1
    for rec in out.values():
        if rec["n"] > 1:
            rec["iv_bid"] /= rec["n"]
            rec["iv_ask"] /= rec["n"]
    return list(out.values())


# ------------------------------------------------------------------ storage
_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    ts REAL, T REAL, K REAL, iv_bid REAL, iv_ask REAL, mark_iv REAL, spot REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at REAL, kind TEXT, expiry_short REAL, expiry_long REAL,
    k_min REAL, k_max REAL, severity REAL,
    quote_json TEXT, resolved_at REAL
);
CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes(ts);
"""


def init_store(path: str = "data/scanner.db") -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    # One-time normalization: rows written before single-expiry alerts used
    # NULL stored a literal 0.0 in expiry_long, which `list` then rendered as
    # "T=0.080->0.000". Treat 0 as "no second expiry" — idempotent, and safe
    # because a real expiry_long of exactly 0.0 years is not physically meaningful.
    con.execute("UPDATE alerts SET expiry_long = NULL WHERE expiry_long = 0")
    con.commit()
    return con


def save_quotes(con: sqlite3.Connection, rows: list[dict]) -> int:
    con.executemany(
        "INSERT INTO quotes VALUES (:ts,:T,:K,:iv_bid,:iv_ask,:mark_iv,:spot)",
        [{"ts": time.time(), **{k: r[k] for k in ("T", "K", "iv_bid", "iv_ask", "mark_iv", "spot")}}
         for r in rows],
    )
    con.commit()
    return len(rows)


def load_latest_snapshot(con: sqlite3.Connection, max_age_sec: float = 3600) -> list[dict]:
    cutoff = time.time() - max_age_sec
    cur = con.execute(
        "SELECT T,K,iv_bid,iv_ask,mark_iv,spot FROM quotes WHERE ts > ? ORDER BY T, K",
        (cutoff,))
    return [{"T": t, "K": k, "iv_bid": b, "iv_ask": a, "mark_iv": m, "spot": s}
            for t, k, b, a, m, s in cur.fetchall()]


def log_alert(con: sqlite3.Connection, kind: str, expiry_short: float,
              expiry_long: float, k_min: float, k_max: float,
              severity: float, quote_json: str = "{}") -> int:
    cur = con.execute(
        "INSERT INTO alerts (detected_at,kind,expiry_short,expiry_long,k_min,k_max,severity,quote_json)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (time.time(), kind, expiry_short, expiry_long, k_min, k_max, severity, quote_json))
    con.commit()
    return int(cur.lastrowid)


def resolve_alerts(con: sqlite3.Connection, older_than_sec: float = 86400) -> int:
    """Mark stale alerts resolved; persistence stats come from alert_stats."""
    cutoff = time.time() - older_than_sec
    cur = con.execute("UPDATE alerts SET resolved_at=? WHERE resolved_at IS NULL AND detected_at < ?",
                      (time.time(), cutoff))
    con.commit()
    return cur.rowcount


def alert_stats(con: sqlite3.Connection) -> dict:
    rows = con.execute(
        "SELECT kind, COUNT(*), AVG(resolved_at IS NULL), AVG(severity) FROM alerts GROUP BY kind"
    ).fetchall()
    return {
        kind: {"count": n, "open_share": round(open_share, 3), "avg_severity": round(sev, 4)}
        for kind, n, open_share, sev in rows
    }


def fetch_alerts(con: sqlite3.Connection, kind: str | None = None,
                 unresolved_only: bool = False, limit: int = 20) -> list[dict]:
    """Recent alerts for `sviscan list`; newest first."""
    where, args = [], []
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if unresolved_only:
        where.append("resolved_at IS NULL")
    query = ("SELECT id, detected_at, kind, expiry_short, expiry_long,"
             " k_min, k_max, severity, resolved_at FROM alerts")
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY detected_at DESC LIMIT ?"
    args.append(limit)
    return [dict(row) for row in con.execute(query, args).fetchall()]
