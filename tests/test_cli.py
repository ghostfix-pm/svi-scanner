"""CLI: offline scan cycle, --json summary, report, list."""

import json
import time

from sviscan.cli import main
from sviscan.store import init_store, save_quotes


def _seed_db(db_path, n_strikes=11):
    """Two clean expiries with a smooth smile; spot=100 (same shape as test_svi)."""
    rows = []
    spot = 100.0
    for T_sec, iv0 in ((30 * 86400, 0.55), (90 * 86400, 0.60)):
        T = T_sec / 365.0 / 24.0
        for strike in [70 + i * 6 for i in range(n_strikes)]:
            k = (strike / spot - 1.0)  # rough moneyness proxy is fine for CLI seeding
            iv = iv0 * (1 - 0.3 * k + 1.0 * k * k)
            rows.append({"T": T, "K": float(strike),
                         "iv_bid": max(iv - 0.01, 1e-3), "iv_ask": iv + 0.01,
                         "spot": spot, "mark_iv": iv})
    con = init_store(str(db_path))
    save_quotes(con, rows)
    con.close()


def test_scan_offline_empty_db_is_graceful(tmp_path, capsys):
    db = tmp_path / "scan.db"
    main(["scan", "--offline", "--db", str(db)])
    out = capsys.readouterr().out
    assert "no violations detected" in out
    assert "offline: using 0 cached quotes" in out


def test_scan_offline_seeded_db_writes_alerts_jsonl(tmp_path, capsys):
    db = tmp_path / "scan.db"
    _seed_db(db)
    main(["scan", "--offline", "--db", str(db), "--gate", "1e-6"])
    out = capsys.readouterr().out
    assert "built 2 expiry slices" in out
    jl = tmp_path / "alerts.jsonl"
    assert jl.exists()
    for line in jl.read_text().strip().splitlines():
        rec = json.loads(line)
        assert {"id", "kind", "T_short", "T_long", "k_min", "k_max", "severity"} <= set(rec)


def test_scan_json_output_is_pure_json(tmp_path, capsys):
    db = tmp_path / "scan.db"
    _seed_db(db)
    main(["scan", "--offline", "--db", str(db), "--json"])
    captured = capsys.readouterr()
    doc = json.loads(captured.out)  # stdout must parse as one JSON doc
    assert captured.err  # status lines went to stderr
    assert doc["offline"] is True
    assert doc["n_slices"] == 2
    assert doc["currency"] == "BTC"
    assert {"alerts", "slices"} <= set(doc)
    assert doc["slices"][0]["n"] >= 5
    assert {"T", "n", "rmse", "params"} <= set(doc["slices"][0])


def test_report_prints_stats_json(tmp_path, capsys, monkeypatch):
    db = tmp_path / "scan.db"
    con = init_store(str(db))
    con.execute("INSERT INTO alerts (detected_at,kind,expiry_short,expiry_long,"
                "k_min,k_max,severity) VALUES (?,?,?,?,?,?,?)",
                (time.time(), "butterfly", 0.08, 0.0, -0.2, 0.1, 0.01))
    con.commit()
    con.close()
    main(["report", "--db", str(db)])
    stats = json.loads(capsys.readouterr().out)
    assert stats["butterfly"]["count"] == 1
    assert stats["butterfly"]["open_share"] == 1.0


def test_list_human_and_json(tmp_path, capsys):
    db = tmp_path / "scan.db"
    con = init_store(str(db))
    con.execute("INSERT INTO alerts (detected_at,kind,expiry_short,expiry_long,"
                "k_min,k_max,severity) VALUES (?,?,?,?,?,?,?)",
                (time.time(), "butterfly", 0.08, 0.0, -0.2, 0.1, 0.01))
    con.commit()
    con.close()

    main(["list", "--db", str(db)])
    out = capsys.readouterr().out
    assert "butterfly" in out
    assert "T=0.080" in out

    main(["list", "--db", str(db), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc[0]["kind"] == "butterfly"
    assert doc[0]["resolved_at"] is None

    main(["list", "--db", str(db), "--kind", "calendar"])
    assert capsys.readouterr().out.strip() == "no alerts"

    main(["list", "--db", str(db), "--limit", "0"])
    assert capsys.readouterr().out.strip() == "no alerts"


def test_list_unresolved_filter(tmp_path, capsys):
    db = tmp_path / "scan.db"
    con = init_store(str(db))
    con.execute("INSERT INTO alerts (detected_at,kind,expiry_short,expiry_long,"
                "k_min,k_max,severity) VALUES (?,?,?,?,?,?,?)",
                (time.time(), "butterfly", 0.08, 0.0, -0.2, 0.1, 0.01))
    con.execute("INSERT INTO alerts (detected_at,kind,expiry_short,expiry_long,"
                "k_min,k_max,severity,resolved_at) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), "calendar", 0.08, 0.16, 0.0, 0.1, 0.02, time.time()))
    con.commit()
    con.close()
    main(["list", "--db", str(db), "--unresolved", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert [a["kind"] for a in doc] == ["butterfly"]