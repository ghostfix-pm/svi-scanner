"""CLI: `sviscan scan` runs one detection cycle; `sviscan report`/`list` summarize alerts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .scanner import build_slices, scan_surface
from .store import (
    alert_stats,
    fetch_alerts,
    fetch_chain,
    init_store,
    load_latest_snapshot,
    log_alert,
    resolve_alerts,
    save_quotes,
)


def _scan(args):
    quotes = fetch_chain(args.currency) if not args.offline else []
    con = init_store(args.db)
    if quotes:
        n = save_quotes(con, quotes)
        print(f"fetched {n} option quotes from Deribit ({args.currency})",
              file=sys.stderr if args.json else None)
    else:
        quotes = load_latest_snapshot(con)
        print(f"offline: using {len(quotes)} cached quotes",
              file=sys.stderr if args.json else None)

    slices = build_slices(quotes)
    print(f"built {len(slices)} expiry slices",
          file=sys.stderr if args.json else None)
    alerts = scan_surface(slices, rmse_gate=args.gate)
    jsonl_path = Path(args.db).parent / "alerts.jsonl"

    # Single write path for both modes: log + append the jsonl record once, and
    # additionally print human lines when not in --json mode. (Previously the
    # --json branch duplicated this whole loop.)
    verbose = not args.json
    rows = []
    with open(jsonl_path, "a") as jl:
        for a in alerts:
            aid = log_alert(con, a["kind"], a["T_short"], a["T_long"],
                            a["k_min"], a["k_max"], a["severity"])
            rec = {"id": aid, "detected_at": time.time(), **a}
            jl.write(json.dumps(rec) + "\n")
            rows.append(rec)
            if verbose:
                print(f"[{aid}] {a['kind']}: T={a['T_long'] or a['T_short']:.3f} "
                      f"k=[{a['k_min']:.2f},{a['k_max']:.2f}] severity={a['severity']:.5f}")
    if verbose and not alerts:
        print("no violations detected")
    resolve_alerts(con, older_than_sec=7 * 86400)

    if args.json:
        out = {
            "currency": args.currency,
            "offline": args.offline,
            "n_quotes": len(quotes),
            "n_slices": len(slices),
            "slices": [{"T": t, "n": sl["n"], "rmse": round(sl["rmse"], 6),
                        "params": [round(float(p), 6) for p in sl["params"]]}
                       for t, sl in sorted(slices.items())],
            "alerts": rows,
        }
        print(json.dumps(out, indent=2))
        return


def _report(args):
    con = init_store(args.db)
    print(json.dumps(alert_stats(con), indent=2))


def _list(args):
    con = init_store(args.db)
    alerts = fetch_alerts(con, kind=args.kind, unresolved_only=args.unresolved,
                          limit=args.limit)
    if args.json:
        print(json.dumps(alerts, indent=2))
        return
    if not alerts:
        print("no alerts")
        return
    for a in alerts:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(a["detected_at"]))
        expiry = f"T={a['expiry_short']:.3f}" if a["expiry_long"] is None \
            else f"T={a['expiry_short']:.3f}->{a['expiry_long']:.3f}"
        state = "open " if a["resolved_at"] is None else "done "
        print(f"[{a['id']:>5}] {state} {when} {a['kind']:9s} {expiry} "
              f"k=[{a['k_min']:.2f},{a['k_max']:.2f}] sev={a['severity']:.5f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sviscan", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="fetch chain, fit slices, detect arbs, log")
    s.add_argument("--currency", default="BTC")
    s.add_argument("--db", default="data/scanner.db")
    s.add_argument("--gate", type=float, default=0.004)
    s.add_argument("--offline", action="store_true", help="reuse latest cached snapshot")
    s.add_argument("--json", action="store_true",
                   help="machine-readable one-cycle summary (slices + alerts)")
    s.set_defaults(func=_scan)

    r = sub.add_parser("report", help="alert persistence stats (JSON)")
    r.add_argument("--db", default="data/scanner.db")
    r.set_defaults(func=_report)

    ls = sub.add_parser("list", help="recent alerts, newest first")
    ls.add_argument("--db", default="data/scanner.db")
    ls.add_argument("--kind", default=None, help="filter: butterfly|calendar|bad_fit")
    ls.add_argument("--unresolved", action="store_true", help="open alerts only")
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_list)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
