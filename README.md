# svi-scanner

**SVI volatility surfaces** from live Deribit option chains, scanned for
no-arbitrage violations and logged over time:

- per-expiry raw-SVI slice fits: `w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))`
- **butterfly** check via the Gatheral–Jacquier `g(k) >= 0` condition
- **calendar** check: total variance non-decreasing in maturity at fixed k
- bad fits are gated (`rmse > gate` → explicit `bad_fit` alert, never silently scanned)
- every quote snapshot + alert persisted in SQLite; weekly resolution job;
  persistence statistics via `sviscan report`

## Quickstart

```bash
pip install -e ".[dev]"
make test
python -m sviscan.cli scan                  # live Deribit BTC chain
python -m sviscan.cli scan --offline        # re-scan last cached snapshot
python -m sviscan.cli report                # alert counts / open share / severity
python -m sviscan.cli list                  # recent alerts, newest first
```

Schedule it (systemd timer or GH Actions cron) to accumulate weeks of alerts,
then read `report` / `list` as your findings memo.

## CLI

| command | what it does |
|---|---|
| `sviscan scan [--currency BTC] [--db data/scanner.db] [--gate 0.004]` | fetch chain, fit slices, detect arbs, log to SQLite + `alerts.jsonl` |
| `sviscan scan --offline` | re-scan the newest cached snapshot (no network; requires a snapshot from the last hour) |
| `sviscan scan --json` | machine-readable one-cycle summary: per-slice params/rmse + fresh alerts as a single JSON doc on stdout (status lines go to stderr) |
| `sviscan report [--db ...]` | persistence stats as JSON: count / open share / avg severity per kind |
| `sviscan list [--kind butterfly\|calendar\|bad_fit] [--unresolved] [--limit N] [--json]` | recent alerts, newest first |

The cached snapshot keeps each quote's `spot` and `mark_iv`, so offline scans
fit slices with real log-moneyness — never degenerate k=0 points.

## Correctness gates (enforced in tests)

- SVI fit recovers synthetic parameters to <5% (RMSE < 1e-6)
- Clean surface passes `g >= 0` everywhere; deliberately-constructed arbed
  slice is flagged
- Calendar detector flags variance decrease across all strikes, accepts increase
- A clean two-expiry synthetic surface produces **zero** hard alerts
- Corrupted quotes trip the RMSE gate instead of producing phantom arbs
- `g_function` sign is cross-checked against the true Breeden–Litzenberger
  density (finite differences on call prices) on clean and arbed slices —
  no false positives, no missed arb regions

## Automation (accumulating history)

The repo ships two schedulers for the same scan cycle:

- **GitHub Actions** (`.github/workflows/scheduled-scan.yml`): every 6h, runs a
  live scan and commits `data/alerts.jsonl` back to the repo. Free, no server.
- **launchd** (`dev/com.hemang.sviscan.plist`): local timer, same cadence.

After a few weeks, `sviscan report` is your findings memo: violation counts by
type, open share (persistence), average severity.

## Honest scope

- `ponytail:` raw SVI per slice; eSSVI/joint-surface fitting is the upgrade when
  calendar violations between *fits* need finer separation from fit noise.
- Severity is currently magnitude-based (min g, max w-gap); P&L attribution of
  each signal comes after enough logged history exists.
