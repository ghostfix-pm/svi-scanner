"""Calibration gate (marked `slow`, non-blocking in CI).

These tests exercise the detector against a *calibrated* synthetic surface across
several maturities and assert the published default gate behaves as documented:

  * a genuinely arbitrage-free surface produces ZERO hard alerts at the default
    `rmse_gate`, and
  * an injected calendar-arbitrage (total variance falling with maturity) IS
    detected.

They are heavier than the unit tests (a dense strike grid across four
maturities, each slice fitted), which is why they carry the `slow` marker and
run as a separate, non-blocking CI step. Without this file, CI's
`pytest -m slow` matched nothing and exited 5 every run.

Note on injection choice: a *butterfly* bump is a poor calibration probe here
because the raw-SVI fit is flexible enough to absorb a local convexity bump on
a smooth smile (verified: g(k) stayed positive for bumps up to 60% deep). A
calendar inversion is structural — it cannot be fitted away — so it makes the
deterministic detector assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from sviscan.scanner import build_slices, scan_surface

pytestmark = pytest.mark.slow

SPOT = 100.0
EXPIRIES = (7 * 86400, 30 * 86400, 90 * 86400, 180 * 86400)
N_STRIKES = 41


def _surface_rows(invert_longest: bool = False) -> list[dict]:
    """A dense, smooth smile across four maturities.

    With `invert_longest`, the longest-dated slice is given a much lower level
    of implied vol, so total variance drops as maturity rises -> a calendar
    (total-variance monotonicity) violation the scanner must catch.
    """
    rows = []
    for idx, T_sec in enumerate(EXPIRIES):
        T = T_sec / (365.0 * 24.0 * 3600.0)
        base_iv = 0.5 + 0.05 * idx
        if invert_longest and T_sec == EXPIRIES[-1]:
            base_iv = 0.28
        for strike in np.linspace(60, 160, N_STRIKES):
            k = np.log(strike / SPOT)
            iv = base_iv * (1 - 0.35 * k + 1.4 * k * k)
            rows.append({
                "T": T, "K": float(strike),
                "iv_bid": max(iv - 0.01, 1e-3), "iv_ask": iv + 0.01,
                "spot": SPOT, "mark_iv": iv,
            })
    return rows


def test_calibration_clean_surface_is_arb_free_at_default_gate():
    """A calibrated clean surface must not trip the default rmse gate."""
    slices = build_slices(_surface_rows())
    assert len(slices) == len(EXPIRIES)

    alerts = scan_surface(slices, rmse_gate=0.004)
    hard = [a for a in alerts if a["kind"] in ("butterfly", "calendar")]
    assert hard == [], f"clean surface produced hard alerts: {hard}"


def test_calibration_detects_injected_calendar_violation():
    """Total variance must not fall with maturity; the scanner must flag it."""
    slices = build_slices(_surface_rows(invert_longest=True))
    alerts = scan_surface(slices, rmse_gate=0.004)
    kinds = [a["kind"] for a in alerts]
    assert "calendar" in kinds, f"injected calendar arb not detected; got {kinds}"
