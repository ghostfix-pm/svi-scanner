"""build_slices robustness: unusable-spot quotes must be skipped, not k=0."""

import numpy as np

from sviscan.scanner import build_slices


def _quotes(n=11, T=0.08, spot=100.0, bad=None):
    rows = []
    bad = bad or set()
    for i, strike in enumerate([70 + i * 6 for i in range(n)]):
        k = np.log(strike / spot)
        iv = 0.55 * (1 - 0.35 * k + 1.2 * k * k)
        rows.append({"T": T, "K": float(strike), "iv_bid": iv - 0.01,
                     "iv_ask": iv + 0.01, "spot": spot if i not in bad else np.nan})
    return rows


def test_min_strikes_gate_drops_skinny_slices():
    assert build_slices(_quotes(n=4)) == {}


def test_nan_spot_rows_are_skipped_not_zeroed():
    """Regression: a nan-spot row used to become k=0 and poison the fit."""
    rows = _quotes(n=11, bad={3, 5})
    slices = build_slices(rows)
    assert len(slices) == 1
    sl = slices[0.08]
    assert sl["n"] == 9
    # all fitted ks must be strictly increasing (no duplicated k=0 point)
    ks = sl["ks"]
    assert np.all(np.diff(ks) > 0)
    assert ks[0] < 0 < ks[-1]


def test_all_invalid_spot_drops_slice():
    assert build_slices(_quotes(n=11, bad=set(range(11)))) == {}


def test_zero_or_negative_strike_skipped():
    rows = _quotes(n=11)
    rows[0]["K"] = 0.0
    rows[1]["K"] = -5.0
    slices = build_slices(rows)
    assert len(slices) == 1
    assert slices[0.08]["n"] == 9


def test_multiple_expiries_and_spot_per_row_variation():
    rows = _quotes(n=11, T=0.08)
    rows += _quotes(n=11, T=0.16, spot=99.0)
    slices = build_slices(rows)
    assert set(slices) == {0.08, 0.16}
