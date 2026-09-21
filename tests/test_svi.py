import numpy as np
import pytest
from scipy.stats import norm

from sviscan.scanner import build_slices, scan_surface
from sviscan.svi import (
    butterfly_ok,
    calendar_ok,
    fit_slice,
    g_function,
    svi_w,
)


def arb_free_params(atm_w: float = 0.04) -> np.ndarray:
    # b small, rho mild: known-clean surface shape for short horizons
    return np.array([atm_w * 0.8, atm_w * 0.4, -0.3, 0.0, 0.15])


# ------------------------------------------------------------------ fitting
def test_svi_fit_recovers_synthetic_params():
    true = np.array([0.032, 0.08, -0.55, 0.02, 0.12])
    ks = np.linspace(-0.3, 0.3, 21)
    ws = svi_w(ks, true)
    params, rmse = fit_slice(ks, ws)
    assert rmse < 1e-6
    assert np.allclose(params, true, rtol=0.05, atol=0.005)


def test_fit_is_deterministic_for_fixed_input():
    ks = np.linspace(-0.25, 0.25, 15)
    w = svi_w(ks, arb_free_params())
    p1, _ = fit_slice(ks, w)
    p2, _ = fit_slice(ks, w)
    assert np.allclose(p1, p2)


# ------------------------------------------------------------------ arbitrage
def test_butterfly_detector_passes_clean_and_flags_arb():
    clean = arb_free_params()
    assert butterfly_ok(clean, (-0.5, 0.5))
    assert (g_function(np.linspace(-0.5, 0.5, 101), clean) > -1e-12).all()

    # construct an arbed slice: negative total-variance curvature at the wings
    arbed = np.array([0.05, 0.9, 0.0, 0.0, 0.01])  # near-flat core, tiny sigma
    assert not butterfly_ok(arbed, (-0.5, 0.5))


def test_calendar_detector_flags_variance_decrease():
    kgrid = np.linspace(-0.3, 0.3, 61)
    short = np.array([0.020, 0.05, -0.3, 0.0, 0.15])
    long_ok = np.array([0.030, 0.05, -0.3, 0.0, 0.15])   # strictly above -> fine
    long_bad = np.array([0.010, 0.05, -0.3, 0.0, 0.15])  # below short -> calendar arb
    assert calendar_ok(short, long_ok, (-0.3, 0.3))
    assert not calendar_ok(short, long_bad, (-0.3, 0.3))
    from sviscan.svi import find_calendar_violations

    flagged = find_calendar_violations(short, long_bad, kgrid)
    assert flagged.size == kgrid.size


# ------------------------------------------------------------------ scanner
def _synthetic_quotes() -> list[dict]:
    """Two clean expiries with a smooth smile; spot=100."""
    quotes = []
    spot = 100.0
    base_iv = {30 * 86400: 0.55, 90 * 86400: 0.60}
    for T_sec, iv0 in base_iv.items():
        T = T_sec / 365.0 / 24.0  # ~years-ish scale irrelevant for consistency
        for strike in np.linspace(70, 130, 13):
            k = np.log(strike / spot)
            iv = iv0 * (1 - 0.35 * k + 1.2 * k * k)
            quotes.append({"T": T, "K": float(strike),
                           "iv_bid": max(iv - 0.01, 1e-3),
                           "iv_ask": iv + 0.01,
                           "spot": spot})
    return quotes


def test_scanner_reports_no_false_positives_on_clean_surface():
    slices = build_slices(_synthetic_quotes())
    assert len(slices) == 2
    alerts = scan_surface(slices)
    hard = [a for a in alerts if a["kind"] in ("butterfly", "calendar")]
    assert hard == [], f"clean surface must not alert: {hard}"


def test_scanner_skips_bad_fits_with_explicit_alert():
    quotes = _synthetic_quotes()
    # corrupt one expiry so its fit is garbage
    for q in quotes:
        if q["T"] < 0.002:
            q["iv_bid"], q["iv_ask"] = q["iv_bid"] * 3, q["iv_ask"] * 3 + 0.05
    slices = build_slices(quotes)
    alerts = scan_surface(slices, rmse_gate=1e-4)
    kinds = {a["kind"] for a in alerts}
    assert "bad_fit" in kinds


# ------------------------------------------------------------------ oracle
# Independent correctness gate: no-butterfly-arbitrage must agree with the true
# risk-neutral density from Breeden-Litzenberger (g_function is a closed-form
# rearrangement; the oracle below is finite differences on call prices).

_F = 100.0


def _call_price(K, w):
    k = np.log(K / _F)
    d1 = -k / np.sqrt(w) + np.sqrt(w) / 2.0
    return _F * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(w))


def _density_fd(params, ks, h=1e-3):
    """q(k) = K * d2C/dK2 via central differences; positive iff no arb."""
    q = np.empty_like(ks)
    for i, k in enumerate(ks):
        K = _F * np.exp(k)
        lo, hi = K * (1 - h), K * (1 + h)
        q[i] = K * (_call_price(lo, svi_w(np.log(lo / _F), params))
                    - 2 * _call_price(K, svi_w(k, params))
                    + _call_price(hi, svi_w(np.log(hi / _F), params))) / ((K * h) ** 2)
    return q


@pytest.mark.parametrize("params", [
    np.array([0.032, 0.08, -0.55, 0.02, 0.12]),            # clean fitted slice
    np.array([0.02, 0.9, 0.1, 0.05, 0.03]),                 # sharp smile, arbed
    np.array([0.05, 0.9, 0.0, 0.0, 0.01]),                  # repo arbed example
    np.array([0.04, 0.9, -0.6, -0.05, 0.02]),               # steep asymmetric
])
def test_g_function_sign_matches_density_oracle(params):
    ks = np.linspace(-0.5, 0.5, 201)
    g = g_function(ks, params)
    q = _density_fd(params, ks)
    # g must never claim a violation where the true density is positive
    assert not np.any((g < 0) & (q > 1e-9)), "g_function false positive"
    # and must flag (as arbed) most of what the density flags
    flagged = q < -1e-9
    if flagged.any():
        assert np.sum(g < 0) >= 0.8 * np.sum(flagged), "g_function misses real arbs"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
