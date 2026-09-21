"""Surface construction + arbitrage scanning over fetched quotes."""

from __future__ import annotations

import itertools

import numpy as np

from .svi import (
    find_butterfly_violations,
    find_calendar_violations,
    fit_slice,
    g_function,
    svi_w,
)

__all__ = ["build_slices", "scan_surface"]


def build_slices(quotes: list[dict], min_strikes: int = 5) -> dict[float, dict]:
    """Group quotes by expiry; use mid IV for fitting, keep raw for severity.

    Quotes without a usable spot are skipped (log-moneyness is undefined)
    rather than silently collapsed to k=0 -- a poisoned row otherwise fits a
    constant and either trips bad_fit or produces a phantom arb.
    """
    by_exp: dict[float, list[dict]] = {}
    for q in quotes:
        by_exp.setdefault(round(q["T"], 6), []).append(q)
    slices = {}
    for T, rows in sorted(by_exp.items()):
        usable = [r for r in rows
                  if np.isfinite(r.get("spot", np.nan)) and r.get("spot", 0) > 0
                  and np.isfinite(r.get("K", np.nan)) and r.get("K", 0) > 0]
        if len(usable) < min_strikes:
            continue
        spot = float(np.mean([r["spot"] for r in usable]))
        ks, mids = [], []
        for r in sorted(usable, key=lambda r: r["K"]):
            ks.append(np.log(r["K"] / spot))
            mids.append((r["iv_bid"] + r["iv_ask"]) / 2.0)
        ks_arr = np.asarray(ks)
        w = (np.asarray(mids) ** 2) * T  # total variance
        params, rmse = fit_slice(ks_arr, w)
        slices[T] = {"params": params, "rmse": rmse, "ks": ks_arr,
                     "w": w, "n": len(usable)}
    return slices


def scan_surface(slices: dict[float, dict], rmse_gate: float = 0.004,
                 k_span: tuple[float, float] = (-0.35, 0.35)) -> list[dict]:
    """Detect butterfly and calendar violations on the fitted surface.

    Bad fits (rmse above gate) are reported and skipped -- never silently scanned.
    """
    alerts: list[dict] = []
    kgrid = np.linspace(*k_span, 141)

    for T, sl in sorted(slices.items()):
        if sl["rmse"] > rmse_gate:
            alerts.append({"kind": "bad_fit", "T_short": T, "T_long": None,
                           "k_min": float(sl["ks"].min()), "k_max": float(sl["ks"].max()),
                           "severity": sl["rmse"]})
            continue
        bad_k = find_butterfly_violations(sl["params"], kgrid)
        if bad_k.size:
            g_min = float(g_function(kgrid, sl["params"]).min())
            alerts.append({"kind": "butterfly", "T_short": T, "T_long": None,
                           "k_min": float(bad_k.min()), "k_max": float(bad_k.max()),
                           "severity": -g_min})

    expiries = sorted(slices)
    for t1, t2 in itertools.pairwise(expiries):
        s1, s2 = slices[t1], slices[t2]
        if s1["rmse"] > rmse_gate or s2["rmse"] > rmse_gate:
            continue
        bad_k = find_calendar_violations(s1["params"], s2["params"], kgrid)
        if bad_k.size:
            gaps = svi_w(bad_k, s1["params"]) - svi_w(bad_k, s2["params"])
            alerts.append({"kind": "calendar", "T_short": t1, "T_long": t2,
                           "k_min": float(bad_k.min()), "k_max": float(bad_k.max()),
                           "severity": float(np.max(gaps))})
    return alerts
