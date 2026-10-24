"""Engagement lift per viewer-hour, by streamer tier (a distributed-lag panel regression).

Regress the deseasonalized outcome on recent viewer-hours from each tier,
summed over lag windows. Removing each hour's cross-game mean soaks up
anything that moves every game at once (a Steam sale, a weekend evening).
Summing coefficient × window width gives the cumulative lift from 1,000 extra
viewer-hours, in log-point-hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from streamads.panels import TIER_LABELS

LAG_BINS = [(0, 5), (6, 23), (24, 71)]


def lagged_exposure(panel: pd.DataFrame, tiers: list[str] = TIER_LABELS,
                    bins: list[tuple[int, int]] = LAG_BINS) -> pd.DataFrame:
    """Add vh_<tier>_l<a>_<b>: viewer-hours (thousands) from hours t-a..t-b."""
    out = panel.sort_values(["steam_appid", "hour"]).copy()
    g = out.groupby("steam_appid")
    for tier in tiers:
        x = out[f"vh_{tier}"] / 1_000
        csum = x.groupby(out["steam_appid"]).cumsum()
        for a, b in bins:
            hi = csum.groupby(out["steam_appid"]).shift(a).fillna(0)
            lo = csum.groupby(out["steam_appid"]).shift(b + 1).fillna(0)
            out[f"vh_{tier}_l{a}_{b}"] = hi - lo
        # The first max-lag hours of each game don't have a full history.
    max_lag = max(b for _, b in bins)
    out["full_history"] = g.cumcount() > max_lag
    return out


def dose_response(panel: pd.DataFrame, outcome: str = "log_players_ds", tiers: list[str] = TIER_LABELS,
                  bins: list[tuple[int, int]] = LAG_BINS, controls: list[str] | None = None):
    """Fit the regression; return (lag_coefs, cumulative_by_tier, fit).

    `controls` (e.g. ["log_interest"]) enter unlagged; use them to absorb hype
    that streamers chase, which otherwise inflates the exposure coefficients.
    """
    controls = controls or []
    d = lagged_exposure(panel, tiers, bins)
    d = d[d["full_history"] & d[outcome].notna()].dropna(subset=controls)
    xcols = [f"vh_{t}_l{a}_{b}" for t in tiers for a, b in bins]
    cols = [outcome, *xcols, *controls]
    demeaned = d[cols] - d.groupby("hour")[cols].transform("mean")
    fit = sm.OLS(demeaned[outcome], demeaned[[*xcols, *controls]]).fit(
        cov_type="cluster", cov_kwds={"groups": d["steam_appid"].astype("category").cat.codes})

    lag = pd.DataFrame({"term": xcols, "coef": fit.params[xcols].to_numpy(), "se": fit.bse[xcols].to_numpy()})
    lag["tier"] = [c.split("_")[1] for c in xcols]
    lag["lag_hours"] = [c.split("_l")[1].replace("_", "-") for c in xcols]

    widths = np.array([b - a + 1 for a, b in bins], dtype=float)
    rows = []
    for t in tiers:
        r = np.zeros(len(xcols) + len(controls))
        r[[xcols.index(f"vh_{t}_l{a}_{b}") for a, b in bins]] = widths
        test = fit.t_test(r)
        ci = test.conf_int()[0]
        rows.append({"tier": t, "cum_lift": float(np.ravel(test.effect)[0]), "se": float(np.ravel(test.sd)[0]),
                     "ci_low": ci[0], "ci_high": ci[1]})
    return lag, pd.DataFrame(rows), fit


def to_player_hours(cumulative: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Translate log-point-hours into extra player-hours per 1,000 viewer-hours for a typical game."""
    typical = panel.groupby("steam_appid")["current_players"].mean().median()
    out = cumulative.copy()
    for c in ["cum_lift", "ci_low", "ci_high"]:
        out[c.replace("cum_lift", "player_hours").replace("ci_", "player_hours_")] = out[c] * typical
    out.attrs["typical_players"] = typical
    return out
