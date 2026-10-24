"""Synthetic snapshots with the collector's exact schema, for rehearsing the analysis.

The real panel needs weeks of hourly data. This fakes it, with a known answer:
streams raise player counts through carryover and saturation, mid-size streams
are the most efficient per viewer-hour, and streamers pile onto games whose hype
is about to rise (the selection problem the event study has to survive).

    from streamads.synthetic import make_synthetic
    sim = make_synthetic()
    steam, twitch = sim["steam"], sim["twitch"]   # same columns as the collected CSVs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from streamads.panels import load_games

# The planted effect: lift (log points) = sum_tier beta * A / (A + k), where A is an
# hourly EWMA (decay EWMA_DECAY) of the tier's concurrent viewers in thousands.
TRUTH = {
    "ewma_decay": 0.97,
    "beta": {"small": 0.15, "mid": 0.30, "mega": 0.50},
    "k": {"small": 4.0, "mid": 6.0, "mega": 40.0},
}
POOL = {  # streamers per game, typical viewers (low, high), chance of going live per day
    "small": (150, (20, 900), 0.45),
    "mid": (20, (1_000, 9_000), 0.30),
    "mega": (3, (12_000, 60_000), 0.07),
}
SPONSORED_P = {"small": 0.005, "mid": 0.02, "mega": 0.05}
TOP_N = 300  # the collector keeps the top 300 streams per game


def make_synthetic(days: int = 63, n_games: int = 12, start: str = "2026-08-03", seed: int = 0) -> dict:
    """Return {"steam", "twitch", "trends", "truth"} DataFrames/dict."""
    rng = np.random.default_rng(seed)
    games = load_games().head(n_games).reset_index(drop=True)
    n_hours = days * 24
    hours = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    hod = np.arange(n_hours) % 24
    dow = hours.dayofweek.to_numpy()

    # Hype: a slow daily random walk per game, which drives players and trends.
    hype_daily = np.cumsum(rng.normal(0, 0.04, size=(n_games, days + 1)), axis=1)
    hype_hourly = np.repeat(hype_daily[:, :days], 24, axis=1)

    # Streaming sessions: streamers go live more often when tomorrow's hype is higher.
    sessions = []
    for g in range(n_games):
        dz = np.clip(hype_daily[g, 1:] - hype_daily[g, :-1], -0.1, 0.1)
        for tier, (n, (lo, hi), p) in POOL.items():
            base = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
            p_live = np.clip(p * (1 + 8 * dz), 0, 1)
            live = rng.random((n, days)) < p_live
            s_idx, d_idx = np.nonzero(live)
            k = len(s_idx)
            sessions.append(pd.DataFrame({
                "game": g, "tier": tier, "streamer": [f"{tier}_{g}_{i}" for i in s_idx],
                "base": base[s_idx],
                "start": d_idx * 24 + rng.integers(12, 24, size=k),
                "dur": rng.integers(2, 7, size=k),
                "sponsored": rng.random(k) < SPONSORED_P[tier],
            }))
    sess = pd.concat(sessions, ignore_index=True)
    sess["stream_id"] = np.arange(len(sess)) + 40_000_000_000
    rows = sess.loc[sess.index.repeat(sess["dur"])].copy()
    rows["h"] = rows["start"] + rows.groupby(level=0).cumcount()
    rows = rows[rows["h"] < n_hours].reset_index(drop=True)
    rows["viewer_count"] = np.maximum(1, (rows["base"] * rng.lognormal(0, 0.15, len(rows)))).round().astype(int)

    # Planted effect on players.
    lift = np.zeros((n_games, n_hours))
    marginal = {}
    a = TRUTH["ewma_decay"]
    for tier in POOL:
        conc = np.zeros((n_games, n_hours))
        sub = rows[rows["tier"] == tier]
        np.add.at(conc, (sub["game"].to_numpy(), sub["h"].to_numpy()), sub["viewer_count"].to_numpy() / 1_000)
        ewma = np.zeros_like(conc)
        for t in range(n_hours):
            ewma[:, t] = (1 - a) * conc[:, t] + (a * ewma[:, t - 1] if t else 0)
        beta, k = TRUTH["beta"][tier], TRUTH["k"][tier]
        lift += beta * ewma / (ewma + k)
        # Extra 1,000 viewer-hours -> cumulative log-point-hours = slope of the curve at A.
        marginal[tier] = float(np.mean(beta * k / (ewma + k) ** 2))

    base_log = np.log(rng.uniform(5_000, 400_000, size=n_games))[:, None]
    phase = rng.integers(0, 24, size=n_games)[:, None]
    cycle = 0.25 * np.cos(2 * np.pi * (hod[None, :] - phase) / 24) + 0.06 * (dow[None, :] >= 5)
    noise = np.zeros((n_games, n_hours))
    eps = rng.normal(0, 0.015, size=(n_games, n_hours))
    for t in range(1, n_hours):
        noise[:, t] = 0.7 * noise[:, t - 1] + eps[:, t]
    players = np.exp(base_log + cycle + hype_hourly + lift + noise).round().astype(int)

    new_reviews = rng.poisson(players * 2e-4)
    total_reviews = rng.integers(50_000, 2_000_000, size=n_games)[:, None] + np.cumsum(new_reviews, axis=1)

    # Snapshot timestamps drift past the scheduled minute; ~2% of runs are skipped.
    minute = rng.integers(17, 45, size=n_hours)
    stamps = (hours + pd.to_timedelta(minute, unit="m") + pd.to_timedelta(rng.integers(0, 60, n_hours), unit="s"))
    stamp_str = np.array([s.isoformat() for s in stamps])
    kept = rng.random(n_hours) > 0.02

    steam = pd.DataFrame({
        "snapshot_utc": np.tile(stamp_str, n_games),
        "steam_appid": np.repeat(games["steam_appid"].to_numpy(), n_hours),
        "steam_name": np.repeat(games["steam_name"].to_numpy(), n_hours),
        "current_players": players.ravel(),
        "total_reviews": total_reviews.ravel(),
    })
    steam["total_positive"] = (steam["total_reviews"] * 0.85).round().astype(int)
    steam["total_negative"] = steam["total_reviews"] - steam["total_positive"]
    steam["review_score"] = 8
    steam["review_score_desc"] = "Very Positive"
    steam["error"] = ""
    steam = steam[np.tile(kept, n_games)].reset_index(drop=True)

    rows = rows[kept[rows["h"].to_numpy()]]
    rows = rows.sort_values("viewer_count", ascending=False).groupby(["game", "h"]).head(TOP_N)
    g = games.loc[rows["game"].to_numpy()]
    twitch = pd.DataFrame({
        "snapshot_utc": stamp_str[rows["h"].to_numpy()],
        "steam_appid": g["steam_appid"].to_numpy(),
        "twitch_game_id": (rows["game"] + 500_000).to_numpy(),
        "twitch_game_name": g["twitch_game_name"].to_numpy(),
        "stream_id": rows["stream_id"].to_numpy(),
        "user_id": pd.factorize(rows["streamer"])[0] + 100_000,
        "user_login": rows["streamer"].to_numpy(),
        "viewer_count": rows["viewer_count"].to_numpy(),
        "language": "en",
        "started_at": [hours[s].isoformat().replace("+00:00", "Z") for s in rows["start"]],
        "is_mature": False,
        "title": np.where(rows["sponsored"], "#ad trying the new season with our partner", "chill ranked grind"),
        "tags": np.where(rows["sponsored"], "English|Sponsored", "English"),
        "sponsored_flag": rows["sponsored"].to_numpy(),
    }).sort_values("snapshot_utc").reset_index(drop=True)

    dates = pd.date_range(start, periods=days, freq="D", tz="UTC")
    interest = 50 * np.exp(hype_daily[:, :days] - hype_daily[:, :days].mean(axis=1, keepdims=True))
    trends = pd.DataFrame({
        "date": np.tile(dates, n_games),
        "steam_appid": np.repeat(games["steam_appid"].to_numpy(), days),
        "keyword": np.repeat(games["steam_name"].to_numpy(), days),
        "interest": (interest * rng.lognormal(0, 0.05, interest.shape)).ravel().round(1),
    })
    truth = {**TRUTH, "marginal_per_1k_vh": marginal}
    return {"steam": steam, "twitch": twitch, "trends": trends, "truth": truth}
