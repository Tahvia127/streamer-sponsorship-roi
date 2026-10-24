"""Stream events and a stacked event study on the hourly Steam panel.

The design: for each clean event (a large or sponsored stream on game g at hour
t0), compare g's outcome to the average of games with no event in the window,
hour by hour, relative to a pre-event baseline. Averaging those gaps across
events gives the event-study curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REF_HOURS = (-6, -1)  # baseline window; effects are measured relative to it


def find_events(twitch: pd.DataFrame, q: float = 0.98, min_gap_hours: int = 72) -> pd.DataFrame:
    """Large and sponsored stream events, one row per stream.

    - large: the first hour a stream's viewers exceed the game's `q` quantile of
      "biggest single stream in the hour".
    - sponsored: the first hour a stream carries `sponsored_flag`.
    - clean: no other event on the same game in the previous `min_gap_hours`,
      so pre-periods aren't contaminated by an earlier event.
    """
    hourly_max = twitch.groupby(["steam_appid", "hour"])["viewer_count"].max()
    threshold = hourly_max.groupby("steam_appid").quantile(q).rename("threshold")
    t = twitch.join(threshold, on="steam_appid")
    t["is_large"] = t["viewer_count"] > t["threshold"]

    stream_vh = twitch.groupby("stream_id")["viewer_count"].sum().rename("stream_viewer_hours")
    stream_peak = twitch.groupby("stream_id")["viewer_count"].max().rename("peak_viewers")
    flagged = t[t["is_large"] | t["sponsored_flag"]].sort_values("hour")
    first = flagged.groupby("stream_id").first()
    any_large = t.groupby("stream_id")["is_large"].any()
    any_sponsored = t.groupby("stream_id")["sponsored_flag"].any()
    events = first[["steam_appid", "hour", "user_login"]].copy()
    events["large"] = any_large.reindex(events.index)
    events["sponsored"] = any_sponsored.reindex(events.index)
    events = events.join(stream_vh).join(stream_peak).reset_index()

    events = events.sort_values(["steam_appid", "hour"]).reset_index(drop=True)
    prev = events.groupby("steam_appid")["hour"].shift()
    gap = (events["hour"] - prev).dt.total_seconds() / 3600
    events["clean"] = gap.isna() | (gap > min_gap_hours)
    events["event_id"] = np.arange(len(events))
    return events


def placebo_events(panel: pd.DataFrame, events: pd.DataFrame, n: int, before_hours: int = 72,
                   after_hours: int = 48, seed: int = 0) -> pd.DataFrame:
    """Random game-hours with no real event on that game in [t - before_hours, t + after_hours].

    The defaults mirror the real design: the same 72 h clean rule before, and no
    event inside the 48 h post window.
    """
    rng = np.random.default_rng(seed)
    cand = panel.loc[~panel["missing"], ["steam_appid", "hour"]]
    hours = cand["hour"].unique()
    lo, hi = hours.min() + pd.Timedelta(hours=before_hours), hours.max() - pd.Timedelta(hours=after_hours)
    cand = cand[(cand["hour"] > lo) & (cand["hour"] < hi)]
    near = cand.merge(events[["steam_appid", "hour"]], on="steam_appid", suffixes=("", "_ev"))
    lag = near["hour"] - near["hour_ev"]
    near = near[(lag >= -pd.Timedelta(hours=after_hours)) & (lag <= pd.Timedelta(hours=before_hours))]
    cand = cand.merge(near[["steam_appid", "hour"]].drop_duplicates(), how="left", indicator=True)
    cand = cand[cand["_merge"] == "left_only"].drop(columns="_merge")
    pick = cand.sample(n=min(n, len(cand)), random_state=rng.integers(1 << 31))
    return pick.assign(event_id=np.arange(len(pick)), clean=True, large=False, sponsored=False)


def event_stack(panel: pd.DataFrame, events: pd.DataFrame, outcome: str = "log_players_ds",
                pre: int = 24, post: int = 48, exclude: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table: one row per (event, game, relative hour) for the treated game and its controls.

    `events` are the events to study; `exclude` is every event that should
    disqualify a control (pass the full find_events table, so a game with any
    large or sponsored stream in the window can't serve as a control).
    Defaults to `events`. Controls are games with no excluded event within
    [t0 - pre, t0 + post].
    """
    y = panel.set_index(["steam_appid", "hour"])[outcome]
    games = panel["steam_appid"].unique()
    rel = np.arange(-pre, post + 1)
    exclude = events if exclude is None else exclude
    rows = []
    for ev in events[events["clean"]].itertuples():
        t0 = ev.hour
        win_lo, win_hi = t0 - pd.Timedelta(hours=pre), t0 + pd.Timedelta(hours=post)
        busy = set(exclude.loc[(exclude["hour"] >= win_lo) & (exclude["hour"] <= win_hi), "steam_appid"])
        controls = [g for g in games if g not in busy]
        if not controls:
            continue
        hours = t0 + pd.to_timedelta(rel, unit="h")
        for g in [ev.steam_appid, *controls]:
            vals = y.reindex(pd.MultiIndex.from_arrays([np.full(len(hours), g), hours])).to_numpy()
            rows.append(pd.DataFrame({"event_id": ev.event_id, "steam_appid": g, "rel_hour": rel,
                                      "treated": g == ev.steam_appid, "y": vals}))
    if not rows:
        raise ValueError("no clean events with controls; collect more data or relax min_gap_hours")
    return pd.concat(rows, ignore_index=True)


def event_gaps(stack: pd.DataFrame) -> pd.DataFrame:
    """Treated minus control mean per event and hour, re-centred on the baseline window."""
    wide = stack.pivot_table(index=["event_id", "rel_hour"], columns="treated", values="y")
    gaps = (wide[True] - wide[False]).rename("gap").reset_index()
    base = gaps[gaps["rel_hour"].between(*REF_HOURS)].groupby("event_id")["gap"].mean()
    gaps["gap"] = gaps["gap"] - gaps["event_id"].map(base)
    game = stack.loc[stack["treated"]].drop_duplicates("event_id").set_index("event_id")["steam_appid"]
    gaps["steam_appid"] = gaps["event_id"].map(game)
    return gaps.dropna(subset=["gap"])


def event_study(gaps: pd.DataFrame, bin_hours: int = 6):
    """Binned event-study coefficients with SEs clustered by game, plus a pre-trend test.

    Returns (coefs, pretrend) where coefs has bin, coef, ci_low, ci_high, and
    pretrend is the joint Wald test that every pre-event bin equals zero.
    """
    d = gaps.assign(bin=(gaps["rel_hour"] // bin_hours) * bin_hours)
    ref_bin = (REF_HOURS[0] // bin_hours) * bin_hours
    d = d[d["bin"] != ref_bin]
    if d["steam_appid"].nunique() < 2:
        raise ValueError("events cover fewer than 2 games, so game-clustered SEs can't be computed; "
                         "collect more data or relax the event filters")
    fit = smf.ols("gap ~ 0 + C(bin)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["steam_appid"].astype("category").cat.codes})
    ci = fit.conf_int()
    bins = np.array([int(name.split("[")[1].rstrip("]")) for name in fit.params.index])
    coefs = pd.DataFrame({"bin": bins, "coef": fit.params.to_numpy(),
                          "ci_low": ci[0].to_numpy(), "ci_high": ci[1].to_numpy()})
    coefs = pd.concat([coefs, pd.DataFrame({"bin": [ref_bin], "coef": [0.0], "ci_low": [0.0], "ci_high": [0.0]})])
    coefs = coefs.sort_values("bin").reset_index(drop=True)

    pre = np.flatnonzero(bins < ref_bin)
    pretrend = fit.wald_test(np.eye(len(bins))[pre], scalar=True) if len(pre) else None
    return coefs, pretrend


def event_effects(gaps: pd.DataFrame, events: pd.DataFrame, panel: pd.DataFrame,
                  post: tuple[int, int] = (0, 48)) -> pd.DataFrame:
    """Per event: average post-event lift (log points) and extra player-hours.

    extra_player_hours sums baseline_players * (exp(gap) - 1) over the post
    window, where baseline_players is the treated game's mean in the baseline window.
    """
    post_gaps = gaps[gaps["rel_hour"].between(*post)]
    lift = post_gaps.groupby("event_id")["gap"].mean().rename("lift_log")

    players = panel.set_index(["steam_appid", "hour"])["current_players"]
    ev = events.set_index("event_id").loc[lift.index]
    base = []
    for eid, row in ev.iterrows():
        hrs = row["hour"] + pd.to_timedelta(np.arange(REF_HOURS[0], REF_HOURS[1] + 1), unit="h")
        base.append(players.reindex(pd.MultiIndex.from_arrays([np.full(len(hrs), row["steam_appid"]), hrs])).mean())
    ev = ev.assign(baseline_players=base).join(lift)
    extra = post_gaps.assign(b=post_gaps["event_id"].map(ev["baseline_players"]))
    extra = (extra["b"] * np.expm1(extra["gap"])).groupby(extra["event_id"]).sum()
    ev["extra_player_hours"] = extra
    ev["extra_per_1k_vh"] = ev["extra_player_hours"] / (ev["stream_viewer_hours"] / 1_000)
    return ev.reset_index()
