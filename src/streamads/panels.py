"""Load collected snapshots and build the hourly panels the analysis runs on.

Usage (from the project root, PYTHONPATH=src):
    from streamads import panels
    steam = panels.load_snapshots("steam")
    twitch = panels.load_snapshots("twitch")
    panel = panels.steam_panel(steam)
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTED = PROJECT_ROOT / "data" / "collected"
GAMES_CSV = PROJECT_ROOT / "data" / "external" / "games.csv"
TRENDS_KEYWORDS_CSV = PROJECT_ROOT / "data" / "external" / "trends_keywords.csv"
TRENDS_DIR = PROJECT_ROOT / "data" / "raw" / "google_trends"

TIER_LABELS = ["small", "mid", "mega"]


def load_games() -> pd.DataFrame:
    return pd.read_csv(GAMES_CSV)


def collected_files(source: str, root: Path = COLLECTED) -> list[Path]:
    """All <source>_YYYY-MM-DD.csv files, skipping the exFAT `._*` metadata files."""
    return sorted(p for p in Path(root).glob(f"**/{source}_*.csv") if not p.name.startswith("._"))


def load_snapshots(source: str, root: Path = COLLECTED) -> pd.DataFrame:
    """Concatenate every collected CSV for `source` ("steam" or "twitch")."""
    files = collected_files(source, root)
    if not files:
        raise FileNotFoundError(f"no {source}_*.csv under {root}; run `git pull` first")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    return clean_snapshots(df, source)


def clean_snapshots(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Parse timestamps, add an `hour` key, and drop duplicate rows within an hour.

    Scheduled runs drift and a manual run can land in the same hour, so the
    hour (not the exact timestamp) is the panel's time key.
    """
    df = df.copy()
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True, format="ISO8601")
    df["hour"] = df["snapshot_utc"].dt.floor("h")
    if source == "steam":
        df = df.sort_values("snapshot_utc").drop_duplicates(["steam_appid", "hour"], keep="last")
    else:
        # Pagination can return the same stream twice in one snapshot.
        df["sponsored_flag"] = df["sponsored_flag"].astype(str).str.lower().eq("true")
        df = df.sort_values("viewer_count").drop_duplicates(["stream_id", "hour"], keep="last")
    return df.sort_values(["hour"]).reset_index(drop=True)


def coverage(steam: pd.DataFrame) -> pd.DataFrame:
    """Per day: hours collected (of 24) and games with errors."""
    has_error = steam["error"].fillna("").ne("")
    return (
        steam.assign(date=steam["hour"].dt.date, has_error=has_error)
        .groupby("date")
        .agg(hours=("hour", "nunique"), error_rows=("has_error", "sum"))
    )


def steam_panel(steam: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Balanced game × hour panel with players, new reviews, and a deseasonalized outcome.

    - Missing hours are added (`missing=True`) and gaps up to `max_gap_hours`
      are interpolated; longer gaps stay NaN.
    - `log_players_ds` is log players minus that game's hour-of-week mean, so
      each game's own daily and weekly cycle doesn't show up as an "effect".
    """
    hours = pd.date_range(steam["hour"].min(), steam["hour"].max(), freq="h")
    games = steam[["steam_appid", "steam_name"]].drop_duplicates("steam_appid")
    grid = pd.MultiIndex.from_product([games["steam_appid"], hours], names=["steam_appid", "hour"])
    cols = ["current_players", "total_reviews", "review_score"]
    panel = steam.set_index(["steam_appid", "hour"])[cols].reindex(grid).reset_index()
    panel["missing"] = panel["current_players"].isna()
    panel = panel.merge(games, on="steam_appid", how="left")

    g = panel.groupby("steam_appid", group_keys=False)
    for col in ["current_players", "total_reviews"]:
        panel[col] = g[col].apply(lambda s: s.interpolate(limit=max_gap_hours, limit_area="inside"))
    # Reviews are cumulative; deletions can make the difference negative.
    panel["new_reviews"] = g["total_reviews"].diff().clip(lower=0)

    panel["log_players"] = np.log1p(panel["current_players"])
    panel["hour_of_week"] = panel["hour"].dt.dayofweek * 24 + panel["hour"].dt.hour
    seasonal = panel.groupby(["steam_appid", "hour_of_week"])["log_players"].transform("mean")
    panel["log_players_ds"] = panel["log_players"] - seasonal
    return panel


def assign_tiers(twitch: pd.DataFrame, cutoffs: tuple[int, int] = (1_000, 10_000)) -> pd.Series:
    """Tier per streamer from median concurrent viewers across all their snapshots."""
    typical = twitch.groupby("user_login")["viewer_count"].median()
    bins = [-np.inf, cutoffs[0], cutoffs[1], np.inf]
    return pd.cut(typical, bins, labels=TIER_LABELS, right=False).rename("tier")


def game_hour_exposure(twitch: pd.DataFrame, tiers: pd.Series) -> pd.DataFrame:
    """Viewer-hours per game-hour by tier (each hourly snapshot counts as one hour).

    Columns: vh_small, vh_mid, vh_mega, vh_total, vh_sponsored, n_streams.
    """
    t = twitch.assign(tier=twitch["user_login"].map(tiers))
    vh = t.pivot_table(index=["steam_appid", "hour"], columns="tier", values="viewer_count",
                       aggfunc="sum", fill_value=0, observed=False)
    vh = vh.reindex(columns=TIER_LABELS, fill_value=0).add_prefix("vh_")
    vh.columns.name = None
    vh["vh_total"] = vh.sum(axis=1)
    by_gh = t.assign(sponsored_vc=t["viewer_count"].where(t["sponsored_flag"], 0)).groupby(["steam_appid", "hour"])
    vh["vh_sponsored"] = by_gh["sponsored_vc"].sum()
    vh["n_streams"] = by_gh.size()
    return vh.reset_index()


def add_exposure(panel: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Join exposure onto the Steam panel; hours with no tracked streams get 0."""
    out = panel.merge(exposure, on=["steam_appid", "hour"], how="left")
    cols = [c for c in exposure.columns if c.startswith("vh_") or c == "n_streams"]
    out[cols] = out[cols].fillna(0)
    return out


def add_trends(panel: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    """Join daily Trends interest onto the hourly panel as `interest` and `log_interest`.

    Trends dates are local-day buckets; matching on the UTC date is close enough
    for a daily control.
    """
    t = trends.assign(date=pd.to_datetime(trends["date"], utc=True).dt.floor("D"))
    t = t.groupby(["steam_appid", "date"], as_index=False)["interest"].mean()
    out = panel.assign(date=panel["hour"].dt.floor("D")).merge(t, on=["steam_appid", "date"], how="left")
    out["interest"] = out.groupby("steam_appid")["interest"].transform(lambda s: s.ffill())
    out["log_interest"] = np.log(out["interest"].clip(lower=0.5))
    return out.drop(columns="date")


# --- Google Trends (hand-downloaded CSVs) ----------------------------------

_HEADER_FIRST = {"day", "week", "month", "time"}


def read_trends_csv(path: Path) -> pd.DataFrame:
    """One Trends "interest over time" export -> long (date, keyword, interest).

    Handles both export layouts (with or without the "Category:" preamble),
    strips the ": (Worldwide)" suffix from column names, and reads "<1" as 0.5.
    """
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.split(",")[0].strip().lower() in _HEADER_FIRST)
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    date_col = df.columns[0]
    df = df.rename(columns={c: re.sub(r":\s*\(.*\)$", "", c).strip() for c in df.columns[1:]})
    long = df.melt(id_vars=date_col, var_name="keyword", value_name="interest")
    long["interest"] = pd.to_numeric(long["interest"].astype(str).str.replace("<1", "0.5"), errors="coerce")
    long["date"] = pd.to_datetime(long.pop(date_col)).dt.tz_localize("UTC")
    long["batch"] = Path(path).stem
    return long


def load_trends(anchor: str, directory: Path = TRENDS_DIR) -> pd.DataFrame:
    """All Trends batches, rescaled onto one scale via the anchor keyword in every batch.

    Each batch is scaled so its anchor series matches the first batch's anchor.
    Returns (date, steam_appid, keyword, interest) using data/external/trends_keywords.csv.
    """
    paths = sorted(p for p in Path(directory).glob("*.csv") if not p.name.startswith("._"))
    if not paths:
        raise FileNotFoundError(f"no Trends CSVs in {directory}")
    batches = [read_trends_csv(p) for p in paths]
    ref = batches[0].query("keyword == @anchor").set_index("date")["interest"]
    scaled = []
    for b in batches:
        mine = b.query("keyword == @anchor").set_index("date")["interest"]
        overlap = ref.index.intersection(mine.index)
        factor = ref.loc[overlap].mean() / mine.loc[overlap].mean()
        scaled.append(b.assign(interest=b["interest"] * factor))
    trends = pd.concat(scaled).groupby(["date", "keyword"], as_index=False)["interest"].mean()
    with TRENDS_KEYWORDS_CSV.open(newline="", encoding="utf-8") as f:
        mapping = {row["trends_keyword"].lower(): int(row["steam_appid"]) for row in csv.DictReader(f)}
    trends["steam_appid"] = trends["keyword"].str.lower().map(mapping)
    return trends.dropna(subset=["steam_appid"]).astype({"steam_appid": int})
