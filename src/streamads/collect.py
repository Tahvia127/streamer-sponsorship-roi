"""Collect hourly Steam and Twitch snapshots and append them to dated CSVs.

Usage (from the project root):
    PYTHONPATH=src python -m streamads.collect --sources steam twitch

Output: data/collected/YYYY/MM/<source>_YYYY-MM-DD.csv, one row per game (Steam)
or per live stream (Twitch), each stamped with snapshot_utc.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_CSV = PROJECT_ROOT / "data" / "external" / "games.csv"
OUT_DIR = PROJECT_ROOT / "data" / "collected"

STEAM_PLAYERS_URL = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
STEAM_REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_API = "https://api.twitch.tv/helix"

# Disclosure terms in titles and tags. "ad" only counts as a hashtag or a bracketed
# tag so words like "ad-free" or "road" don't match.
SPONSOR_RE = re.compile(
    r"(#ad\b|#sponsored\b|#sponsor\b|#partner\b|[\[(]ad[\])]|\bsponsored\b|\bpaid partnership\b|\bpartnered with\b)",
    re.IGNORECASE,
)
SPONSOR_TAGS = {"ad", "sponsored", "sponsor", "sponsorship", "paidpromotion"}

log = logging.getLogger("streamads")


def request_with_retry(method: str, url: str, retries: int = 4, backoff: float = 2.0, **kwargs) -> requests.Response:
    """Retry on network errors, 429s, and 5xx with exponential backoff."""
    kwargs.setdefault("timeout", 20)
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} from {url}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(exc.response, "status_code", None) if isinstance(exc, requests.HTTPError) else None
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            log.warning("retry %d/%d in %.0fs: %s", attempt + 1, retries - 1, wait, exc)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def load_games() -> list[dict]:
    with GAMES_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(source: str, rows: list[dict], snapshot: datetime) -> Path | None:
    if not rows:
        return None
    path = OUT_DIR / f"{snapshot:%Y}" / f"{snapshot:%m}" / f"{source}_{snapshot:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return path


# --- Steam -----------------------------------------------------------------

def collect_steam(games: list[dict], snapshot_utc: str) -> list[dict]:
    rows = []
    for g in games:
        appid = g["steam_appid"]
        row = {
            "snapshot_utc": snapshot_utc,
            "steam_appid": appid,
            "steam_name": g["steam_name"],
            "current_players": None,
            "total_reviews": None,
            "total_positive": None,
            "total_negative": None,
            "review_score": None,
            "review_score_desc": None,
            "error": "",
        }
        try:
            resp = request_with_retry("GET", STEAM_PLAYERS_URL, params={"appid": appid})
            row["current_players"] = resp.json()["response"].get("player_count")
        except Exception as exc:  # one game failing shouldn't drop the rest
            row["error"] += f"players: {exc}; "
            log.warning("steam players failed for %s: %s", appid, exc)
        try:
            resp = request_with_retry(
                "GET",
                STEAM_REVIEWS_URL.format(appid=appid),
                params={"json": 1, "filter": "summary", "num_per_page": 0, "language": "all", "purchase_type": "all"},
            )
            summary = resp.json().get("query_summary", {})
            row.update(
                total_reviews=summary.get("total_reviews"),
                total_positive=summary.get("total_positive"),
                total_negative=summary.get("total_negative"),
                review_score=summary.get("review_score"),
                review_score_desc=summary.get("review_score_desc"),
            )
        except Exception as exc:
            row["error"] += f"reviews: {exc}; "
            log.warning("steam reviews failed for %s: %s", appid, exc)
        rows.append(row)
        time.sleep(0.5)  # be polite to the store endpoint
    return rows


# --- Twitch ----------------------------------------------------------------

def twitch_headers() -> dict | None:
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    resp = request_with_retry(
        "POST",
        TWITCH_TOKEN_URL,
        params={"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"},
    )
    return {"Client-Id": client_id, "Authorization": f"Bearer {resp.json()['access_token']}"}


def resolve_twitch_ids(games: list[dict], headers: dict) -> dict[str, str]:
    """Map twitch_game_name -> id, using games.csv ids where filled in."""
    ids = {g["twitch_game_name"]: g["twitch_game_id"] for g in games if g.get("twitch_game_id")}
    missing = [g["twitch_game_name"] for g in games if not g.get("twitch_game_id")]
    for i in range(0, len(missing), 100):
        batch = missing[i : i + 100]
        resp = request_with_retry("GET", f"{TWITCH_API}/games", headers=headers, params=[("name", n) for n in batch])
        for item in resp.json().get("data", []):
            ids[item["name"]] = item["id"]
    # Twitch matches names case-insensitively but returns its own casing.
    lower = {k.lower(): v for k, v in ids.items()}
    resolved = {}
    for g in games:
        gid = lower.get(g["twitch_game_name"].lower())
        if gid:
            resolved[g["twitch_game_name"]] = gid
        else:
            log.warning("twitch game not found, fix twitch_game_name in games.csv: %r", g["twitch_game_name"])
    return resolved


def is_sponsored(title: str, tags: list[str]) -> bool:
    if SPONSOR_RE.search(title or ""):
        return True
    return any(t.lower().replace(" ", "") in SPONSOR_TAGS for t in tags or [])


def collect_twitch(games: list[dict], snapshot_utc: str, max_streams: int = 300) -> list[dict]:
    headers = twitch_headers()
    if headers is None:
        log.warning("TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET not set; skipping Twitch")
        return []
    name_to_id = resolve_twitch_ids(games, headers)
    steam_by_name = {g["twitch_game_name"]: g["steam_appid"] for g in games}
    rows = []
    for name, game_id in name_to_id.items():
        cursor, fetched = None, 0
        try:
            while fetched < max_streams:
                params = {"game_id": game_id, "first": 100, "type": "live"}
                if cursor:
                    params["after"] = cursor
                payload = request_with_retry("GET", f"{TWITCH_API}/streams", headers=headers, params=params).json()
                streams = payload.get("data", [])
                for s in streams:
                    tags = s.get("tags") or []
                    rows.append(
                        {
                            "snapshot_utc": snapshot_utc,
                            "steam_appid": steam_by_name[name],
                            "twitch_game_id": game_id,
                            "twitch_game_name": s.get("game_name") or name,
                            "stream_id": s.get("id"),
                            "user_id": s.get("user_id"),
                            "user_login": s.get("user_login"),
                            "viewer_count": s.get("viewer_count"),
                            "language": s.get("language"),
                            "started_at": s.get("started_at"),
                            "is_mature": s.get("is_mature"),
                            "title": s.get("title"),
                            "tags": "|".join(tags),
                            "sponsored_flag": is_sponsored(s.get("title", ""), tags),
                        }
                    )
                fetched += len(streams)
                cursor = payload.get("pagination", {}).get("cursor")
                if not streams or not cursor:
                    break
        except Exception as exc:
            log.warning("twitch streams failed for %s: %s", name, exc)
    return rows


COLLECTORS = {"steam": collect_steam, "twitch": collect_twitch}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", nargs="+", choices=sorted(COLLECTORS), default=["steam", "twitch"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N games (for testing)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    snapshot = datetime.now(timezone.utc).replace(microsecond=0)
    snapshot_utc = snapshot.isoformat()
    games = load_games()[: args.limit]

    wrote_any = False
    for source in args.sources:
        try:
            rows = COLLECTORS[source](games, snapshot_utc)
        except Exception as exc:  # keep going so one source can't block the other
            log.error("%s collector failed: %s", source, exc)
            continue
        path = append_rows(source, rows, snapshot)
        if path:
            wrote_any = True
            log.info("%s: wrote %d rows to %s", source, len(rows), path.relative_to(PROJECT_ROOT))
    return 0 if wrote_any else 1


if __name__ == "__main__":
    sys.exit(main())
