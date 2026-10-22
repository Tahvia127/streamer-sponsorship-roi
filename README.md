# Are Streamer Sponsorships Worth It?

## Business question

For each dollar of marketing, how much engagement does a sponsored stream buy, and does it depend on streamer size? What share of a game's marketing budget should go to streamers, and to which tier?

## Recommendation

_TBD._ Expected shape: "Shift X% of the streamer budget from mega to mid-size streamers. A dollar on mid-size streams returns about Y extra concurrent players over 72 hours, versus Z for mega streams, which saturate quickly."

## Who cares

Marketing analytics and user acquisition teams at Microsoft Gaming, Electronic Arts, Riot Games, Epic Games, Take-Two, Twitch, and YouTube Gaming.

## Data

Publishers don't share campaign spend or install data, so this project builds its own panel from public APIs, collected hourly.

| Source | What it gives | Proxy for | Limits |
|--------|---------------|-----------|--------|
| Steam `GetNumberOfCurrentPlayers` | Live concurrent players per game | Engagement | Steam only (no console, no Epic) |
| Steam `appreviews` summary | Cumulative review count and score | Sales (reviews track purchases) | Lags purchases; review bombing |
| Twitch Get Streams | Live streams per game: viewers, streamer, title, tags | Exposure (viewer-hours) | Snapshot only; no history, so we collect it |
| `#ad` / `#sponsored` in titles and tags | Sponsored flag | Paid placements | Under-disclosure means false negatives |
| Google Trends (hand-downloaded CSVs) | Search interest per game | Control for underlying hype | Weekly, relative scale |

Tracked games are listed in [data/external/games.csv](data/external/games.csv) (20 Steam titles across shooters, survival, and RPGs). Edit it to add games; Twitch IDs are resolved at run time.

## Method

1. **Collection and data quality** ([notebook](notebooks/01_collection_and_data_quality.ipynb)). A GitHub Actions job snapshots Steam and Twitch every hour and commits CSVs. Build tidy game × hour and stream × hour panels.
2. **Event study** ([notebook](notebooks/02_event_study.ipynb)). Track player counts and new reviews before and after major or sponsored streams, with pre-trend checks and Google Trends controls (diff-diff).
3. **Streamer tiers** ([notebook](notebooks/03_streamer_tiers.ipynb)). Compare small, mid-size, and mega streamers on engagement lift per viewer-hour.
4. **Response curves and budget split** ([notebook](notebooks/04_response_curves_budget.ipynb)). Marketing mix model with carryover and diminishing returns (PyMC-Marketing), then an optimized budget split.

## The catch

Streamers pick games that are already trending, and publishers time sponsorships around launches and updates. A naive before/after comparison credits the stream for growth that was coming anyway. Pre-trend checks, not-yet-treated games as controls, and Google Trends interest as a covariate address this.

## Deliverable

A recommended budget split across streamer tiers, with the expected engagement lift per dollar.

## Repo layout

```
├── .github/workflows/collect.yml   hourly collector (git scraping)
├── data/
│   ├── collected/YYYY/MM/          steam_*.csv, twitch_*.csv (committed by the workflow)
│   ├── external/games.csv          tracked games
│   └── raw/google_trends/          hand-downloaded Trends CSVs (git-ignored)
├── notebooks/                      01–04, one per method step
├── src/streamads/collect.py        the collector
├── requirements-collector.txt      minimal deps for the workflow
└── requirements.txt                full analysis environment
```

## Start the collector

The APIs only return current snapshots, so every day the collector isn't running is history you can't get back. Start it first.

1. **Create a GitHub repo** (e.g. `streamer-sponsorships`) and push this folder as its root, so `.github/workflows/` sits at the top level:
   ```bash
   git init && git add . && git commit -m "Initial scaffold" && git branch -M main
   git remote add origin https://github.com/<you>/streamer-sponsorships.git && git push -u origin main
   ```
2. **Register a Twitch app** at [dev.twitch.tv/console](https://dev.twitch.tv/console) → Register Your Application. Any name, OAuth redirect `http://localhost`, category "Analytics Tool". Copy the Client ID and generate a Client Secret.
3. **Add repository secrets:** repo → Settings → Secrets and variables → Actions → New repository secret. Add `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`. Never put them in code.
4. **Enable Actions:** repo → Actions tab, enable workflows if prompted. Under Settings → Actions → General, confirm "Read and write permissions" is allowed (the workflow also requests `contents: write`).
5. **Trigger a first run:** Actions → "Collect Steam and Twitch snapshots" → Run workflow. Check that a `Snapshot ...` commit appears and the log has no "twitch game not found" warnings (fix names in `games.csv` if it does).

After that it runs hourly on its own. Notes:

- GitHub disables scheduled workflows after 60 days without repository activity. The workflow's own commits count as activity, so it keeps itself alive as long as it's committing data. If it ever stops, re-enable it from the Actions tab.
- Scheduled runs can be delayed or skipped when GitHub is busy. Notebook 01 checks for gaps.
- Without Twitch secrets the Steam half still runs and Twitch is skipped with a warning.

## How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One collector run locally (Twitch needs the two env vars)
PYTHONPATH=src python -m streamads.collect --sources steam twitch

# Pull the latest data the workflow committed, then open the notebooks
git pull && jupyter lab notebooks/
```

## Status

- [x] Collector (Steam verified live; Twitch untested until credentials are added)
- [x] Hourly GitHub Actions workflow
- [ ] Repo pushed and collector running
- [ ] Google Trends CSVs downloaded for tracked games
- [ ] Data quality checks (notebook 01)
- [ ] Event study (notebook 02)
- [ ] Streamer tier comparison (notebook 03)
- [ ] MMM and budget split (notebook 04)
- [ ] Recommendation written above

## Resources

- Twitch [Get Streams](https://dev.twitch.tv/docs/api/reference#get-streams) and [Get Games](https://dev.twitch.tv/docs/api/reference#get-games) reference
- Steam current players: `https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=APPID`
- Steam reviews summary: `https://store.steampowered.com/appreviews/APPID?json=1&filter=summary&num_per_page=0&language=all`
- Git scraping: [GitHub Actions data pipeline tutorial](https://actions-pipeline.readthedocs.io/), [scraping on a schedule](https://palewi.re/docs/go-big-with-github-actions/scraper.html), [Simon Willison's example workflow](https://github.com/simonw/scrape-open-data/blob/04426eeb3ae32581bebb6bbed944f35843d09cde/.github/workflows/scrape.yml)
- Event studies: [diff-diff docs](https://diff-diff.readthedocs.io/), [The Effect, ch. 18](https://theeffectbook.net/ch-DifferenceinDifference.html)
- MMM: [PyMC-Marketing end-to-end example](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html), [Meridian tutorial](https://www.youtube.com/watch?v=dWWaY4dZPYQ), [PyMC-Marketing vs Meridian](https://www.youtube.com/watch?v=xRddUq8A6ZM)
