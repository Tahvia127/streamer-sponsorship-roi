# Step-by-step guide: Are streamer sponsorships worth it?

This guide walks through the whole project, from setup to the final recommendation. Each method step matches a notebook in `notebooks/` and uses the helpers in `src/streamads/`.

**Test status labels**
- **Tested:** the code ran on the real collected data.
- **Tested on substitute:** the code ran end to end on synthetic data with the collector's exact columns (see the [appendix](#appendix-the-synthetic-rehearsal-data)), because the real panel needs weeks of hourly snapshots first.
- **Untested:** written against the library's current documentation, not run here.

**How the pieces fit**

```
hourly snapshots ─► Step 1: game × hour panel + viewer-hours by tier
                     ├─► Step 2: event study (does a big stream move players?)
                     ├─► Step 3: lift per 1,000 viewer-hours, by tier
                     └─► Step 4: response curves ─► $ budget split
```

You can do every step today on synthetic data, then rerun the same code on real data once the collector has about 8 weeks of history. The only switch is `USE_SYNTHETIC`.

---

## Step 0: Setup

*Status: the environment commands are Untested here; the setup cell itself is Tested (the guide's Python ran in a shared environment with the same packages, on pandas 2.3).*

Create the environment on your Mac's internal disk, not on the YOTUO drive. The drive is formatted exFAT, and macOS writes `._*` files there that break pip.

```bash
python3 -m venv ~/.venvs/streamer-sponsorship-roi
source ~/.venvs/streamer-sponsorship-roi/bin/activate
cd "/Volumes/YOTUO/Portfolio Project/07-gaming-streamer-sponsorships"
pip install -r requirements.txt
```

`pymc-marketing` pulls in PyMC and is the largest install (about 1 GB). If disk space is tight, install it last, only when you reach Step 4b.

Start Jupyter with `src/` on the import path so `from streamads import ...` works:

```bash
PYTHONPATH=src jupyter lab notebooks/
```

Every notebook starts with the same cell. Flip `USE_SYNTHETIC` to `False` when you have real data.

```python
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from streamads import panels

USE_SYNTHETIC = True  # False once the collector has ~8 weeks of data
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TAG = "synthetic" if USE_SYNTHETIC else "real"
```

---

## Step 0.5: Get the data

*Status: Steam loader Tested on the real collected snapshot (20 games, 1 hour so far). Twitch and Trends loaders Tested on substitute: until those files exist they stop with a clear "no twitch_*.csv" / "no Trends CSVs" error, which is expected.*

**1. The collector.** The GitHub Actions workflow in [Tahvia127/streamer-sponsorship-roi](https://github.com/Tahvia127/streamer-sponsorship-roi) runs every hour at about minute 17 and commits CSVs to `data/collected/YYYY/MM/`. Check that it's running:

```bash
gh run list --workflow collect.yml --limit 5
```

**2. Twitch secrets.** Until you add them, only Steam is collected. Register an app at [dev.twitch.tv/console](https://dev.twitch.tv/console), then paste each value when prompted:

```bash
gh secret set TWITCH_CLIENT_ID
```

```bash
gh secret set TWITCH_CLIENT_SECRET
```

Then trigger a run and look for `twitch: wrote N rows` and any `twitch game not found` warnings in the log. Fix unmatched names in `data/external/games.csv` so they match Twitch's category name exactly.

```bash
gh workflow run collect.yml && sleep 10 && gh run watch
```

**3. Pull before every analysis session.** The workflow commits every hour, so your local copy goes stale fast:

```bash
git pull
```

**4. Google Trends (manual).** Trends is the control for underlying hype (see "The catch" in the README). There's no reliable API, so download CSVs by hand at [trends.google.com](https://trends.google.com/trends/explore):

- **Time range:** custom, from the day the collector started (2026-10-22) to today. Keep it under 270 days so Trends returns *daily* values.
- **Region:** Worldwide (Steam players are global).
- **Batches:** Trends compares at most 5 terms and rescales each download to 0–100. Put the anchor term **Rust** in every batch so the batches can be stitched onto one scale. Pick the "video game" topic when Trends offers it.

| File to save | Terms |
|---|---|
| `data/raw/google_trends/batch1.csv` | Rust, Counter-Strike 2, Dota 2, PUBG, Apex Legends |
| `data/raw/google_trends/batch2.csv` | Rust, Marvel Rivals, Overwatch 2, Call of Duty, Rainbow Six Siege |
| `data/raw/google_trends/batch3.csv` | Rust, Naraka Bladepoint, Destiny 2, Warframe, Helldivers 2 |
| `data/raw/google_trends/batch4.csv` | Rust, Palworld, GTA V, Baldur's Gate 3, Elden Ring |
| `data/raw/google_trends/batch5.csv` | Rust, Monster Hunter Wilds, Path of Exile 2, Starfield |

The term-to-game mapping is in `data/external/trends_keywords.csv`. If Trends labels a column differently (for example "Grand Theft Auto V"), edit `trends_keyword` there to match. Re-download all five batches each time you refresh: Trends rescales the whole range on every download.

**5. How long to wait.** The event study needs clean events (no other big stream on that game for 3 days before) and enough of them across games. Plan on **at least 8 weeks** of hourly data before trusting Steps 2–4, and 12+ weeks for Step 4. Rehearse on synthetic data now.

**6. Load the data.** This cell is the same in every notebook:

```python
if USE_SYNTHETIC:
    from streamads.synthetic import make_synthetic
    sim = make_synthetic(seed=0)
    steam = panels.clean_snapshots(sim["steam"], "steam")
    twitch = panels.clean_snapshots(sim["twitch"], "twitch")
    trends = sim["trends"]
else:
    steam = panels.load_snapshots("steam")
    twitch = panels.load_snapshots("twitch")
    trends = panels.load_trends(anchor="Rust")

print(f"{steam['hour'].nunique()} Steam hours, {twitch['hour'].nunique()} Twitch hours, "
      f"{twitch['stream_id'].nunique()} streams")
```

`load_snapshots` reads every `data/collected/**/<source>_*.csv`, skips the drive's `._*` files, adds an `hour` key, and drops duplicate rows within an hour (a manual run can land in the same hour as a scheduled one).

---

## Step 1: Collection and data quality

*Notebook: `01_collection_and_data_quality.ipynb` · Status: Tested on substitute.*

**Goal:** turn raw snapshots into two clean tables: a balanced game × hour panel with player counts, and viewer-hours per game-hour split by streamer tier.

**Concept.** Each hourly snapshot is a photo, not a log. Twitch tells you who is live *right now* and how many people are watching, so one snapshot of a stream with 5,000 viewers is about 5,000 viewer-hours. Summing over streams gives exposure. Steam gives concurrent players (engagement) and the cumulative review count (its first difference is a rough sales proxy). Missing hours are normal: GitHub sometimes delays or skips scheduled runs.

**Code.**

```python
# Coverage: hours collected per day (should be 22–24) and rows with errors
cov = panels.coverage(steam)
print(cov.describe())
print("days with < 20 hours:", (cov["hours"] < 20).sum())
```

```python
# Balanced game × hour panel. Short gaps (≤3 h) are interpolated; longer ones stay missing.
panel = panels.steam_panel(steam)
print(panel["missing"].mean().round(3), "share of game-hours missing")

# Streamer tiers from each streamer's median concurrent viewers
tiers = panels.assign_tiers(twitch, cutoffs=(1_000, 10_000))
print(tiers.value_counts())

# Viewer-hours per game-hour, by tier, joined onto the panel
exposure = panels.game_hour_exposure(twitch, tiers)
panel = panels.add_exposure(panel, exposure)
panel = panels.add_trends(panel, trends)
panel[["steam_name", "hour", "current_players", "new_reviews", "vh_small", "vh_mid", "vh_mega", "log_interest"]].head()
```

```python
# One game's week: players vs. viewer-hours
g = panel["steam_appid"].iloc[0]
week = panel[(panel["steam_appid"] == g) & (panel["hour"] < panel["hour"].min() + pd.Timedelta(days=7))]
fig = go.Figure()
fig.add_scatter(x=week["hour"], y=week["current_players"], name="players")
fig.add_scatter(x=week["hour"], y=week["vh_total"], name="viewer-hours", yaxis="y2")
fig.update_layout(title=f"{week['steam_name'].iloc[0]}: players and stream exposure",
                  yaxis2=dict(overlaying="y", side="right"))
fig.write_html(FIGURES / f"01_one_game_week_{TAG}.html")
```

```python
PROCESSED.mkdir(parents=True, exist_ok=True)
panel.to_parquet(PROCESSED / f"panel_{TAG}.parquet")
tiers.to_frame().to_parquet(PROCESSED / f"tiers_{TAG}.parquet")
twitch.to_parquet(PROCESSED / f"twitch_{TAG}.parquet")
```

**Checkpoint.** On the synthetic data: 12 games × 1,512 hours, about 2% of hours missing (the simulator skips runs at random), and roughly 1,800 small, 240 mid, and 36 mega streamers. On real data, expect a much longer small-streamer tail.

**Pitfalls.**
- **The collector keeps only the top 300 streams per game.** For big games (Counter-Strike, Dota) that cuts off the small-streamer tail, so small-tier viewer-hours are undercounted there. Say so in the README; it biases the small tier's exposure down and its per-viewer-hour effect up.
- **Tier cutoffs are a choice.** Rerun Step 3 with `(500, 5_000)` and `(2_000, 20_000)` and report whether the ranking holds.
- **Steam's Call of Duty app (1938090) is the launcher for every CoD title,** not just Warzone. Treat its results with care.
- **`new_reviews` is noisy hour to hour.** Use it at the daily level (Step 4), not hourly.

**What to write in the README.** One paragraph on the panel: date range, games, share of hours missing, number of streamers per tier, and the top-300 limit.

---

## Step 2: Event study

*Notebook: `02_event_study.ipynb` · Status: Tested on substitute.*

**Goal:** measure what happens to a game's player count in the hours after a big or sponsored stream, compared with what was already happening.

**Concept.** An *event* is the first hour a stream gets unusually large for its game (above the game's 98th percentile of "biggest stream in the hour") or carries a sponsorship disclosure (`#ad`, `#sponsored`, a Sponsored tag). For each event, compare the treated game with games that had no event in the same window, hour by hour, relative to a baseline 1–6 hours before the event. Averaging those gaps across events gives the event-study curve. If the curve is flat *before* the event (no pre-trend), the jump *after* is credible; if it's already rising beforehand, streamers were chasing hype.

**Code.**

```python
from streamads import events as ev

panel = pd.read_parquet(PROCESSED / f"panel_{TAG}.parquet")
twitch = pd.read_parquet(PROCESSED / f"twitch_{TAG}.parquet")

events = ev.find_events(twitch, q=0.98, min_gap_hours=72)
print(events[["large", "sponsored", "clean"]].sum())
large = events[events["large"]]
```

`clean` means no other event on that game in the previous 72 hours, so the pre-period isn't polluted by an earlier stream. Only clean events enter the study.

```python
# The naive version, for contrast: treated game's change from 24 h before to 24 h after, no controls
y = panel.set_index(["steam_appid", "hour"])["log_players_ds"]
naive = []
for e in large[large["clean"]].itertuples():
    before = y.get((e.steam_appid, e.hour - pd.Timedelta(hours=24)), np.nan)
    after = y.get((e.steam_appid, e.hour + pd.Timedelta(hours=24)), np.nan)
    naive.append(after - before)
print("naive 24h change:", np.nanmean(naive).round(3))
```

```python
# exclude=events: a game with ANY event (large or sponsored) in the window can't be a control
stack = ev.event_stack(panel, large, outcome="log_players_ds", pre=24, post=48, exclude=events)
gaps = ev.event_gaps(stack)
coefs, pretrend = ev.event_study(gaps, bin_hours=6)
print(coefs.round(3))
print("pre-trend test p =", round(float(pretrend.pvalue), 3), "| events:", gaps["event_id"].nunique())
```

```python
fig = go.Figure()
fig.add_scatter(x=coefs["bin"], y=coefs["coef"], mode="lines+markers", name="effect",
                error_y=dict(type="data", symmetric=False,
                             array=coefs["ci_high"] - coefs["coef"], arrayminus=coefs["coef"] - coefs["ci_low"]))
fig.add_hline(y=0, line_dash="dot")
fig.add_vline(x=0, line_dash="dash")
fig.update_layout(title="Player count around large streams (log points vs. 1–6 h before)",
                  xaxis_title="hours relative to event (6-hour bins)", yaxis_title="log points")
fig.write_html(FIGURES / f"02_event_study_{TAG}.html")
```

```python
# Placebo: the same pipeline on random game-hours with no real event in the 72 h before or
# 48 h after. Its post-event estimate should be about zero; if it isn't, the design is picking
# up something else. Use many placebo draws: a sample as small as the real one is noisy.
placebo = ev.placebo_events(panel, events, n=300, seed=1)
p_coefs, _ = ev.event_study(ev.event_gaps(ev.event_stack(panel, placebo, pre=24, post=48, exclude=events)))
post_bins = p_coefs["bin"] >= 0
print("placebo post-event mean:", p_coefs.loc[post_bins, "coef"].mean().round(3),
      "| bins with CI above 0:", int((p_coefs.loc[post_bins, "ci_low"] > 0).sum()), "of", int(post_bins.sum()))
print("real post-event mean:   ", coefs.loc[coefs["bin"] >= 0, "coef"].mean().round(3))
```

```python
# Sponsored streams separately: usually fewer, so expect wide intervals
sponsored = events[events["sponsored"]]
if sponsored["clean"].sum() >= 10:
    s_coefs, s_pre = ev.event_study(ev.event_gaps(ev.event_stack(panel, sponsored, pre=24, post=48, exclude=events)))
    print(s_coefs.round(3))
else:
    print("too few clean sponsored events:", sponsored["clean"].sum())
```

```python
# Per-event effects, saved for Step 3
effects = ev.event_effects(gaps, large, panel)
effects.to_parquet(PROCESSED / f"event_effects_{TAG}.parquet")
effects[["steam_appid", "user_login", "peak_viewers", "lift_log", "extra_player_hours"]].describe()
```

**Checkpoint (synthetic, seed 0).** `find_events` returns 101 large and 366 sponsored streams, 86 of them clean. 18 clean large events have at least one control game with no event of any kind in the window, and those are what the study uses. Pre-event bins sit within about ±0.006 of zero (pre-trend p ≈ 0.62). The effect is +0.046 log points in the first 6 hours, rises to about +0.07 (≈7% more players), and holds for two days (post-event mean +0.065). The naive 24-hour change is +0.051. The placebo (300 draws) averages +0.006, with no post-event bin's interval above zero. Sponsored streams rise more slowly, reaching about +0.07 after 30 hours.

**Pitfalls.**
- **Few clusters.** Standard errors are clustered by game, and you only have 20 games (often fewer per study, since games with events in the window drop out as controls). With so few clusters, p-values are approximate and the pre-trend Wald test over-rejects: on the synthetic data, placebo samples covering 5–8 games often show pre-trend p ≈ 0.00 even though nothing happened. Treat the plot and the placebo as the main evidence and the p-value as a rough check.
- **Exclude every event from the controls.** `event_stack(..., exclude=events)` drops any game with a large *or* sponsored stream in the window from the control group. Passing only the large events would let sponsored streams lift the controls and shrink the estimate.
- **`log_players_ds`, not `log_players`.** Games have their own daily cycles (an EU-heavy game peaks at a different UTC hour than an NA-heavy one). The deseasonalized outcome removes each game's average hour-of-week pattern so the cycle doesn't look like an effect.
- **`min_gap_hours` trades clean pre-periods against sample size.** Try 48 and 96 and report how the estimate moves.
- **A stream on launch day or patch day isn't a stream effect.** If you know update dates, drop events within a day of them (add a column and filter `events`).

**What to write in the README.** The event plot, the number of clean events, the pre-trend result in words ("flat before, +X% after, lasting about Y hours"), and the placebo.

---

## Step 3: Streamer tiers

*Notebook: `03_streamer_tiers.ipynb` · Status: Tested on substitute.*

**Goal:** compare small, mid-size, and mega streamers on engagement lift *per 1,000 viewer-hours*, the unit you'd actually buy.

**Concept.** The event study is clean but only covers big streams. To compare tiers, use all hours: regress each game's deseasonalized player count on viewer-hours from each tier over the last 0–5, 6–23, and 24–71 hours. Subtracting each hour's average across games absorbs anything that hits all games at once. Adding the Trends control absorbs some of the hype that streamers chase. Summing coefficient × window width gives the cumulative lift from 1,000 extra viewer-hours, in log-point-hours, which converts to extra player-hours.

**Code.**

```python
from streamads import tiers as tr

panel = pd.read_parquet(PROCESSED / f"panel_{TAG}.parquet")

lag_naive, cum_naive, _ = tr.dose_response(panel, outcome="log_players_ds")
lag, cum, fit = tr.dose_response(panel, outcome="log_players_ds", controls=["log_interest"])
print(cum_naive.round(4))
print(cum.round(4))
```

```python
if USE_SYNTHETIC:  # compare with the planted answer
    truth = pd.Series(sim["truth"]["marginal_per_1k_vh"], name="truth")
    print(pd.concat([cum_naive.set_index("tier")["cum_lift"].rename("naive"),
                     cum.set_index("tier")["cum_lift"].rename("with_trends"), truth], axis=1).round(4))
```

```python
per_1k = tr.to_player_hours(cum, panel)
print("typical game:", round(per_1k.attrs["typical_players"]), "concurrent players")
fig = px.bar(per_1k, x="tier", y="player_hours",
             error_y=per_1k["player_hours_high"] - per_1k["player_hours"],
             error_y_minus=per_1k["player_hours"] - per_1k["player_hours_low"],
             title="Extra player-hours per 1,000 viewer-hours, by streamer tier")
fig.write_html(FIGURES / f"03_lift_per_tier_{TAG}.html")
per_1k.to_parquet(PROCESSED / f"tier_lift_{TAG}.parquet")
```

```python
# Robustness: does the ranking survive other tier cutoffs?
twitch = pd.read_parquet(PROCESSED / f"twitch_{TAG}.parquet")
base_cols = [c for c in panel.columns if not c.startswith("vh_") and c != "n_streams"]
for cuts in [(500, 5_000), (2_000, 20_000)]:
    alt = panels.assign_tiers(twitch, cutoffs=cuts)
    alt_panel = panels.add_exposure(panel[base_cols], panels.game_hour_exposure(twitch, alt))
    _, alt_cum, _ = tr.dose_response(alt_panel, controls=["log_interest"])
    print(cuts, alt_cum.set_index("tier")["cum_lift"].round(4).to_dict())
```

**Checkpoint (synthetic, seed 0).** The planted answer is mid ≈ 0.021 > small ≈ 0.015 > mega ≈ 0.012 log-point-hours per 1,000 viewer-hours. The naive regression says small ≈ 0.065: over four times too big, because small streamers in the simulation pile onto games whose hype is about to rise. Adding Trends cuts that bias by about two-thirds (small ≈ 0.023) but doesn't remove it, and mid comes out low (≈ 0.011). That's the honest lesson: controls reduce selection bias; they don't eliminate it. Report both numbers and say which direction any remaining bias goes.

**Pitfalls.**
- **Viewer-hours aren't independent of hype.** That's the whole catch. Show the naive and controlled estimates side by side; the gap is your evidence that selection matters.
- **Linear in viewer-hours.** This step gives the average marginal effect at current levels. Saturation (diminishing returns) is Step 4's job.
- **Mega streams are rare.** The mega estimate leans on a few dozen streamers; its interval will be wide.

**What to write in the README.** The bar chart, both estimates, and one sentence on what the tier ranking means for buying streams.

---

## Step 4: Response curves and budget split

*Notebook: `04_response_curves_budget.ipynb` · Status: 4a Tested on substitute; 4b Untested.*

**Goal:** turn lift per viewer-hour into dollars, account for diminishing returns, and find the split across tiers that buys the most engagement.

**Concept.** A marketing mix model (MMM) explains an outcome (daily players) with each channel's spend, transformed twice: **adstock** spreads a stream's effect over the following days, and **saturation** makes each extra viewer-hour worth a bit less than the last. Here the "channels" are the three streamer tiers. Once you have each tier's response curve and a cost per 1,000 viewer-hours, the best budget split is the one where the last dollar buys the same lift in every tier.

### 4a. Fast least-squares MMM (rehearsal and sanity check)

```python
from streamads import mmm

panel = pd.read_parquet(PROCESSED / f"panel_{TAG}.parquet")
daily = mmm.daily_panel(panel.drop(columns=["interest", "log_interest"]), trends)

fit_naive = mmm.fit_mmm(daily, controls=[])
fit = mmm.fit_mmm(daily, controls=["interest"])
print("without Trends:", {c: round(v, 3) for c, v in fit_naive.beta.items()})
print("with Trends:   ", {c: round(v, 3) for c, v in fit.beta.items()})
print("carryover alpha:", {c: round(float(v), 2) for c, v in fit.alpha.items()},
      "| half-saturation k:", {c: round(float(v), 1) for c, v in fit.k.items()})
```

```python
# Response curves: lift (log points) vs. daily viewer-hours per game
fig = go.Figure()
for c in fit.channels:
    x = np.linspace(0, daily[f"vh_{c}"].quantile(0.99) * 1.5, 100)
    fig.add_scatter(x=x, y=fit.response(c, x), name=c)
fig.update_layout(title="Response curves by tier", xaxis_title="daily viewer-hours (thousands)",
                  yaxis_title="lift in daily players (log points)")
fig.write_html(FIGURES / f"04_response_curves_{TAG}.html")
```

**The cost assumption.** Twitch doesn't publish sponsorship prices, so you have to assume a cost per 1,000 viewer-hours by tier. The numbers below are **placeholders, not sourced rates**. Replace them with figures from a public rate card or industry report, cite it in the README, and show how the answer changes if they're off by 50%.

```python
COST_PER_1K_VIEWER_HOURS = {"small": 15.0, "mid": 20.0, "mega": 30.0}  # ASSUMPTION: replace with a sourced rate
DAILY_BUDGET_PER_GAME = 5_000

split = mmm.optimize_split(fit, DAILY_BUDGET_PER_GAME, COST_PER_1K_VIEWER_HOURS)
print(split.round(4))
```

At the optimum, `marginal_lift_per_1k$` (extra log points from the next $1,000) is equal across funded tiers. That's the check that the optimizer worked.

```python
# Uncertainty: refit on games resampled with replacement (about 90 seconds for 30 draws)
draws = mmm.bootstrap_split(daily, DAILY_BUDGET_PER_GAME, COST_PER_1K_VIEWER_HOURS, n=30, controls=["interest"])
summary = draws.describe(percentiles=[0.1, 0.5, 0.9]).T[["10%", "50%", "90%"]]
print(summary.round(2))
fig = px.box(draws.melt(var_name="tier", value_name="share"), x="tier", y="share",
             title="Recommended budget share by tier (bootstrap over games)")
fig.write_html(FIGURES / f"04_budget_split_{TAG}.html")
```

```python
# Sensitivity to the cost assumption: mega 50% cheaper or 50% pricier
for mult in [0.5, 1.5]:
    costs = {**COST_PER_1K_VIEWER_HOURS, "mega": COST_PER_1K_VIEWER_HOURS["mega"] * mult}
    print(mult, mmm.optimize_split(fit, DAILY_BUDGET_PER_GAME, costs).set_index("tier")["share"].round(2).to_dict())
```

**Checkpoint (synthetic, seed 0).** Without the Trends control the fit is unstable (betas in the dozens, or negative). With it, mid's beta lands near the planted 0.30. The point split comes out roughly 25% small / 64% mid / 11% mega, with `marginal_lift_per_1k$` equal (about 0.020) in all three tiers. The bootstrap median is mid-heavy too (24% / 63% / 8%), with wide bands: mid's share runs from 43% to 96% between the 10th and 90th percentiles, and small's from 0% to 47%. Halving or raising mega's cost by 50% only moves mega between 9% and 13%.

### 4b. Bayesian MMM with PyMC-Marketing

*Status: Untested. Written against pymc-marketing 1.3.0, the current release; check its changelog if you install a newer one.*

Use this for the final numbers: it gives a posterior for every curve, so the uncertainty comes for free. It fits one model across games with `dims=("game",)`.

```python
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.mmm import MMM

X = daily.rename(columns={"steam_appid": "game"})
X["game"] = X["game"].astype(str)
X["date"] = X["date"].dt.tz_localize(None)
y = X.pop("log_players")
X = X[["date", "game", "vh_small", "vh_mid", "vh_mega", "interest"]]

model = MMM(
    date_column="date",
    target_column="log_players",
    channel_columns=["vh_small", "vh_mid", "vh_mega"],
    control_columns=["interest"],
    adstock=GeometricAdstock(l_max=7),
    saturation=LogisticSaturation(),
    dims=("game",),
)
model.fit(X, y, random_seed=42)
model.sample_posterior_predictive(X)

curves = model.sample_saturation_curve()   # posterior response curves per channel
adstock = model.sample_adstock_curve()     # how long each tier's effect lasts
```

```python
# Budget split over the next 4 weeks. cost_per_unit converts dollars into the model's
# channel units (thousands of viewer-hours): units = dollars / cost.
start = X["date"].max() + pd.Timedelta(days=1)
end = start + pd.Timedelta(days=27)
window = pd.date_range(start, end, freq="D")
cost = pd.DataFrame({"date": window, **{f"vh_{t}": COST_PER_1K_VIEWER_HOURS[t] for t in ["small", "mid", "mega"]}})

optimizer = model.budget_optimizer(start, end, cost_per_unit=cost)
# total_budget is spend per period (per day) across every game × tier cell
result = optimizer.allocate_budget(total_budget=DAILY_BUDGET_PER_GAME * X["game"].nunique())
print(result.budgets.to_dataframe())
```

The model's `log_players` target means the curves are in log points, like 4a. Compare its channel effects with 4a's: if they disagree badly, look at the priors (`model.model_config`) before trusting either.

**Pitfalls.**
- **Daily, not hourly.** MMMs are built for daily or weekly data. Hourly noise swamps the curves.
- **Without a hype control, the curves are meaningless.** 4a shows it: drop `interest` and the fit falls apart.
- **The cost assumption drives the answer.** The ranking per *viewer-hour* comes from data; the ranking per *dollar* depends on your prices. Say which is which.

**What to write in the README.** The response curves, the recommended split with its bootstrap or posterior range, the cost assumption and its source, and the sensitivity result.

---

## Finishing: writing the recommendation

Fill in the README's Recommendation section in this shape:

> Shift the streamer budget toward **[tier]**: about **X%** to mid-size, **Y%** to small, **Z%** to mega (range across bootstrap draws: …). A large stream lifts a game's concurrent players by about **N%** for **H** hours, with no rise beforehand. Per 1,000 viewer-hours, mid-size streams add about **P** player-hours versus **Q** for mega streams, which saturate. This assumes sponsorship costs of $A / $B / $C per 1,000 viewer-hours ([source]); if mega streams cost half as much, the split moves to ….

Use three charts, saved in `reports/figures/`:
1. **Event study** (Step 2): flat before, a jump after. It answers "do streams work at all?"
2. **Lift per 1,000 viewer-hours by tier** (Step 3), naive vs. controlled. It answers "which tier is most efficient?" and shows you handled selection.
3. **Budget split with uncertainty** (Step 4): the recommendation.

Export the HTML charts to PNG for the README with `fig.write_image(...)` (needs `pip install kaleido`) or take screenshots.

---

## Interview talking points

- **Why build a collector?** Twitch and Steam only expose the present. The data didn't exist, so I built an hourly git-scraping pipeline on GitHub Actions and let it run for months.
- **The catch, in one sentence.** Streamers pick games that are already trending, so a naive comparison credits streams for growth that was coming anyway.
- **How I handled it.** Clean events with a pre-trend check, controls that hadn't been treated yet, a placebo test, and Google Trends as a hype control. In simulation, the naive estimate was 4× too big; the controls removed most, not all, of the bias, and I report that rather than hide it.
- **Per viewer-hour vs. per dollar.** The data ranks tiers per viewer-hour. Turning that into a budget needs a price assumption, which I made explicit and stress-tested.
- **Why an MMM after an event study?** The event study shows streams cause a lift; the MMM adds carryover and diminishing returns, which is what a budget decision needs.
- **What I'd do with internal data.** Replace viewer-hours with actual sponsorship spend and Steam players with installs and revenue, and run a geo or time-based holdout test to validate the MMM.

---

## Appendix: the synthetic rehearsal data

`streamads.synthetic.make_synthetic(seed=0)` returns Steam and Twitch tables with the collector's exact columns, plus daily Trends interest. It simulates 12 of the tracked games over 9 weeks:

- **Players** follow a game-specific daily cycle, a weekend bump, a slow hype random walk, and hourly noise.
- **Streams** come from pools of small (150), mid (20), and mega (3) streamers per game. They go live more often when the game's hype is about to rise: the selection problem.
- **The planted effect:** each tier's concurrent viewers feed an exponentially decaying stock (carryover) that raises log players through a saturating curve. Per extra 1,000 viewer-hours, mid-size streams are the most efficient, then small, then mega. The true values are in `sim["truth"]["marginal_per_1k_vh"]`.
- **Collector quirks:** timestamps drift past minute 17, about 2% of hourly runs are skipped, and only the top 300 streams per game-hour are kept.

Use it to check that each step recovers the planted answer before you trust it on real data. Delete the `*_synthetic.*` files in `data/processed/` and `reports/figures/` when you switch.
