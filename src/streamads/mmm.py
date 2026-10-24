"""A small marketing mix model you can fit in seconds: adstock + saturation, by least squares.

It's a stand-in for the Bayesian PyMC-Marketing model: same building blocks
(geometric adstock for carryover, a saturation curve for diminishing returns),
no posterior. Use it to rehearse the pipeline and sanity-check the Bayesian fit.

Model, per game g and day d:
    log_players[g,d] = game FE + day-of-week FE + gamma * trends[g,d]
                       + sum_c beta_c * x_c / (x_c + k_c),   x_c = adstock(vh_c, alpha_c)
where vh_c is viewer-hours (thousands) from tier c on that game-day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TIERS = ["small", "mid", "mega"]


def daily_panel(panel: pd.DataFrame, trends: pd.DataFrame | None = None) -> pd.DataFrame:
    """Game × day table: mean players, new reviews, viewer-hours (thousands) by tier, trends."""
    p = panel.assign(date=panel["hour"].dt.floor("D"))
    agg = {"current_players": "mean", "new_reviews": "sum",
           **{f"vh_{t}": "sum" for t in TIERS}, "vh_sponsored": "sum"}
    daily = p.groupby(["steam_appid", "date"]).agg(agg).reset_index()
    for t in [*TIERS, "sponsored"]:
        daily[f"vh_{t}"] = daily[f"vh_{t}"] / 1_000
    daily["log_players"] = np.log1p(daily["current_players"])
    daily["dow"] = daily["date"].dt.dayofweek
    if trends is not None:
        daily = daily.merge(trends[["steam_appid", "date", "interest"]], on=["steam_appid", "date"], how="left")
        daily["interest"] = daily.groupby("steam_appid")["interest"].transform(lambda s: s.ffill().bfill())
    return daily.sort_values(["steam_appid", "date"]).reset_index(drop=True)


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 7) -> np.ndarray:
    """Carryover with weights alpha**lag, normalized to sum to 1 (steady state = x)."""
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    return np.convolve(x, w)[: len(x)]


def saturation(x: np.ndarray, k: float) -> np.ndarray:
    """Michaelis-Menten curve: 0 at x=0, half of the max effect at x=k."""
    return x / (x + k)


@dataclass
class MMMFit:
    channels: list[str]
    alpha: dict[str, float]
    k: dict[str, float]
    beta: dict[str, float]
    r2: float
    data: pd.DataFrame

    def response(self, channel: str, x: np.ndarray) -> np.ndarray:
        """Steady-state lift in log points at daily viewer-hours x (thousands)."""
        return self.beta[channel] * saturation(np.asarray(x, dtype=float), self.k[channel])


def _design(daily: pd.DataFrame, channels: list[str], alpha: dict, k: dict, controls: list[str]):
    sat = {}
    for c in channels:
        ad = daily.groupby("steam_appid")[f"vh_{c}"].transform(lambda s: geometric_adstock(s.to_numpy(), alpha[c]))
        sat[c] = saturation(ad.to_numpy(), k[c])
    fe = pd.get_dummies(daily["steam_appid"].astype(str), prefix="g", dtype=float)
    dow = pd.get_dummies(daily["dow"], prefix="dow", drop_first=True, dtype=float)
    X = np.column_stack([*sat.values(), daily[controls].to_numpy(dtype=float), fe.to_numpy(), dow.to_numpy()])
    return X


def fit_mmm(daily: pd.DataFrame, channels: list[str] = TIERS, controls: list[str] | None = None,
            outcome: str = "log_players") -> MMMFit:
    """Fit alpha and k per channel by profiling: for each (alpha, k), the rest is OLS."""
    controls = controls or []
    d = daily.dropna(subset=[outcome, *controls]).reset_index(drop=True)
    y = d[outcome].to_numpy()
    scale = {c: max(d[f"vh_{c}"].replace(0, np.nan).median(), 1e-3) for c in channels}

    def unpack(theta):
        a = {c: 1 / (1 + np.exp(-theta[2 * i])) for i, c in enumerate(channels)}
        k = {c: scale[c] * np.exp(theta[2 * i + 1]) for i, c in enumerate(channels)}
        return a, k

    def sse(theta):
        a, k = unpack(theta)
        X = _design(d, channels, a, k, controls)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ coef) ** 2).sum())

    best = minimize(sse, np.zeros(2 * len(channels)), method="Nelder-Mead",
                    options={"maxiter": 2_000, "xatol": 1e-3, "fatol": 1e-6})
    a, k = unpack(best.x)
    X = _design(d, channels, a, k, controls)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    beta = {c: float(coef[i]) for i, c in enumerate(channels)}
    return MMMFit(channels, a, k, beta, float(r2), d)


def optimize_split(fit: MMMFit, budget: float, cost_per_1k_vh: dict[str, float],
                   min_share: float = 0.0) -> pd.DataFrame:
    """Daily budget split across tiers that maximizes total steady-state lift.

    cost_per_1k_vh converts dollars to viewer-hours (thousands): x_c = spend_c / cost_c.
    """
    ch = fit.channels
    cost = np.array([cost_per_1k_vh[c] for c in ch])

    def neg_lift(share):  # optimize shares, not dollars, so the problem is well scaled
        return -sum(fit.response(c, share[i] * budget / cost[i]) for i, c in enumerate(ch))

    res = minimize(neg_lift, np.full(len(ch), 1 / len(ch)), method="SLSQP",
                   bounds=[(min_share, 1)] * len(ch),
                   constraints=[{"type": "eq", "fun": lambda share: share.sum() - 1}],
                   options={"ftol": 1e-12})
    spend = res.x * budget
    out = pd.DataFrame({"tier": ch, "spend": spend, "share": spend / budget,
                        "viewer_hours_k": spend / cost,
                        "lift_log": [fit.response(c, spend[i] / cost[i]) for i, c in enumerate(ch)]})
    # Marginal lift per extra $1,000 (log points): equal across funded tiers at the optimum.
    eps = budget * 1e-4
    out["marginal_lift_per_1k$"] = [1_000 * (fit.response(c, (spend[i] + eps) / cost[i])
                                             - fit.response(c, spend[i] / cost[i])) / eps
                                    for i, c in enumerate(ch)]
    return out


def bootstrap_split(daily: pd.DataFrame, budget: float, cost_per_1k_vh: dict[str, float],
                    n: int = 30, controls: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    """Refit on games resampled with replacement; return one row of tier shares per draw.

    The spread of these shares is the honest uncertainty on the recommended split.
    """
    rng = np.random.default_rng(seed)
    games = daily["steam_appid"].unique()
    draws = []
    for b in range(n):
        pick = rng.choice(games, size=len(games), replace=True)
        boot = pd.concat([daily[daily["steam_appid"] == g].assign(steam_appid=i) for i, g in enumerate(pick)])
        fit = fit_mmm(boot, controls=controls)
        split = optimize_split(fit, budget, cost_per_1k_vh)
        draws.append(split.set_index("tier")["share"].rename(b))
    return pd.DataFrame(draws).reset_index(drop=True)
