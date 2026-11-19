"""Forecast GTA VI trophy earn rates on Exophase.

The program reads one case per *.in file (schema makefun/gta6-trophies-case/v1)
and writes one <id>.out forecast per case: five quantiles (5/25/50/75/95%) of
each trophy's earn rate, in percent, at the case's target time.

How the forecast is built, per trophy:

  * Earners and tracked players are modelled separately.  A trophy's rate is
    earners / players_tracked, and the two move for different reasons: players
    keep playing and earning trophies, while Exophase only refreshes the page
    every few hours, so each refresh dilutes every rate as newly tracked
    players (who hold far fewer trophies) join the denominator.  Fitting the
    rate alone confuses the two; fitting earners and players separately, each
    against log time since release, keeps them apart.
  * When the whole history is flat -- every read identical, players tracked
    unchanged -- no trend can be fitted.  That means the page has not
    refreshed since the first probe, so a refresh is overdue: the next refresh
    multiplies the tracked players several-fold and drags every rate down
    towards the newcomers' much lower earn rates.  The median then assumes
    that refresh lands before the target, and the spread stays wide and
    lopsided, reaching the no-refresh rate at the top and a bigger refresh
    at the bottom.
  * When the history moves, a least-squares line is fitted to the logit of
    the rate against log(1 + hours since release), and a second pair of lines
    to log earners and log players tracked.  Both are read at the target and
    averaged on the logit scale.  The slope is shrunk towards zero when the
    fit is weak, the extrapolated move is capped, and a mild tier prior
    (platinum barely moves, early bronze climbs) pulls the median a quarter
    of the way towards what that tier usually does over the horizon.
  * The spread is honest about the horizon: it grows with the fit residual,
    with how far the median moved from the last read, and with the horizon
    itself.  A forecast that is too narrow pays for every trophy outside it;
    one too wide pays at every trophy.

Everything is deterministic: fixed arithmetic on the case file only, no
randomness, no clock, no network.  Python 3.13 standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at those levels.
ZS = (-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722)

LO = 0.01
HI = 99.99

# How a trophy of each tier is expected to move, in logit units over a
# 4.5-day horizon, before any trend is fitted: the platinum is earned by
# almost nobody new in a week, while ordinary bronze trophies climb as
# players work through the game.  Only a mild prior; the fitted trend
# carries three quarters of the median.
TIER_DRIFT = {
    "platinum": -0.8,
    "gold": 0.5,
    "silver": 0.8,
    "bronze": 1.2,
}

# Median rate (percent) used when a case arrives with no history at all.
TIER_PRIOR_RATE = {
    "platinum": 0.25,
    "gold": 6.0,
    "silver": 10.0,
    "bronze": 18.0,
}


def _clamp_pct(x):
    """Keep a percent inside the range the logit is defined on."""
    return min(HI, max(LO, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    """Round to 4 decimals, halves up, returning an int for whole values."""
    q = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    f = float(q)
    return int(f) if f == int(f) else f


def _parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


def _least_squares(xs, ys):
    """Ordinary least squares of ys on xs.

    Returns (slope, intercept, residual_std, r_squared).  With fewer than
    two points, or no spread in xs, the slope is zero.
    """
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sse / sst if sst > 0.0 else 0.0
    std = math.sqrt(sse / max(1, n - 2)) if n > 2 else 0.0
    return slope, intercept, std, r2


def _quantiles_from_logit(median_logit, sigma_lo, sigma_hi):
    """Five quantiles around a logit median, with a possibly lopsided spread."""
    out = []
    for z in ZS:
        sigma = sigma_lo if z < 0 else sigma_hi
        out.append(_round4(_clamp_pct(_inv_logit(median_logit + z * sigma))))
    # Rounding can only tie neighbours, never reorder them, but be safe.
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def _flat_history_forecast(last_rate, tier, horizon_hours):
    """Forecast when every read so far is identical.

    The page has not refreshed since the first probe, so the next refresh is
    overdue.  Expect the tracked-player count to multiply (about five-fold
    over a 90-minute horizon, more over longer ones, capped at twelve-fold)
    while the newcomers hold each trophy at only a small fraction of the old
    rate, so the median rate is the diluted one.  The upper quantiles reach
    back to the no-refresh rate; the lower ones allow a still bigger refresh.
    """
    if last_rate >= 99.5:
        return [99.0, 99.5, 99.9, 99.99, 99.99]
    growth = 1.0 + 4.5 * math.sqrt(max(horizon_hours, 0.0) / 1.5)
    growth = min(growth, 12.0)
    # Newcomers earn a trophy at ~7% of the old rate (platinum: ~2%).
    newcomer_ratio = 0.02 if tier == "platinum" else 0.07
    factor = (1.0 + (growth - 1.0) * newcomer_ratio) / growth
    median = _clamp_pct(last_rate * factor)
    median_logit = _logit(median)
    gap = _logit(last_rate) - median_logit  # logit distance to no-refresh
    sigma_hi = max(0.5, gap / 1.6448536269514722)
    sigma_lo = 0.55
    return _quantiles_from_logit(median_logit, sigma_lo, sigma_hi)


def _moving_history_forecast(reads, last_rate, tier, horizon_days,
                             target_x, release):
    """Forecast when the history shows movement.

    `reads` is a list of (x, rate, players) with x = log(1 + hours since
    release).  Two views are combined: a line fitted to the logit's rate,
    and separate lines fitted to log earners and log players, read at the
    target and divided.  Both slopes are shrunk when their fit is weak, and
    the move from the last read is capped.
    """
    xs = [r[0] for r in reads]
    logits = [_logit(r[1]) for r in reads]
    slope, intercept, resid, r2 = _least_squares(xs, logits)
    shrink = min(1.0, max(0.25, r2))
    slope *= shrink
    last_x = xs[-1]
    dx = target_x - last_x
    move = slope * dx
    move = min(4.0, max(-3.0, move))
    trend_logit = _logit(last_rate) + move

    # Earners / players view.
    earn_xs, earn_ys, play_xs, play_ys = [], [], [], []
    for x, rate, players in reads:
        if players is None or players <= 0:
            continue
        earners = max(0.0, rate * players / 100.0)
        earn_xs.append(x)
        earn_ys.append(math.log(earners + 0.5))
        play_xs.append(x)
        play_ys.append(math.log(players))
    combined_logit = trend_logit
    if len(earn_xs) >= 4 and len(play_xs) >= 4:
        es, ei, _, er2 = _least_squares(earn_xs, earn_ys)
        ns, ni, _, nr2 = _least_squares(play_xs, play_ys)
        es *= min(1.0, max(0.25, er2))
        ns *= min(1.0, max(0.25, nr2))
        earners_hat = math.exp(ei + es * target_x) - 0.5
        players_hat = math.exp(ni + ns * target_x)
        players_hat = max(players_hat, reads[-1][2] or 0.0)
        if earners_hat > 0.0 and players_hat > 0.0:
            rate_hat = _clamp_pct(100.0 * earners_hat / players_hat)
            combined_logit = 0.5 * trend_logit + 0.5 * _logit(rate_hat)

    # Mild tier prior: pull a quarter of the way towards what the tier
    # usually does over this horizon.
    drift = TIER_DRIFT.get(tier, 1.0) * horizon_days / 4.5
    prior_logit = _logit(last_rate) + drift
    median_logit = 0.75 * combined_logit + 0.25 * prior_logit
    delta = median_logit - _logit(last_rate)
    delta = min(4.5, max(-3.0, delta))
    median_logit = _logit(last_rate) + delta

    sigma = 0.45 + resid + 0.12 * abs(delta)
    sigma = min(1.6, max(0.55, sigma))
    return _quantiles_from_logit(median_logit, sigma, sigma)


def _forecast_trophy(trophy, column, history, horizon_minutes, target, release):
    tier = trophy.get("tier", "bronze")
    idx = column  # `rates` follows the trophies' order, so the column is the position
    horizon_hours = horizon_minutes / 60.0
    horizon_days = horizon_minutes / 1440.0

    if not history:
        rate = TIER_PRIOR_RATE.get(tier, 10.0)
        return _quantiles_from_logit(_logit(rate), 1.2, 1.2)

    # The last read's rate, falling back to the newest non-null one.
    last_rate = None
    for read in reversed(history):
        rate = read["rates"][idx]
        if rate is not None:
            last_rate = rate
            break
    if last_rate is None:
        rate = TIER_PRIOR_RATE.get(tier, 10.0)
        return _quantiles_from_logit(_logit(rate), 1.2, 1.2)
    if last_rate >= 99.5:
        return [99.0, 99.5, 99.9, 99.99, 99.99]

    # Build the (x, rate, players) series, oldest first.
    series = []
    for read in history:
        rate = read["rates"][idx]
        if rate is None:
            continue
        at = _parse_time(read["at"])
        x = math.log1p(max(0.0, _hours_between(release, at)))
        series.append((x, rate, read.get("players_tracked")))
    if not series:
        rate = TIER_PRIOR_RATE.get(tier, 10.0)
        return _quantiles_from_logit(_logit(rate), 1.2, 1.2)

    flat = (len({round(s[1], 9) for s in series}) == 1
            and len({s[2] for s in series}) == 1)
    if flat:
        if len(series) < 3 or horizon_hours <= 12.0:
            return _flat_history_forecast(last_rate, tier, horizon_hours)
        # Flat for a long horizon: the refresh prior still dilutes, but
        # real progression over days pulls the other way; split the
        # difference between the diluted median and persistence.
        diluted = _flat_history_forecast(last_rate, tier, horizon_hours)
        mid = _logit(diluted[2])
        pers = _logit(last_rate)
        blend = 0.5 * mid + 0.5 * pers
        return _quantiles_from_logit(blend, 0.9, 0.9)

    target_x = math.log1p(max(0.0, _hours_between(release, target)))
    return _moving_history_forecast(series, last_rate, tier,
                                    horizon_days, target_x, release)


def forecast_case(case):
    if case.get("levels") != list(LEVELS):
        raise ValueError("unexpected levels %r" % (case.get("levels"),))
    history = case.get("history") or []
    if not history:
        raise ValueError("case %s has no history" % case.get("id"))

    release_text = (case.get("game") or {}).get("release_date", "2026-11-19")
    try:
        release = _parse_time(release_text + "T00:00:00Z")
    except ValueError:
        release = _parse_time(history[0]["at"])
    target = _parse_time(case["target"])
    horizon_minutes = case["horizon_minutes"]

    # `rates` follows the trophies' order; the column of a trophy is its
    # position in that order.
    trophies = case["trophies"]
    rows = []
    for pos, trophy in enumerate(trophies):
        q = _forecast_trophy(trophy, pos, history, horizon_minutes,
                             target, release)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def _persistence_fallback(case):
    """Last-read rates with a wide spread: used only if the model fails."""
    last = (case.get("history") or [{}])[-1]
    rates = last.get("rates") or []
    rows = []
    for pos, trophy in enumerate(case.get("trophies") or []):
        rate = rates[pos] if pos < len(rates) and rates[pos] is not None else 10.0
        if rate >= 99.5:
            rows.append({"id": trophy["id"],
                         "q": [99.0, 99.5, 99.9, 99.99, 99.99]})
        else:
            rows.append({"id": trophy["id"],
                         "q": _quantiles_from_logit(_logit(rate), 1.0, 1.0)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception as exc:  # never lose a case to a crash
            print("warning: %s: %s; using persistence fallback"
                  % (path.name, exc), file=sys.stderr)
            forecast = _persistence_fallback(case)
        out_path = OUTPUT_DIR / ("%s.out" % path.stem)
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
