"""gta6-trophy-rarity-forecast-01: deterministic trophy earn-rate forecaster.

Forecasts, for every trophy of Grand Theft Auto VI (PS5), its earn rate on
Exophase at the case's target, as five quantiles (5/25/50/75/95%) in percent.

Method (stdlib only, no network, no ML, fully deterministic):
  Per trophy, two independent extrapolations of the case history are combined
  in logit space:
    1. Trend fit: weighted least squares of logit(rate) against
       ln(1 + hours since release 2026-11-19T00:00Z), weights decaying with a
       12-hour half-life so recent reads matter most. The fitted line is read
       at the target with the extrapolation distance damped by 15%.
    2. Count model: earners E ~= rate * players_tracked / 100 are counted
       per read. ln(E) and ln(players_tracked) are each fitted against the
       same log-time axis and extrapolated; the forecast rate is E/N at the
       target. This separates real earning growth from dilution: when many new
       players arrive, the rate can fall even while earners keep growing.
  The median is the (weighted) mean of the two models' logits; with too
  little history it falls back to persistence (the last read's rate).
  The spread (logit sigma) combines a random-walk term growing with
  sqrt(horizon), the trend fit's residual, a model-error term growing with
  extrapolation distance, and a refresh term: Exophase only refreshes its
  counts on its own schedule, so a short-horizon forecast carries real risk
  of an unseen refresh jump. Quantiles are median + z_tau * sigma on the
  logit scale, rounded half-up to 4 decimals and clamped to [0.01, 99.99].

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(moment):
    """ln(1 + hours since release); negative hours (pre-release reads) clamp."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log(1.0 + max(hours, -0.99))


def wlsq(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return a, b, math.sqrt(var)


def distinct_snapshots(history):
    """Collapse consecutive reads with identical rates+players to one snapshot.

    Keeps the time the snapshot first appeared; identical re-reads carry no
    new information for a trend fit, they would only inflate its weight.
    """
    snaps = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read["rates"]))
        if key != prev_key:
            snaps.append(read)
            prev_key = key
    return snaps


def forecast_trophy(series, x_target, days):
    """series: list of (x, time_hours, rate, players). Returns (median_logit, sigma).

    Two lessons drive the shape here. First, earn curves saturate: a trophy's
    rate rises steeply just after release and then flattens as the pool of
    players who can earn it soon is exhausted, so a straight line fitted over
    the whole history and read far ahead overshoots badly. The trend is
    therefore fitted only on the last 24 hours of snapshots, and the
    extrapolated gain is damped by 1/(1+dx) so the projected logit gain is
    concave in log-time. Second, the total move from the last read is capped:
    no trophy's logit rate moves more than +2.5 / -1.5 over the horizon.
    """
    last_rate = series[-1][2]
    last_x = series[-1][0]
    last_logit = logit(last_rate)
    dx = max(0.0, x_target - last_x)
    decay = 1.0 / (1.0 + dx)  # saturating extrapolation in log-time
    x_eff = last_x + 0.85 * dx

    # Recency weights: half-life 12 hours from the last snapshot.
    t_last = series[-1][1]
    ws = [0.5 ** ((t_last - t) / 12.0) for _, t, _, _ in series]
    recent = [s for s in series if s[1] >= t_last - 24.0]
    recent_ws = ws[len(series) - len(recent):]

    logits = []
    weights = []
    resid = 0.25

    # --- Model 1: logit(rate) trend in log-time, fitted on the last 24h.
    fit_series, fit_ws = None, None
    if len(recent) >= 3 and (recent[-1][1] - recent[0][1]) >= 4.0:
        fit_series, fit_ws = recent, recent_ws
        shrink = 1.0
    elif len(series) >= 3 and (series[-1][1] - series[0][1]) >= 2.0:
        fit_series, fit_ws = series, ws
        shrink = 0.5
    if fit_series is not None:
        fit = wlsq([s[0] for s in fit_series], [logit(s[2]) for s in fit_series], fit_ws)
        if fit is not None:
            a, b, resid = fit
            b = min(2.0, max(-0.75, b)) * shrink
            at_last = a + b * fit_series[-1][0]
            logits.append(at_last + b * dx * decay)
            weights.append(0.6)
            resid = min(0.9, max(0.12, resid))

    # --- Model 2: earners / players-tracked count model.
    pos = [(s, math.log(max(s[2] * s[3] / 100.0, 0.5))) for s in series if s[2] > 0 and s[3]]
    if len(pos) >= 3:
        xs_e = [s[0] for s, _ in pos]
        fit_e = wlsq(xs_e, [le for _, le in pos], ws[-len(pos):])
        fit_n = wlsq([s[0] for s in series], [math.log(s[3]) for s in series], ws)
        if fit_e is not None and fit_n is not None:
            ae, be, _ = fit_e
            an, bn, _ = fit_n
            be = min(2.5, max(0.0, be))
            bn = min(1.5, max(0.0, bn))
            # Earners saturate like the rate does; players-tracked keeps
            # growing, so only the earners' extrapolation is damped.
            e_tgt = math.exp(ae + be * (xs_e[-1] + dx * decay))
            n_tgt = math.exp(an + bn * x_eff)
            if n_tgt > 0:
                logits.append(logit(100.0 * e_tgt / n_tgt))
                weights.append(0.4)

    if logits:
        median_logit = sum(w * l for w, l in zip(weights, logits)) / sum(weights)
    else:
        median_logit = last_logit
    # Cap the total move from the last read's rate.
    median_logit = min(last_logit + 2.5, max(last_logit - 1.5, median_logit))

    # --- Spread (logit sigma): random walk + fit residual + extrapolation
    # model error + unseen-Exophase-refresh risk (large at short horizons).
    rw = 0.5 * math.sqrt(days)
    slope_unc = min(0.5, 0.20 * dx)
    refresh = 0.45 * math.exp(-days / 2.0)
    raw_sigma = math.sqrt(max(rw, 0.30) ** 2 + resid ** 2 + slope_unc ** 2 + refresh ** 2)
    # Calibration 0.85: on the synthetic final-like backtest (3 scenarios x 3
    # seeds) the raw band scored coverage 1.00, i.e. wider than honest; 0.85
    # cut the loss further while keeping coverage >= 0.98 there. The full
    # optimum in that smooth simulation was 0.75, but the real recording is
    # noisier than the simulation, so only part of the narrowing is taken.
    sigma = 0.85 * raw_sigma

    # --- Edge cases: un-earned and fully-earned trophies.
    if last_rate <= 0.0:
        # Nobody has earned it yet in the window; hedge the median up a little
        # (logit losses punish a near-zero median far more than a small one)
        # and keep the band wide: it may start moving during the week.
        median_logit = max(median_logit, logit(0.05))
        sigma = max(sigma, 1.0)
    elif last_rate >= 100.0:
        median_logit = min(median_logit, logit(99.9))
        sigma = max(sigma, 0.6)
    return median_logit, sigma


def forecast_case(case):
    if not case.get("history"):
        raise ValueError("case has no history")
    days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = log_time(target)
    snaps = distinct_snapshots(case["history"])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        series = []
        for read in snaps:
            rate = read["rates"][i]
            if rate is None:
                continue
            moment = parse_time(read["at"])
            series.append((log_time(moment),
                           (moment - RELEASE).total_seconds() / 3600.0,
                           float(rate),
                           read.get("players_tracked") or 0))
        if not series:
            median_logit, sigma = logit(1.0), 1.0
        else:
            median_logit, sigma = forecast_trophy(series, x_target, days)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(sigmoid(median_logit + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Never crash a case: persistence rows with a wide band."""
    rows = []
    last = case["history"][-1] if case.get("history") else None
    for i, trophy in enumerate(case["trophies"]):
        rate = None
        if last is not None and last["rates"][i] is not None:
            rate = float(last["rates"][i])
        med = logit(rate if rate is not None else 1.0)
        qs = [round4(clamp_pct(sigmoid(med + z * 1.0))) for z in Z]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
