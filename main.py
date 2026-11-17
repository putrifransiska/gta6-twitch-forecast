"""GTA VI launch-day Twitch viewers, hour by hour: an analog-ensemble quantile forecast.

Standard library only, no network, deterministic. For each case it:
  1. turns every comparable launch into a release-day curve r_c(h) = ln(v_h / v_1), hours 2..24 (UTC);
  2. weights each comparable by how much its run-up to hour 1 looks like the case's (log ratios of hour 1 to the
     hours and days before it) and by its kind of unlock (rolling midnight local vs one global unlock);
  3. forecasts hour h as a weighted mixture of normals around the analog curves, in log space, whose width
     grows with the horizon, and writes that mixture's 5/25/50/75/95% quantiles times hour 1's print.

Run locally:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))

# Tunable parameters (fitted by leave-one-out backtests over the comparables).
PARAMS = {
    # Leave-one-out over the 8 console-only comparables: ~0.20 vs the persistence baseline's 0.323. The landscape is
    # flat near the optimum (0.193); these sit on the safe side (a trace of PC launches, a wider kernel), giving up
    # ~0.005 there for a better worst case and some mass on a global-unlock surge.
    "w_pc": 0.02,        # prior weight of a launch that includes PC (the target is console-only)
    "v1_unlocked": 2500.0,  # hour-1 prints well below this look not-yet-unlocked: PC/global analogs gain weight
    "max_viewers": 8e6,  # ceiling on an analog's level: a 1000x surge is only possible from a tiny hour 1
    # Shift each comparable by its DST difference to the target (US/EU summer time). Tested and left OFF: console-only
    # leave-one-out 0.2040 -> 0.2079 (4 better / 4 worse; the one winter console target, FF7 Rebirth, +0.036); wider
    # non-surging set 0.258 -> 0.248. Mixed, not a clear gain.
    "dst_align": False,
    "gate_power": 4.0,   # steepness of that gate
    "w_early": 0.6,      # extra factor for an early-access-contaminated launch
    "bw_x": 5.0,         # kernel bandwidth on the run-up drop x (wide: x mostly acts through the slope)
    "ridge": 8.0,        # shrinkage of the per-hour slope b_h toward 0 (in weighted-count units)
    "s0": 0.12,          # kernel width at hour 2 (log units)
    "s1": 0.08,          # kernel width growth per sqrt(hour): later hours get wider bands
}


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def series_from(points, key_time="from") -> dict:
    """{hour-start datetime: viewers} from a list of hourly rows."""
    out = {}
    for row in points:
        v = row.get("viewers")
        if v is None:
            continue
        out[parse_time(row[key_time])] = float(v)
    return out


def safe_log(x: float) -> float:
    return math.log(max(x, 1.0))


def lookup(series: dict, t: datetime, tol_h: int = 1):
    """Viewers at hour t, or the nearest point within tol_h hours, else None."""
    if t in series:
        return series[t]
    for d in range(1, tol_h + 1):
        for s in (t - timedelta(hours=d), t + timedelta(hours=d)):
            if s in series:
                return series[s]
    return None


def runup_x(series: dict, day_start: datetime, v1: float):
    """Run-up drop x = ln(mean viewers of the 6 hours before hour 1 / hour 1), None when unknown."""
    pre = [series[t] for t in (day_start - timedelta(hours=k) for k in range(1, 7)) if t in series]
    if not pre:
        return None
    return safe_log(sum(pre) / len(pre)) - safe_log(v1)


def comp_kind(comp: dict) -> tuple:
    """(console_only, early_access) from a comparable's platforms and unlock note."""
    early = "early-access-contaminated" in (comp.get("unlock") or "").lower()
    # "PC (PS5 2023-09-06)" or "Xbox Series, PC (PS5 later)": any platform token that is PC makes it a PC launch.
    tokens = [w.strip(" )") for w in (comp.get("platforms") or "").lower().replace("(", ",").split(",")]
    console_only = "pc" not in tokens
    return console_only, early


def us_dst(day: datetime) -> bool:
    """True when US daylight saving time is in force on this UTC date (second Sunday of March to first Sunday of
    November). Europe's summer time overlaps it almost exactly, so the whole world's evening runs an hour earlier in UTC."""
    d = day.date()
    march = datetime(d.year, 3, 8).date()
    start = march + timedelta(days=(6 - march.weekday()) % 7)
    nov = datetime(d.year, 11, 1).date()
    end = nov + timedelta(days=(6 - nov.weekday()) % 7)
    return start <= d < end


def build_analogs(case: dict, w_pc: float) -> list:
    """Every usable comparable as (features, curve, prior weight)."""
    analogs = []
    target_dst = us_dst(parse_time(case["day_start"]))
    for comp in case.get("comparables") or []:
        # The target's own pre-release category history is not a launch: never use it as an analog curve.
        if "target-history" in (comp.get("unlock") or "").lower():
            continue
        series = series_from(comp.get("hourly") or [])
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, ValueError):
            continue
        if PARAMS["dst_align"]:
            # Align the comparable's clock to the target's: a launch under DST unlocks in the US at 04:00Z, not 05:00Z,
            # so its hour h-1 is the target's hour h (and the reverse for a summer target and a winter comparable).
            day_start += timedelta(hours=int(target_dst) - int(us_dst(day_start)))
        v1 = series.get(day_start)
        if v1 is None or v1 < 20:
            continue
        curve = {}
        for h in HOURS:
            v = series.get(day_start + timedelta(hours=h - 1))
            if v is not None:
                curve[h] = safe_log(v) - safe_log(v1)
        if len(curve) < 18:
            continue
        # Fill gaps by linear interpolation (or the nearest known hour at the ends).
        known = sorted(curve)
        for h in HOURS:
            if h in curve:
                continue
            lo = max((k for k in known if k < h), default=None)
            hi = min((k for k in known if k > h), default=None)
            if lo is None:
                curve[h] = curve[hi]
            elif hi is None:
                curve[h] = curve[lo]
            else:
                curve[h] = curve[lo] + (curve[hi] - curve[lo]) * (h - lo) / (hi - lo)
        console_only, early = comp_kind(comp)
        prior = 1.0 if console_only else w_pc
        if early:
            prior *= PARAMS["w_early"]
        analogs.append((runup_x(series, day_start, v1), curve, prior, comp.get("name", "")))
    return analogs


def pc_weight(case: dict, v1: float) -> float:
    """Weight of PC / single-global-unlock analogs for this case.

    Small by default (GTA VI and the practice launches are console-only, rolling midnight local). It rises to 1 when
    the case's notes describe one simultaneous global unlock, and smoothly as hour 1's print falls well below what an
    already-unlocked big launch draws: in the comparables, every launch that went on to rise 10x-1000x within the day
    had an hour-1 print under 5,000 (nothing unlocked yet), and no other run-up feature told them apart. The gate is
    steep and set at 2,500 so the smallest unlocked console launch (Saros, 6,400) keeps its console-only weighting.
    """
    notes = (case.get("notes") or "").lower()
    rolling = "rolling" in notes or "midnight local" in notes
    if ("simultaneous" in notes or "global unlock" in notes) and not rolling:
        return 1.0
    gate = 1.0 / (1.0 + (v1 / PARAMS["v1_unlocked"]) ** PARAMS["gate_power"])
    return PARAMS["w_pc"] + (1.0 - PARAMS["w_pc"]) * gate


def mixture_quantile(centers: list, weights: list, sigma: float, tau: float) -> float:
    """tau-quantile of sum_i w_i N(c_i, sigma^2), by bisection on the CDF."""
    total = sum(weights)

    def cdf(x: float) -> float:
        acc = 0.0
        for c, w in zip(centers, weights):
            acc += w * 0.5 * (1.0 + math.erf((x - c) / (sigma * math.sqrt(2.0))))
        return acc / total

    lo = min(centers) - 8 * sigma
    hi = max(centers) + 8 * sigma
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case: dict) -> dict:
    p = PARAMS
    day_start = parse_time(case["day_start"])
    first = next(k for k in case["known"] if k["hour"] == 1)
    v1 = max(1.0, float(first["viewers"]))

    history = series_from(case.get("history_hourly") or [])
    if not history:
        # Minute reads (the final's history) -> whole-hour means, keyed by the hour's start.
        sums = {}
        for t, v in series_from(case.get("history") or [], key_time="at").items():
            hour = t.replace(minute=0, second=0, microsecond=0)
            acc = sums.setdefault(hour, [0.0, 0])
            acc[0] += v
            acc[1] += 1
        history = {t: s / n for t, (s, n) in sums.items()}
    history[day_start] = v1
    x_t = runup_x(history, day_start, v1)

    analogs = [a for a in build_analogs(case, pc_weight(case, v1)) if a[0] is not None]
    if not analogs:
        analogs = [(0.0, {h: 0.0 for h in HOURS}, 1.0, "persistence")]
    if x_t is None:
        x_t = sum(a[0] * a[2] for a in analogs) / sum(a[2] for a in analogs)
    weights = [prior * math.exp(-0.5 * ((x_t - x) / p["bw_x"]) ** 2) for x, _c, prior, _n in analogs]
    if sum(weights) <= 1e-300:
        weights = [a[2] for a in analogs]
    wsum = sum(weights)
    xbar = sum(w * a[0] for w, a in zip(weights, analogs)) / wsum

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        # Weighted ridge slope of r_c(h) on x, then shift each analog curve to the target's x.
        ys = [a[1].get(h, 0.0) for a in analogs]
        ybar = sum(w * y for w, y in zip(weights, ys)) / wsum
        sxy = sum(w * (a[0] - xbar) * (y - ybar) for w, a, y in zip(weights, analogs, ys))
        sxx = sum(w * (a[0] - xbar) ** 2 for w, a in zip(weights, analogs))
        b = sxy / (sxx + p["ridge"] * wsum / max(len(analogs), 1))
        cap = max(math.log(p["max_viewers"] / v1), math.log(3.0))
        centers = [min(y + b * (x_t - a[0]), cap) for a, y in zip(analogs, ys)]
        sigma = p["s0"] + p["s1"] * math.sqrt(h - 1)
        qs = []
        for tau in LEVELS:
            x = mixture_quantile(centers, weights, sigma, tau)
            q = v1 * math.exp(x)
            # Round first, then clamp: a tiny q must never round to 0 (q > 0 is required).
            q = min(max(round(q, 2), 0.01), 1e9)
            if qs and q < qs[-1]:
                q = qs[-1]
            qs.append(q)
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
