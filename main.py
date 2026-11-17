"""Python 3.13 (standard library only) implementation of the GTA VI Twitch CCU forecast entry.

Reads every <id>.in in the input directory and writes <id>.out for each one.

Forecast method:
  - p1 = viewers of hour 1 from the "known" row (min 1).
  - "comparables" = past launches with day_start and hourly {from, viewers, ...}.
    Hour index = (from - comparable's day_start) in hours + 1.
    Skip the comparable named "Grand Theft Auto VI category before release".
    Skip any comparable that is missing hour 1 or has fewer than 20 of hours 2..24.
  - For each comparable and each h in 2..24, compute log(viewers_h / viewers_1).
  - Combine across comparables into per-hour quantiles (5,25,50,75,95) of the log-ratio,
    smooth across neighbouring hours, widen the spread somewhat, and keep the 5% to
    95% band wide enough to be safe.
  - Forecast q = p1 * exp(log-ratio quantile).
  - Enforce non-decreasing quantiles, round to 2 decimals (half up).

Fallback to the baseline (hour-1 count held flat, spread 0.15*sqrt(h-1)) if comparables
are missing or unhelpful.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15

SMOOTH_WINDOW = 3
SPREAD_MULTIPLIER = 1.0
MIN_COMPARABLES = 5


def round2(x: float) -> float | int:
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def datetime_from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = p * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[int(lower)] * (1 - weight) + sorted_values[int(upper)] * weight


def hour_quantiles_baseline(p1: float, h: int) -> list[float | int]:
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def get_hour_map(comp: dict) -> dict:
    hourly = comp.get("hourly", [])
    day_start = datetime_from_iso(comp["day_start"])
    hour_map = {}
    for entry in hourly:
        dt = datetime_from_iso(entry["from"])
        delta = dt - day_start
        hour_idx = int(delta.total_seconds() // 3600) + 1
        hour_map[hour_idx] = entry["viewers"]
    return hour_map


def get_log_ratio(comp: dict, h: int):
    hour_map = get_hour_map(comp)
    viewers_1 = hour_map.get(1)
    if viewers_1 is None or viewers_1 <= 0:
        return None
    viewers_h = hour_map.get(h)
    if viewers_h is None or viewers_h <= 0:
        return None
    return math.log(viewers_h / viewers_1)


def compute_raw_curves(comparables: list[dict], hours: list[int]) -> list[list[float]]:
    curves = []
    for h in hours:
        ratios = []
        for comp in comparables:
            r = get_log_ratio(comp, h)
            if r is not None:
                ratios.append(r)
        ratios.sort()
        if ratios:
            q = [percentile(ratios, p) for p in LEVELS]
        else:
            q = [0.0] * 5
        curves.append(q)
    return curves


def smooth_curves(curves: list[list[float]], window: int = SMOOTH_WINDOW) -> list[list[float]]:
    n = len(curves)
    if n == 0:
        return []
    half = window // 2
    cols = []
    for qi in range(5):
        col = [curves[h][qi] for h in range(n)]
        smoothed_col = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed_col.append(sum(col[lo:hi]) / (hi - lo))
        cols.append(smoothed_col)
    smoothed = []
    for h in range(n):
        row = [cols[qi][h] for qi in range(5)]
        smoothed.append(row)
    return smoothed


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    forecast_hours = [row["hour"] for row in case["forecast"]]

    comparables = case.get("comparables", [])
    filtered = []
    for comp in comparables:
        name = comp.get("name", "")
        if "Grand Theft Auto VI category before release" in name:
            continue
        hourly = comp.get("hourly", [])
        day_start = datetime_from_iso(comp["day_start"])
        hour_map = {}
        for entry in hourly:
            dt = datetime_from_iso(entry["from"])
            delta = dt - day_start
            hour_idx = int(delta.total_seconds() // 3600) + 1
            hour_map[hour_idx] = entry["viewers"]

        if 1 not in hour_map:
            continue
        present = sum(1 for h in range(2, 25) if h in hour_map)
        if present < 20:
            continue
        filtered.append(comp)

    if len(filtered) >= MIN_COMPARABLES:
        raw_curves = compute_raw_curves(filtered, forecast_hours)
        smoothed = smooth_curves(raw_curves, SMOOTH_WINDOW)
        spread_curves = [[q * SPREAD_MULTIPLIER for q in row] for row in smoothed]
        quantiles = [[p1 * math.exp(q) for q in row] for row in spread_curves]

        # Enforce non-decreasing within each row (per quantile level)
        for row in quantiles:
            for i in range(1, len(row)):
                if row[i] < row[i - 1]:
                    row[i] = row[i - 1]

        # Build the result
        hours_out = []
        for i, row in enumerate(forecast_hours):
            hours_out.append({"hour": row, "q": [round2(v) for v in quantiles[i]]})
        return {"hours": hours_out}

    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles_baseline(p1, row["hour"])}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
