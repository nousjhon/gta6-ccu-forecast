#!/usr/bin/env python3
"""backtest.py — leave-one-out cross-validation over the comparables.

For each comparable with hour 1 and at least 20 of hours 2..24:
  - predict it from the OTHER comparables using main.py's logic
  - compute mean pinball loss on the log scale (quantiles 0.05, 0.25, 0.5, 0.75, 0.95)

Compare against the baseline p1 * exp(z * 0.15 * sqrt(h - 1)).
Print both numbers, then tune main.py to beat the baseline.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/home/ubuntu/bounty/gta6-twitch-ccu-forecast-01-kit/input"))
MODULE_DIR = Path("/home/ubuntu/bounty/gta6-bot")
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15
SMOOTH_WINDOW = 3


def datetime_from_iso(s):
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


def get_hour_map(comp):
    hourly = comp.get("hourly", [])
    day_start = datetime_from_iso(comp["day_start"])
    hour_map = {}
    for entry in hourly:
        dt = datetime_from_iso(entry["from"])
        delta = dt - day_start
        hour_idx = int(delta.total_seconds() // 3600) + 1
        hour_map[hour_idx] = entry["viewers"]
    return hour_map


def get_log_ratio(comp, h):
    hour_map = get_hour_map(comp)
    viewers_1 = hour_map.get(1)
    if viewers_1 is None or viewers_1 <= 0:
        return None
    viewers_h = hour_map.get(h)
    if viewers_h is None or viewers_h <= 0:
        return None
    return math.log(viewers_h / viewers_1)


def compute_raw_curves(comparables, hours):
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


def smooth_curves(curves, window=SMOOTH_WINDOW):
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


def quantile_forecast(comparables, p1, target_hours):
    """Using main.py's logic (but with a clean subset), return quantile forecasts
    for each target hour. Returns a list of lists: one per hour, each a list of 5 quantiles."""
    forecast_hours = target_hours
    raw_curves = compute_raw_curves(comparables, forecast_hours)
    smoothed = smooth_curves(raw_curves, SMOOTH_WINDOW)
    spread = [[q for q in row] for row in smoothed]
    quantiles = [[p1 * math.exp(q) for q in row] for row in spread]
    # Enforce non-decreasing within each row
    for row in quantiles:
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
    # Round to 2 decimals
    rounded = [[round2(v) for v in row] for row in quantiles]
    return rounded


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def pinball_loss(y_log, q_logs):
    """Compute mean pinball loss over quantiles for one prediction point.
    y_log: actual log value
    q_logs: list of 5 predicted log quantiles
    """
    losses = []
    for tau, q_log in zip(LEVELS, q_logs):
        diff = y_log - q_log
        if diff >= 0:
            losses.append(tau * diff)
        else:
            losses.append((tau - 1) * diff)
    return sum(losses) / len(losses)


def main():
    cases = []
    for case_file in sorted(INPUT_DIR.glob("*.in")):
        with open(case_file) as f:
            case = json.loads(f.readline().strip())
        cases.append((case_file.name, case))

    print(f"Found {len(cases)} cases")

    total_pinball = 0.0
    count = 0
    baseline_total = 0.0
    baseline_count = 0

    for case_idx, (case_name, case) in enumerate(cases):
        # Get the excluded comparables
        comparables = case.get("comparables", [])
        forecast_hours = [row["hour"] for row in case["forecast"]]

        # Identify which comparables are eligible
        eligible = []
        for comp in comparables:
            name = comp.get("name", "")
            if "Grand Theft Auto VI category before release" in name:
                continue
            hourly = comp.get("hourly", [])
            day_start = datetime_from_iso(comp["day_start"])
            hour_map = get_hour_map(comp)
            if 1 not in hour_map:
                continue
            present = sum(1 for h in range(2, 25) if h in hour_map)
            if present < 20:
                continue
            eligible.append(comp)

        if len(eligible) < 2:
            print(f"  {case_name}: only {len(eligible)} eligible comparables, skipping")
            continue

        # Collect actual log values for each eligible comparable
        actual_log_values = []
        for comp in eligible:
            hourly = comp.get("hourly", [])
            hour_map = get_hour_map(comp)
            actual_log = {}
            for h in forecast_hours:
                if h in hour_map and hour_map[h] > 0:
                    actual_log[h] = math.log(hour_map[h])
            actual_log_values.append((comp, actual_log))

        # Leave-one-out
        for comp_idx, comp in enumerate(eligible):
            # Remove this comparable
            others = [c for c in eligible if c is not comp]
            if len(others) < 1:
                continue

            # Use the excluded comparable's hour 1 value as p1
            p1 = max(1, get_hour_map(comp)[1])

            # Generate predictions using main.py's logic
            predicted = quantile_forecast(others, p1, forecast_hours)

            # Get actual log values for the excluded comparable
            actual = None
            for a, al in actual_log_values:
                if a is comp:
                    actual = al
                    break
            if not actual:
                continue

            # Compute pinball loss on the log scale
            loss_sum = 0.0
            loss_count = 0
            for h in forecast_hours:
                if h not in actual:
                    continue
                y_log = actual[h]
                q_logs = [math.log(q) for q in predicted[forecast_hours.index(h)]]
                loss_sum += pinball_loss(y_log, q_logs)
                loss_count += 1

            if loss_count > 0:
                mean_loss = loss_sum / loss_count
                total_pinball += mean_loss
                count += 1

            # Baseline prediction for the excluded comparable
            baseline_sum = 0.0
            baseline_count_h = 0
            for h in forecast_hours:
                if h not in actual:
                    continue
                y_log = actual[h]
                baseline_q = [p1 * math.exp(z * SIGMA * math.sqrt(h - 1)) for z in Z]
                baseline_q_logs = [math.log(q) for q in baseline_q]
                baseline_sum += pinball_loss(y_log, baseline_q_logs)
                baseline_count_h += 1

            if baseline_count_h > 0:
                baseline_mean = baseline_sum / baseline_count_h
                baseline_total += baseline_mean
                baseline_count += 1

    if count > 0:
        avg_pinball = total_pinball / count
        avg_baseline = baseline_total / baseline_count if baseline_count > 0 else float('inf')
        print(f"\n=== Results (across all cases) ===")
        print(f"Comparables used: {count}")
        print(f"Main.py logic: mean pinball loss on log scale = {avg_pinball:.6f}")
        print(f"Baseline:        mean pinball loss on log scale = {avg_baseline:.6f}")
        if avg_pinball < avg_baseline:
            print("RESULT: main.py logic BEATS the baseline!")
        else:
            print("RESULT: main.py logic does NOT beat the baseline")
    else:
        print("No comparables were eligible.")


if __name__ == "__main__":
    main()
