#!/usr/bin/env python3
"""Grid search for best parameters to reduce pinball loss."""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path('/home/ubuntu/bounty/gta6-twitch-ccu-forecast-01-kit/input')
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15


def datetime_from_iso(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def get_hour_map(comp):
    hourly = comp.get('hourly', [])
    day_start = datetime_from_iso(comp['day_start'])
    hour_map = {}
    for entry in hourly:
        dt = datetime_from_iso(entry['from'])
        delta = dt - day_start
        hour_idx = int(delta.total_seconds() // 3600) + 1
        hour_map[hour_idx] = entry['viewers']
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


def smooth_curves(curves, window):
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


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def pinball_loss(y_log, q_logs):
    losses = []
    for tau, q_log in zip(LEVELS, q_logs):
        diff = y_log - q_log
        if diff >= 0:
            losses.append(tau * diff)
        else:
            losses.append((tau - 1) * diff)
    return sum(losses) / len(losses)


def load_cases():
    cases = []
    for case_file in sorted(INPUT_DIR.glob('*.in')):
        with open(case_file) as f:
            case = json.loads(f.readline().strip())
        cases.append((case_file.name, case))
    return cases


def get_eligible_comparables(comparables):
    eligible = []
    for comp in comparables:
        name = comp.get('name', '')
        if 'Grand Theft Auto VI category before release' in name:
            continue
        hourly = comp.get('hourly', [])
        day_start = datetime_from_iso(comp['day_start'])
        hour_map = get_hour_map(comp)
        if 1 not in hour_map:
            continue
        present = sum(1 for h in range(2, 25) if h in hour_map)
        if present < 20:
            continue
        eligible.append(comp)
    return eligible


def compute_pinball_loss(eligible, forecast_hours, others, p1, spread_mult, window):
    raw_curves = compute_raw_curves(others, forecast_hours)
    smoothed = smooth_curves(raw_curves, window)
    spread_curves = [[q * spread_mult for q in row] for row in smoothed]
    quantiles = [[p1 * math.exp(q) for q in row] for row in spread_curves]
    for row in quantiles:
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
    rounded = [[round2(v) for v in row] for row in quantiles]

    actual = None
    for a, al in actual_log_values:
        if a is comp:
            actual = al
            break
    if not actual:
        return None, None

    loss_sum = 0.0
    loss_count = 0
    for h in forecast_hours:
        if h not in actual:
            continue
        y_log = actual[h]
        q_logs = [math.log(q) for q in rounded[forecast_hours.index(h)]]
        loss_sum += pinball_loss(y_log, q_logs)
        loss_count += 1

    if loss_count == 0:
        return None, None

    mean_loss = loss_sum / loss_count

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

    if baseline_count_h == 0:
        return None, None

    baseline_mean = baseline_sum / baseline_count_h
    return mean_loss, baseline_mean


if __name__ == '__main__':
    cases = load_cases()

    # Pre-compute eligible comparables and actual log values for each case
    case_data = []
    for case_name, case in cases:
        comparables = case.get('comparables', [])
        forecast_hours = [row['hour'] for row in case['forecast']]
        eligible = get_eligible_comparables(comparables)

        actual_log_values = []
        for comp in eligible:
            hour_map = get_hour_map(comp)
            actual_log = {}
            for h in forecast_hours:
                if h in hour_map and hour_map[h] > 0:
                    actual_log[h] = math.log(hour_map[h])
            actual_log_values.append((comp, actual_log))

        case_data.append((case_name, case, forecast_hours, eligible, actual_log_values))

    results = []
    for window in [3, 5, 7, 11, 15]:
        for spread_mult in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            total_pinball = 0.0
            count = 0
            baseline_total = 0.0
            baseline_count = 0

            for case_name, case, forecast_hours, eligible, actual_log_values in case_data:
                for comp_idx, comp in enumerate(eligible):
                    others = [c for c in eligible if c is not comp]
                    if len(others) < 1:
                        continue
                    p1 = max(1, get_hour_map(comp)[1])
                    mean_loss, baseline_mean = compute_pinball_loss(
                        eligible, forecast_hours, others, p1, spread_mult, window
                    )
                    if mean_loss is None:
                        continue
                    total_pinball += mean_loss
                    count += 1
                    baseline_total += baseline_mean
                    baseline_count += 1

            if count > 0:
                avg_pinball = total_pinball / count
                avg_baseline = baseline_total / baseline_count if baseline_count > 0 else float('inf')
                results.append((avg_pinball, avg_baseline, window, spread_mult))
                print(f'window={window}, spread={spread_mult}: main.py={avg_pinball:.6f}, baseline={avg_baseline:.6f}', flush=True)

    results.sort(key=lambda x: x[0])
    print(f'\nBest 5 combinations:')
    for i, (pinball, baseline, window, spread) in enumerate(results[:5]):
        print(f'  window={window}, spread={spread}: main.py={pinball:.6f}, baseline={baseline:.6f}')
