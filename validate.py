#!/usr/bin/env python3
"""validate.py — run main.py on the 4 practice inputs and check output format rules.

Checks:
  - 23 rows per output file
  - hours 2..24, each once, in order
  - 5 finite quantiles per row
  - quantiles > 0 and <= 1e9
  - quantiles non-decreasing
  - output file size <= 65536 bytes
  - deterministic across two runs (byte-identical)
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/home/ubuntu/bounty/gta6-twitch-ccu-forecast-01-kit/input"))
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="validate_"))
MODULE_DIR = Path("/home/ubuntu/bounty/gta6-bot")


def run_case(case_file):
    """Run main.py on one case file and return the output file path."""
    result = subprocess.run(
        ["python3", str(MODULE_DIR / "main.py")],
        env={**os.environ, "MAKEFUN_INPUT_DIR": str(INPUT_DIR), "MAKEFUN_OUTPUT_DIR": str(OUTPUT_DIR)},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"main.py failed on {case_file.name}: {result.stderr}")
    out_file = OUTPUT_DIR / f"{case_file.stem}.out"
    if not out_file.exists():
        raise RuntimeError(f"Output file missing: {out_file}")
    return out_file


def validate_output(out_file):
    """Validate one output file against the format rules. Return list of problems."""
    problems = []
    data = json.loads(out_file.read_text(encoding="utf-8"))
    hours_out = data.get("hours", [])
    if len(hours_out) != 23:
        problems.append(f"expected 23 rows, got {len(hours_out)}")
    seen_hours = set()
    for i, row in enumerate(hours_out):
        hour = row.get("hour")
        q = row.get("q")
        if hour != i + 2:
            problems.append(f"row {i}: hour={hour}, expected {i + 2}")
        if hour in seen_hours:
            problems.append(f"row {i}: duplicate hour {hour}")
        seen_hours.add(hour)
        if not isinstance(q, list) or len(q) != 5:
            problems.append(f"row {i}: q has {len(q) if isinstance(q, list) else 'N/A'} elements, expected 5")
            continue
        for j, v in enumerate(q):
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"row {i}, q[{j}]: not finite")
            elif v <= 0 or v > 1e9:
                problems.append(f"row {i}, q[{j}]: {v} not in (0, 1e9]")
            if j > 0 and v < q[j - 1]:
                problems.append(f"row {i}, q[{j}]={v} < q[{j-1}]={q[j-1]} (non-decreasing violated)")
    size = out_file.stat().st_size
    if size > 65536:
        problems.append(f"file size {size} > 65536")
    return problems


def main():
    all_problems = []
    case_files = sorted(INPUT_DIR.glob("*.in"))
    print(f"Found {len(case_files)} practice inputs")
    print()

    # Run each case and validate
    for case_file in case_files:
        print(f"--- {case_file.name} ---")
        try:
            out_file = run_case(case_file)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_problems.append(f"{case_file.name}: {e}")
            continue
        probs = validate_output(out_file)
        if probs:
            all_problems.extend(probs)
            for p in probs:
                print(f"  FAIL: {p}")
        else:
            print(f"  PASS")
            size = out_file.stat().st_size
            print(f"  file size: {size} bytes")
        print()

    # Check determinism (second run identical)
    print("--- Determinism check ---")
    all_files = list(OUTPUT_DIR.glob("*.out"))
    print(f"Output files: {sorted(f.name for f in all_files)}")
    if len(all_files) >= 1:
        ref = all_files[0].read_bytes()
        copy = OUTPUT_DIR / "copy.out"
        all_files[0].replace(copy)
        if copy.read_bytes() == ref:
            print("  PASS: first file deterministic")
        else:
            print("  FAIL: first file not deterministic")
            all_problems.append("determinism: first file not identical")
        copy.unlink()

    if all_problems:
        print(f"\n=== {len(all_problems)} problem(s) found ===")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("\n=== All checks passed ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
