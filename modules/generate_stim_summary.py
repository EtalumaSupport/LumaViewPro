# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Standalone stimulation profiling summary generator.

Parses per-color stimulation profile files written by StimulationController
and generates an aggregate summary report with per-color statistics and
outlier analysis.

Usage:
    python -m modules.generate_stim_summary <folder_path>

The folder should contain a stimulation_profile/ subdirectory with
profile text files from one or more protocol runs.
"""

import math
import os
import re
import statistics
import sys
from pathlib import Path


def parse_stimulation_profile(filepath):
    """Parse a single stimulation profile text file.

    Returns a dict with:
        color, frequency, pulse_width, illumination, pulses_executed,
        end_reason, actual_on, led_on_cmd, led_off_cmd
    """
    result = {
        'color': None,
        'frequency': None,
        'pulse_width': None,
        'illumination': None,
        'pulses_executed': None,
        'end_reason': None,
        'actual_on': [],
        'led_on_cmd': [],
        'led_off_cmd': [],
    }

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as ex:
        print(f"  Warning: could not read {filepath}: {ex}")
        return result

    in_event_log = False

    for line in lines:
        line = line.strip()

        # Header metadata
        m = re.match(r'Stimulation Profile:\s*(.+)', line)
        if m:
            result['color'] = m.group(1)
            continue
        m = re.match(r'Frequency:\s*([\d.]+)', line)
        if m:
            result['frequency'] = float(m.group(1))
            continue
        m = re.match(r'Pulse Width:\s*([\d.]+)', line)
        if m:
            result['pulse_width'] = float(m.group(1))
            continue
        m = re.match(r'Illumination:\s*([\d.]+)', line)
        if m:
            result['illumination'] = float(m.group(1))
            continue
        m = re.match(r'Pulses executed:\s*(\d+)', line)
        if m:
            result['pulses_executed'] = int(m.group(1))
            continue
        m = re.match(r'End reason:\s*(.+)', line)
        if m:
            result['end_reason'] = m.group(1)
            continue

        # Per-pulse event log
        if 'Per-Pulse Event Log' in line:
            in_event_log = True
            continue
        if in_event_log and line and not line.startswith('Pulse'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    on_cmd = float(parts[1]) if parts[1] != '—' else None
                    off_cmd = float(parts[2]) if parts[2] != '—' else None
                    actual_on = float(parts[3]) if parts[3] != '—' else None
                    if on_cmd is not None:
                        result['led_on_cmd'].append(on_cmd)
                    if off_cmd is not None:
                        result['led_off_cmd'].append(off_cmd)
                    if actual_on is not None:
                        result['actual_on'].append(actual_on)
                except ValueError:
                    pass

    return result


def write_timing_stats(f, label, values):
    """Write a timing statistics block."""
    if not values:
        f.write(f"  {label}: no data\n")
        return
    f.write(f"  {label}:\n")
    f.write(f"    count:  {len(values)}\n")
    f.write(f"    mean:   {statistics.mean(values):.4f} ms\n")
    if len(values) > 1:
        f.write(f"    std:    {statistics.stdev(values):.4f} ms\n")
    else:
        f.write(f"    std:    0.0000 ms\n")
    f.write(f"    min:    {min(values):.4f} ms\n")
    f.write(f"    max:    {max(values):.4f} ms\n")
    sv = sorted(values)
    f.write(f"    p95:    {sv[int(len(sv) * 0.95)] if len(sv) >= 20 else max(sv):.4f} ms\n")
    f.write(f"    p99:    {sv[int(len(sv) * 0.99)] if len(sv) >= 100 else max(sv):.4f} ms\n")


def write_outlier_details(f, values, label, expected_ms=None):
    """Write 3-sigma outlier analysis."""
    if len(values) < 2:
        return
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    threshold = mean + 3 * std
    outliers = [(i, v) for i, v in enumerate(values) if v > threshold]
    if outliers:
        f.write(f"  {label} 3-sigma outliers (>{threshold:.4f} ms):\n")
        for idx, val in outliers:
            f.write(f"    pulse {idx}: {val:.4f} ms\n")
    if expected_ms is not None and expected_ms > 0:
        deviations = [(i, v) for i, v in enumerate(values)
                      if abs(v - expected_ms) > 3.0]
        if deviations:
            f.write(f"  {label} >3ms deviation from expected {expected_ms:.1f} ms:\n")
            for idx, val in deviations:
                f.write(f"    pulse {idx}: {val:.4f} ms (delta={val - expected_ms:+.4f})\n")


def generate_stimulation_summary(folder_path):
    """Generate an aggregate summary from all profile files in a folder.

    Looks for stimulation_profile/*.txt files, parses them, and writes
    a stimulation_summary.txt with per-color aggregate statistics.
    """
    folder = Path(folder_path)
    profile_dir = folder / "stimulation_profile"

    if not profile_dir.exists():
        print(f"No stimulation_profile/ directory in {folder}")
        return

    profile_files = sorted(profile_dir.glob("stimulation_profile_*.txt"))
    if not profile_files:
        print(f"No profile files found in {profile_dir}")
        return

    print(f"Found {len(profile_files)} profile file(s)")

    # Parse all files, group by color
    by_color = {}
    for pf in profile_files:
        data = parse_stimulation_profile(pf)
        color = data['color'] or 'Unknown'
        if color not in by_color:
            by_color[color] = {
                'frequency': data['frequency'],
                'pulse_width': data['pulse_width'],
                'illumination': data['illumination'],
                'actual_on': [],
                'led_on_cmd': [],
                'led_off_cmd': [],
                'total_pulses': 0,
                'files': [],
            }
        by_color[color]['actual_on'].extend(data['actual_on'])
        by_color[color]['led_on_cmd'].extend(data['led_on_cmd'])
        by_color[color]['led_off_cmd'].extend(data['led_off_cmd'])
        by_color[color]['total_pulses'] += data.get('pulses_executed', 0) or 0
        by_color[color]['files'].append(pf.name)

    # Write summary
    summary_path = profile_dir / "stimulation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Stimulation Summary Report\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Profile files: {len(profile_files)}\n")
        f.write(f"Colors: {', '.join(sorted(by_color.keys()))}\n\n")

        for color in sorted(by_color.keys()):
            data = by_color[color]
            f.write(f"\n--- {color} ---\n")
            f.write(f"  Frequency:    {data['frequency']} Hz\n")
            f.write(f"  Pulse Width:  {data['pulse_width']} ms\n")
            f.write(f"  Illumination: {data['illumination']} mA\n")
            f.write(f"  Total pulses: {data['total_pulses']}\n")
            f.write(f"  Source files: {len(data['files'])}\n\n")

            write_timing_stats(f, "LED ON command time", data['led_on_cmd'])
            write_timing_stats(f, "LED OFF command time", data['led_off_cmd'])
            write_timing_stats(f, "Actual LED on-time", data['actual_on'])

            f.write(f"\n  --- Outlier Analysis ---\n")
            write_outlier_details(f, data['actual_on'], "Actual on-time",
                                  expected_ms=data['pulse_width'])
            write_outlier_details(f, data['led_on_cmd'], "ON command")
            write_outlier_details(f, data['led_off_cmd'], "OFF command")

    print(f"Summary written to {summary_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m modules.generate_stim_summary <folder_path>")
        sys.exit(1)
    generate_stimulation_summary(sys.argv[1])


if __name__ == '__main__':
    main()
