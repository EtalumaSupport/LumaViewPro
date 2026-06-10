"""Regression for #568: short protocol Interval/Duration must not collapse.

A Duration of 0.001 h reloaded + displayed as 0.0 (the load path rounded to
2 decimal hours) and sub-second values were otherwise accepted. The minimal
fix: a 1-second floor on Interval + Duration (preserving the 0 single-scan
marker) and 6-decimal display precision so short values survive the reload.
The full H:M:S entry is a tracked follow-up.
"""

import csv
import datetime
import pathlib

import pandas as pd

import modules.config_helpers as config_helpers
from modules.protocol import Protocol

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO_ROOT / 'data' / 'tiling.json'


def _read_protocol_row(tsv_path, key):
    with open(tsv_path, newline='') as fp:
        for row in csv.reader(fp, delimiter='\t'):
            if row and row[0] == key:
                return row[1]
    return None


def test_floor_preserves_zero_single_scan_marker():
    assert config_helpers.floor_protocol_time(datetime.timedelta(0)) == datetime.timedelta(0)


def test_floor_bumps_sub_second_to_one_second():
    td = datetime.timedelta(hours=0.0001)  # 0.36 s
    assert config_helpers.floor_protocol_time(td) == datetime.timedelta(seconds=1)


def test_floor_leaves_values_at_or_above_one_second():
    td = datetime.timedelta(seconds=5)
    assert config_helpers.floor_protocol_time(td) == td
    one = datetime.timedelta(seconds=1)
    assert config_helpers.floor_protocol_time(one) == one


def test_settings_time_params_floor_short_duration():
    params = config_helpers.get_protocol_time_params_from_settings(
        {'protocol': {'period': 20, 'duration': 0.0001}}  # duration 0.36 s
    )
    assert params['duration'] == datetime.timedelta(seconds=1)
    assert params['period'] == datetime.timedelta(minutes=20)


def test_settings_time_params_preserve_single_scan_zero():
    params = config_helpers.get_protocol_time_params_from_settings(
        {'protocol': {'period': 0, 'duration': 0}}
    )
    assert params['period'] == datetime.timedelta(0)
    assert params['duration'] == datetime.timedelta(0)


def test_load_display_uses_six_decimal_precision():
    """The reload display must not round duration/period to 2 decimal hours
    (which collapsed a 1s duration to 0.0)."""
    src = (REPO_ROOT / 'ui' / 'protocol_settings.py').read_text(encoding='utf-8')
    assert 'total_seconds() / 3600, 6)' in src, (
        'load display must round duration to 6 decimals so short durations '
        'do not show 0.0 (#568)'
    )
    assert 'total_seconds() / 60, 6)' in src, (
        'load display must round period to 6 decimals (#568)'
    )


def test_save_preserves_one_second_interval(tmp_path):
    """The SAVE path must keep sub-minute Interval precision. A 1-second
    Period written at 2-decimal minutes collapses to 0.02 min on disk -- the
    stored value itself is corrupted (the display fix alone did not cover the
    writer in protocol.to_file)."""
    protocol = Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(),
            'period': datetime.timedelta(seconds=1),
            'duration': datetime.timedelta(hours=1),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )
    out = tmp_path / 'p.tsv'
    assert protocol.to_file(out) is None

    saved_minutes = float(_read_protocol_row(out, 'Period'))
    # 1 second = 0.016667 min; the pre-fix 2-decimal round wrote 0.02.
    assert saved_minutes != 0.02, 'Period saved at 2-decimal minutes (data loss)'
    assert abs(saved_minutes * 60.0 - 1.0) < 0.1, (
        f'1s Interval must survive save; got {saved_minutes} min '
        f'({saved_minutes * 60.0:.4f} s)'
    )
