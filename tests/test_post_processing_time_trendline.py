# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for the cell-count time-axis trendline crash.

PostProcessing built the graph 'time' column as a list of datetime.strptime
objects. update_trendline then did (x - x.min()).dt.total_seconds() on a time
axis, and the .dt accessor raised "Can only use .dt accessor with datetimelike
values", crashing the whole app -- the column was not a pandas datetime64.
set_graphing_source now parses with pd.to_datetime, which is guaranteed
datetime64 and supports the .dt operation the trendline needs.

PostProcessing imports Kivy (not importable in the test env), so this verifies
the fix contract directly plus a source-text lock on the production parse.
"""

from __future__ import annotations

import datetime
import pathlib

import pandas as pd

_SRC = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text()


def test_pd_to_datetime_yields_datetime64_with_working_dt():
    # Parse the same '%c' strings the CSV uses; the result must be datetime64
    # and support (x - x.min()).dt.total_seconds() -- the trendline operation
    # that crashed on the old object-dtype column.
    dts = [datetime.datetime(2026, 5, 31, 12, 0, 0), datetime.datetime(2026, 5, 31, 12, 1, 0)]
    strs = pd.Series([d.strftime('%c') for d in dts])
    parsed = pd.to_datetime(strs, format='%c')
    assert parsed.dtype.kind == 'M'  # datetime64
    secs = (parsed - parsed.min()).dt.total_seconds().tolist()
    assert secs == [0.0, 60.0]


def test_production_parses_time_with_pd_to_datetime():
    # Lock the fix: the time column is parsed to datetime64, not rebuilt as the
    # strptime list comprehension (non-datetime64) that crashed the trendline.
    assert "pd.to_datetime(self.graph_df['time']" in _SRC
    assert 'strptime(datetime_obj' not in _SRC
