# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame's time is the one its file states, else its run record's.

``image_utils.frame_time`` is the one answer: Create Video's overlay and the
count both read it, so a projection (which states no time of its own) is
given its scan's time from the record by each, never a time by one and none
by the other.
"""

from __future__ import annotations

import datetime

import pandas as pd

from _lumascope.modules.image_utils import frame_time

_PDT = datetime.timezone(datetime.timedelta(hours=-7))
_STATED = datetime.datetime(2026, 11, 8, 9, 30, tzinfo=_PDT)
_RECORDED = datetime.datetime(2026, 11, 8, 9, 0, tzinfo=_PDT)


def test_a_time_the_file_states_is_the_frames_over_its_records():
    assert frame_time(_STATED, _RECORDED) == _STATED


def test_a_file_stating_no_time_takes_its_record_rows_time_in_either_cell_form():
    """The loader keeps a run's times as pandas Timestamps when one offset
    runs through it and as datetimes when the offsets differ; each is the
    row's time."""
    assert frame_time(None, _RECORDED) == _RECORDED
    assert frame_time(None, pd.Timestamp(_RECORDED)) == _RECORDED


def test_a_frame_whose_file_and_record_state_no_time_has_none():
    """The loader fills a missing time with the empty string; it is no time,
    and no other time is given in its place."""
    assert frame_time(None, '') is None
    assert frame_time(None, None) is None
