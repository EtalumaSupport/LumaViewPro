# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The frame-validity trace records the comparison the gate actually made.

The gate credits a frame to a pending write only when the frame's arrival
ordinal is greater than the ordinal the write was stamped against, counting
down a per-source budget. The trace has to show both sides of that comparison,
because the question asked of the file at the bench is "did any source clear
below its target", and the file is read by pairing each settled row with its
own invalidate row BY SOURCE.

Two shapes are pinned here because losing either makes the file misleading
rather than merely incomplete:

  - A frame that credits nothing must still leave a row. Without one the file
    is silent in exactly the case worth investigating, and a reader who finds
    no rows concludes no frames arrived.
  - A settled row names ONE source. A row naming two cannot be paired against
    the invalidate rows that produced it.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from lib import profile_trace
from modules.frame_validity import FrameValidity


@pytest.fixture(autouse=True)
def _reset_profile_trace():
    profile_trace.disable()
    yield
    profile_trace.disable()


def _run_dir(base):
    return next(d for d in sorted(base.iterdir()) if d.is_dir())


def _rows(base):
    """Trace rows as dicts, keyed by the header the writer laid down.

    An empty list when the file is absent: the writer opens a file on its
    first row, so "no rows" and "no file" are the same outcome.
    """
    path = _run_dir(base) / 'frame_validity_trace.csv'
    if not path.exists():
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


class _Camera:
    """Stands in for the driver's delivered-frame count."""

    def __init__(self):
        self.delivered = 0

    def __call__(self):
        return self.delivered


class TestTheRetiredQuantityIsGone:
    def test_header_has_no_target_frame(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        fv = FrameValidity(_Camera())
        fv.invalidate('gain')
        header = _rows(tmp_path)[0].keys()
        assert 'target_frame' not in header
        assert {'frame_seq', 'at_seq', 'remaining'} <= set(header)

    def test_invalidate_records_the_ordinal_the_write_was_stamped_against(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        cam = _Camera()
        cam.delivered = 7
        fv = FrameValidity(cam)
        fv.invalidate('gain')
        row = _rows(tmp_path)[0]
        assert row['event'] == 'invalidate'
        assert row['source'] == 'gain'
        assert row['at_seq'] == '7'
        assert row['remaining'] == str(FrameValidity.SKIP_FRAMES['gain'])


class TestAFrameThatCreditsNothingLeavesARow:
    def test_a_frame_older_than_the_write_records_nocredit(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        cam = _Camera()
        cam.delivered = 10
        fv = FrameValidity(cam)
        fv.invalidate('gain')
        fv.count_frame(4)  # in flight before the write; must not credit
        rows = [r for r in _rows(tmp_path) if r['event'] == 'nocredit']
        assert len(rows) == 1
        assert rows[0]['source'] == 'gain'
        assert rows[0]['frame_seq'] == '4'
        assert rows[0]['at_seq'] == '10'
        assert rows[0]['remaining'] == str(FrameValidity.SKIP_FRAMES['gain'])

    def test_a_frame_after_the_write_records_credit_and_the_new_count(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        cam = _Camera()
        cam.delivered = 10
        fv = FrameValidity(cam)
        fv.invalidate('gain')
        fv.count_frame(11)
        row = next(r for r in _rows(tmp_path) if r['event'] == 'credit')
        assert row['frame_seq'] == '11'
        assert row['remaining'] == str(FrameValidity.SKIP_FRAMES['gain'] - 1)


class TestASettledRowNamesOneSource:
    def test_two_sources_settling_together_emit_two_rows(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        cam = _Camera()
        fv = FrameValidity(cam)
        fv.invalidate('gain')
        fv.invalidate('exposure')
        seq = 1
        while fv._pending and seq < 50:
            seq += 1
            fv.count_frame(seq)
        settled = [r for r in _rows(tmp_path) if r['event'] == 'settled']
        assert {r['source'] for r in settled} == {'gain', 'exposure'}
        assert all('+' not in r['source'] for r in settled), 'a joined row cannot be paired'

    def test_every_settled_row_pairs_with_an_invalidate_row(self, tmp_path):
        """The bench pass criterion, run as a test."""
        profile_trace.enable(output_dir=tmp_path)
        fv = FrameValidity(_Camera())
        fv.invalidate('led')
        seq = 0
        while fv._pending and seq < 50:
            seq += 1
            fv.count_frame(seq)
        rows = _rows(tmp_path)
        invalidated = [r['source'] for r in rows if r['event'] == 'invalidate']
        for row in (r for r in rows if r['event'] == 'settled'):
            assert row['source'] in invalidated
            assert row['remaining'] == '0', 'a source cleared below its target'


class TestTheDedupeRowIsBoundedButPresent:
    def test_a_repeat_ordinal_is_recorded_while_a_source_is_pending(self, tmp_path):
        profile_trace.enable(output_dir=tmp_path)
        fv = FrameValidity(_Camera())
        fv.invalidate('gain')
        fv.count_frame(5)
        fv.count_frame(5)  # the same frame polled twice
        assert [r for r in _rows(tmp_path) if r['event'] == 'dedupe']

    def test_an_idle_poller_writes_nothing(self, tmp_path):
        """A camera whose ordinal never advances must not fill the file."""
        profile_trace.enable(output_dir=tmp_path)
        fv = FrameValidity(_Camera())
        fv.count_frame(1)
        for _ in range(200):
            fv.count_frame(1)
        rows = [r for r in _rows(tmp_path) if r['event'] == 'dedupe']
        assert rows == []


class TestEveryRowMatchesTheHeader:
    def test_no_row_is_misaligned(self, tmp_path):
        """A short row still parses, so the damage is wrong numbers, not an error."""
        profile_trace.enable(output_dir=tmp_path)
        cam = _Camera()
        fv = FrameValidity(cam)
        fv.invalidate('gain')
        fv.invalidate('z_move')
        fv.count_frame(1)
        fv.count_frame(1)
        cam.delivered = 5
        for seq in range(2, 20):
            fv.count_frame(seq)
        with open(_run_dir(tmp_path) / 'frame_validity_trace.csv', newline='') as fh:
            rows = list(csv.reader(fh))
        width = len(rows[0])
        assert width == len(FrameValidity._TRACE_HEADER.split(',')) + 1
        assert all(len(r) == width for r in rows[1:])
