# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Frame-ordering regression tests for VideoBuilder's video_Frame sort.

The producer names frames ``{name}_Frame_{frame_num:04}`` -- four-digit
zero padding that grows to five digits at frame 10,000 -- while the
builder's sort key is the LAST FOUR characters of the filename stem. At
frame 10,000 the key wraps ("10000" reads as "0000") and collides with
frame 0, interleaving the second ten thousand frames into the first and
scrambling the output video. Below 10,000 frames the key is correct.

The sub-10k test pins today's correct behavior through the real
_create_video sort. The rollover test is the regression test for the
ordering fix and runs xfail(strict) until that fix lands; when it
starts passing, the marker must come off in the same change.
"""

import pathlib
import random
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.video_builder import VideoBuilder


def _frame_name(n):
    # Mirrors the producer's f'{name}_Frame_{frame_num:04}' plus the
    # TIFF suffix the writer appends.
    return f'Well1_video_Frame_{n:04}.tiff'


def _consumed_order(tmp_path, frame_numbers):
    """Run the real _create_video sort; return frame numbers in the
    order the builder consumed them (writer and file reads stubbed)."""
    shuffled = list(frame_numbers)
    random.Random(42).shuffle(shuffled)
    df = pd.DataFrame({'Filepath': [_frame_name(n) for n in shuffled]})

    consumed = []

    def _record_frame(self, writer, image_path, **kwargs):
        consumed.append(int(image_path.name.rsplit('_', 1)[1].split('.')[0]))
        return True

    writer = MagicMock()
    writer.dropped_frames = 0
    with (
        patch.object(VideoBuilder, '_add_source_frame', _record_frame),
        patch('modules.video_builder.VideoWriter', return_value=writer),
    ):
        builder = VideoBuilder(has_turret=False)
        builder._create_video(
            path=tmp_path,
            df=df,
            frames_per_sec=10,
            enable_timestamp_overlay=False,
            output_file_loc=pathlib.Path('out.mp4'),
        )
    return consumed


def test_sub_10k_frame_ordering_is_numeric(tmp_path):
    frames = list(range(0, 200)) + list(range(9900, 10000))
    consumed = _consumed_order(tmp_path, frames)
    assert consumed == sorted(frames)


@pytest.mark.xfail(
    reason='frame-number sort key wraps at 10,000 (last-4-chars key); '
    'fix is hard-ordered before the protocol video cutover',
    strict=True,
)
def test_frame_ordering_survives_10k_rollover(tmp_path):
    frames = list(range(9990, 10020))
    consumed = _consumed_order(tmp_path, frames)
    assert consumed == sorted(frames)
