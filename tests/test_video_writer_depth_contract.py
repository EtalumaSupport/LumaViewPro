# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
The encode edge cannot guess pixel depth: significant_bits is required
with non-uint8 frames. Both historical guesses corrupted silently -- a
full-16-bit scale renders a right-aligned 12-bit frame near-black, and
a raw astype() cast truncates high bits. The contract makes the missing
depth loud at the exact call that dropped it.
"""

import av
import numpy as np
import pytest

from modules.video_writer import VideoWriter


def _writer(tmp_path):
    return VideoWriter(output_path=tmp_path / 'out.mp4', fps=10, width=32, height=24)


def test_non_uint8_frame_without_depth_raises(tmp_path):
    writer = _writer(tmp_path)
    frame = np.full((24, 32), 4095, dtype=np.uint16)
    with pytest.raises(ValueError, match='significant_bits'):
        writer.add_frame(image=frame)


def test_uint16_frame_with_depth_scales_to_full_brightness(tmp_path):
    # A right-aligned 12-bit frame at full scale must encode bright,
    # not at 4095/65535 of range -- the near-black hazard the required
    # depth exists to prevent.
    writer = _writer(tmp_path)
    writer.add_frame(image=np.full((24, 32), 4095, dtype=np.uint16), significant_bits=12)
    writer.close()

    with av.open(str(writer.output_path)) as container:
        decoded = next(iter(container.decode(video=0)))
        pixels = decoded.to_ndarray(format='gray')
    assert pixels.mean() > 200


def test_uint8_frame_without_depth_is_legal(tmp_path):
    # uint8 needs no scaling; None stays valid for it.
    writer = _writer(tmp_path)
    writer.add_frame(image=np.full((24, 32), 200, dtype=np.uint8))
    writer.close()
    assert writer.output_path.exists()
