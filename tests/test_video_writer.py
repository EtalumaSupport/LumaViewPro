"""Tests for VideoWriter cv2-fallback channel handling.

The cv2 fallback path is the only consumer in the save path that
expects BGR; PyAV and tifffile both want RGB. add_frame converts
RGB->BGR at the cv2 boundary so callers can hand it RGB uniformly.
"""

from unittest import mock

import numpy as np
import pytest

from modules.video_writer import VideoWriter


class _FakeCv2VideoWriter:
    """Records frames passed to write() instead of writing to disk."""

    def __init__(self, *args, **kwargs):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())
        return True

    def release(self):
        pass


@pytest.fixture
def cv2_writer(tmp_path):
    """VideoWriter forced onto the cv2 fallback path, capturing frames in memory."""
    output_path = tmp_path / "test.avi"
    fake = _FakeCv2VideoWriter()
    with mock.patch('modules.video_writer.cv2.VideoWriter', return_value=fake):
        writer = VideoWriter(output_file_loc=output_path, fps=30,
                             include_timestamp_overlay=False)
        writer._use_pyav = False
        yield writer, fake


class TestVideoWriterCv2Fallback:
    """cv2.VideoWriter consumes BGR; callers pass RGB. Conversion happens in add_frame."""

    def test_rgb_red_becomes_bgr(self, cv2_writer):
        writer, fake = cv2_writer
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        writer.add_frame(image=rgb, timestamp=None)
        writer.finish()
        bgr = fake.frames[0]
        assert bgr[:, :, 2].sum() > 0, "Red lands at BGR index 2"
        assert bgr[:, :, 0].sum() == 0
        assert bgr[:, :, 1].sum() == 0

    def test_rgb_blue_becomes_bgr(self, cv2_writer):
        writer, fake = cv2_writer
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 2] = 200
        writer.add_frame(image=rgb, timestamp=None)
        writer.finish()
        bgr = fake.frames[0]
        assert bgr[:, :, 0].sum() > 0, "Blue lands at BGR index 0"
        assert bgr[:, :, 1].sum() == 0
        assert bgr[:, :, 2].sum() == 0

    def test_grayscale_frame_unchanged(self, cv2_writer):
        writer, fake = cv2_writer
        gray = np.full((100, 100), 128, dtype=np.uint8)
        writer.add_frame(image=gray, timestamp=None)
        writer.finish()
        assert fake.frames[0].shape == (100, 100)
        assert fake.frames[0].sum() > 0
