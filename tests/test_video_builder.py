# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#657 regression: VideoBuilder must feed RGB to VideoWriter.

Setup:
  - TIFFs on disk are saved in RGB ordering (canonical save-path convention
    established by e2ef49e: add_false_color returns RGB; write_tiff stores
    photometric=RGB).
  - cv2.imread, used by VideoBuilder._create_video to read frames back,
    always returns BGR for 3-channel images (OpenCV convention).
  - VideoWriter expects RGB per the same canonical convention (PyAV uses
    'rgb24'; cv2 fallback converts to BGR at its own boundary).

Without a BGR->RGB conversion after cv2.imread, the blue channel of an
RGB-saved frame lands in the red channel of the output mp4. That was #657's
video-side symptom after the frame-side was fixed by e2ef49e.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import tifffile

# Heavy deps are mocked by tests/conftest.py at module-import time.


@pytest.fixture
def blue_rgb_frame(tmp_path):
    """Write a single 16-bit TIFF with a known RGB blue value.

    Saved layout: array of shape (H, W, 3) where index 0=Red, 1=Green, 2=Blue
    (canonical save-path RGB ordering). All pixels have R=0, G=0, B=42000.
    """
    arr = np.zeros((4, 4, 3), dtype=np.uint16)
    arr[:, :, 2] = 42000  # Blue at index 2 per RGB convention
    file_loc = tmp_path / "frame_0001.tif"
    tifffile.imwrite(str(file_loc), arr, photometric='rgb')
    return file_loc


class TestVideoBuilderBgrToRgb_657:
    """#657 regression: video build must apply BGR->RGB after cv2.imread."""

    def test_create_video_feeds_rgb_to_videowriter(self, blue_rgb_frame, tmp_path):
        """End-to-end: a Blue-RGB TIFF on disk reaches VideoWriter.add_frame
        as Blue-RGB (channel index 2 still holds the data), not BGR-flipped."""
        from modules.video_builder import VideoBuilder

        # Capture every image handed to VideoWriter.add_frame.
        captured = []

        class CaptureWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_frame(self, image, timestamp=None):
                captured.append(image.copy())

            def finish(self):
                pass

        df = pd.DataFrame({
            'Filepath': [blue_rgb_frame.name],
            'Timestamp': [pd.Timestamp('2026-05-15 12:00:00')],
            'Scan Count': [0],
        })

        builder = VideoBuilder.__new__(VideoBuilder)
        builder._name = 'test_video_builder_657'

        with patch('modules.video_writer.VideoWriter', CaptureWriter):
            builder._create_video(
                path=tmp_path,
                df=df,
                frames_per_sec=10,
                enable_timestamp_overlay=False,
                output_file_loc=Path('out.mp4'),
            )

        assert len(captured) == 1, "VideoWriter.add_frame should fire once per frame"
        frame = captured[0]
        assert frame.ndim == 3, "Color frame expected"
        # If the BGR->RGB conversion is applied, blue stays at channel 2.
        # If the conversion is missing, cv2.imread's BGR-return puts the blue
        # data at channel 0 (red) and 42000 will appear there instead.
        assert frame[0, 0, 2] == 42000, (
            f"Blue channel should be at RGB index 2; got "
            f"R={frame[0, 0, 0]} G={frame[0, 0, 1]} B={frame[0, 0, 2]}"
        )
        assert frame[0, 0, 0] == 0, "Red channel should be zero in a pure-Blue source"
