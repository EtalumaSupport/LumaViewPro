"""Regression: the capture-evidence saturation metric must scale to the
frame's true bit depth, not the container dtype.

12-bit frames ride in uint16 containers. Deriving full scale from
np.iinfo(image.dtype).max (65535) put the saturation threshold at ~64879,
so a fully blown 12-bit frame (every pixel 4095) logged sat=0.0% -- the
exact frames the metric exists to flag read as perfectly exposed. Full
scale must come from the frame's significant bits: (1 << bits) - 1.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from modules.protocol_image_writer import ProtocolImageWriter


def _make_writer(last_capture_info=None):
    writer = ProtocolImageWriter.__new__(ProtocolImageWriter)
    writer._scope = SimpleNamespace(imaging=SimpleNamespace(last_capture_info=last_capture_info))
    return writer


def test_saturated_12bit_frame_in_uint16_container_reports_full_saturation():
    writer = _make_writer()
    frame = np.full((64, 64), 4095, dtype=np.uint16)
    evidence = writer._capture_evidence(frame, 12)
    assert 'sat=100.0%' in evidence
    assert 'mean=4095.0' in evidence


def test_midrange_12bit_frame_reports_zero_saturation():
    writer = _make_writer()
    frame = np.full((64, 64), 2000, dtype=np.uint16)
    evidence = writer._capture_evidence(frame, 12)
    assert 'sat=0.0%' in evidence


def test_saturated_8bit_frame_reports_full_saturation():
    writer = _make_writer()
    frame = np.full((64, 64), 255, dtype=np.uint8)
    evidence = writer._capture_evidence(frame, 8)
    assert 'sat=100.0%' in evidence


def test_saturated_summed_16bit_frame_reports_full_saturation():
    writer = _make_writer()
    frame = np.full((64, 64), 65535, dtype=np.uint16)
    evidence = writer._capture_evidence(frame, 16)
    assert 'sat=100.0%' in evidence
