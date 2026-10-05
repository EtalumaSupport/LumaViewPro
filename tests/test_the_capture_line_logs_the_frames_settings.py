"""The protocol capture line logs the exposure and gain the saved file records.

The line read only the camera's chunk values, which only Pylon stamps, so on
the FX2, IDS and the simulator it said ``exp_ms=na gain_db=na`` while the
saved file carried the applied values from the frame record. It now reads the
frame record, and marks a value that is not the frame's own chunk
``(applied)``.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from modules.lumascope_api.frame_record import FrameRecord
from modules.protocol_image_writer import ProtocolImageWriter
from tests.scope_fakes import build_scope


def _writer_over(scope_like):
    writer = ProtocolImageWriter.__new__(ProtocolImageWriter)
    writer._scope = scope_like
    return writer


@pytest.fixture
def live_scope():
    scope = build_scope(simulate=True)
    scope._led_driver.set_timing_mode('fast')
    scope._motion_driver.set_timing_mode('fast')
    scope._camera_driver.set_timing_mode('fast')
    scope.imaging.start_streaming()
    yield scope
    scope.imaging.stop_streaming()
    scope.disconnect()


def test_a_camera_without_chunks_logs_the_applied_values_marked_applied(live_scope):
    applied_exp = live_scope.imaging.set_exposure_ms(50.0)
    applied_gain = live_scope.imaging.set_gain_db(6.0)
    image = live_scope.imaging._capture_and_wait_impl(accept_dark=True, timeout_s=1.0)
    assert image is not None
    info = live_scope.imaging.last_capture_info
    assert info['chunk_exposure_us'] is None, 'the simulator stamps no chunks'

    evidence = _writer_over(live_scope)._capture_evidence(image, 255)

    assert f'exp_ms={applied_exp:.2f}(applied)' in evidence
    assert f'gain_db={applied_gain:.2f}(applied)' in evidence
    assert '=na' not in evidence


def test_a_camera_with_chunks_logs_the_frames_own_values_unmarked():
    record = FrameRecord(
        captured_at=datetime.datetime.now(),
        exposure_ms=62.003,
        gain_db=3.5,
        black_level=None,
        illumination_ma={},
        frames_summed=1,
        frame_significant_bits=12,
        camera_timestamp_ticks=None,
        camera_tick_hz=None,
        frame_id=None,
        binning_size=1,
        camera_model='a camera that stamps chunks',
    )
    info = {'chunk_exposure_us': 62003.0, 'chunk_gain_db': 3.5, 'frame_record': record}
    writer = _writer_over(SimpleNamespace(imaging=SimpleNamespace(last_capture_info=info)))

    evidence = writer._capture_evidence(np.zeros((8, 8), dtype=np.uint8), 255)

    assert 'exp_ms=62.00 ' in evidence
    assert 'gain_db=3.50' in evidence
    assert '(applied)' not in evidence
