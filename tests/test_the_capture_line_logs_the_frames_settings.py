"""The protocol capture line logs the exposure and gain the saved file records.

The line read only the camera's chunk values, which only a Basler body (and
the simulated camera, which stamps as one) carries, so on the FX2 and IDS it
said ``exp_ms=na gain_db=na`` while the
saved file carried the applied values from the frame record. It now reads the
frame record, and marks a value that is not the frame's own chunk
``(applied)``.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from modules.lumascope_api.frame_record import FrameRecord, SettingSource
from modules.protocol_image_writer import _capture_evidence
from tests.scope_fakes import build_scope, bind_settings_like_a_session


@pytest.fixture
def live_scope():
    # A Classic scope: its FX2 camera stamps no chunks.
    scope = build_scope(simulate=True, sim_model='LS620')
    bind_settings_like_a_session(scope)
    scope.imaging.enable_streaming()
    yield scope
    scope.imaging.disable_streaming()
    scope.disconnect()


def test_a_camera_without_chunks_logs_the_applied_values_marked_applied(live_scope):
    applied_exp = live_scope.camera.set_exposure_ms(50.0)
    applied_gain = live_scope.camera.set_gain_db(6.0)
    image = live_scope.imaging._capture_and_wait_impl(
        accept_dark=True, all_ones_check=True, timeout_s=1.0
    )
    assert image is not None
    info = live_scope.imaging.last_capture_info
    assert info['chunk_exposure_us'] is None, 'the FX2 stamps no chunks'

    evidence = _capture_evidence(image, info)

    assert f'exp_ms={applied_exp:.2f}(applied)' in evidence
    assert f'gain_db={applied_gain:.2f}(applied)' in evidence
    assert '=na' not in evidence


def _record(**fields):
    defaults = {
        'captured_at': datetime.datetime.now(),
        'exposure_ms': 62.003,
        'exposure_source': SettingSource.FRAME,
        'gain_db': 3.5,
        'gain_source': SettingSource.FRAME,
        'black_level': None,
        'illumination_ma': {},
        'frames_summed': 1,
        'frame_significant_bits': 12,
        'camera_timestamp_ticks': None,
        'camera_tick_hz': None,
        'frame_id': None,
        'binning_size': 1,
        'camera_model': 'a camera that stamps chunks',
        'clock_mapping': None,
        'judgments': (),
        'saturated_fraction': 0.0,
        'dark': None,
        'held_channel': None,
        'auto_gain': None,
    }
    return FrameRecord(**{**defaults, **fields})


def test_a_camera_with_chunks_logs_the_frames_own_values_unmarked():
    info = {'chunk_exposure_us': 62003.0, 'chunk_gain_db': 3.5, 'frame_record': _record()}

    evidence = _capture_evidence(np.zeros((8, 8), dtype=np.uint8), info)

    assert 'exp_ms=62.00 ' in evidence
    assert 'gain_db=3.50' in evidence
    assert '(applied)' not in evidence


def test_the_line_states_the_frames_saturation_and_the_images_mean():
    """The saturation is the frame record's, measured by the capture at each
    frame's own depth; the mean is the still's statistics."""
    info = {'frame_record': _record(saturated_fraction=0.25)}

    evidence = _capture_evidence(np.full((8, 8), 4095, dtype=np.uint16), info)

    assert evidence.startswith('mean=4095.0 sat=25.0% ')
