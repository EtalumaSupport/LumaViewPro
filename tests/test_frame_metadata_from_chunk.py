# Copyright Etaluma, Inc.
"""Regression test: a frame's gain/exposure record comes from the camera chunk.

The per-frame chunk carries the camera's ACTUAL ExposureTime + Gain for that
frame (the same values frame_validity checks the camera settled to). The
record used to be built when the file was written, re-reading gain/exposure
live and racing the next step's settings. The capture now builds the frame's
record beside the grab: from the grab-time chunk (microseconds -> ms for
exposure, dB for gain), falling back to the live-confirmed surface
(get_live_camera_settings, which omits a field whose read did not just
succeed) only when no chunk is present. The save writes what the record says.

Driven behaviorally: a stub imaging API provides distinct chunk vs live
values so each test proves which source won.
"""

import datetime
import threading
from types import SimpleNamespace

from modules.image_save import generate_image_metadata
from modules.lumascope_api.imaging import ImagingAPI
from modules import layer_record
from tests.frame_records import frame_record, plate


LIVE_EXPOSURE_MS = 33.0
LIVE_GAIN_DB = 4.5
GRABBED_AT = datetime.datetime(2026, 9, 29, 12, 0, 0)


def _imaging(live=None, chunk_reads=None):
    """An imaging API with just what building a frame's record touches.

    live: what get_live_camera_settings answers (default: both fields read).
    chunk_reads: optional list that records each read of the camera's chunks.
    """

    def get_last_chunks():
        if chunk_reads is not None:
            chunk_reads.append(1)
        return None

    imaging = object.__new__(ImagingAPI)
    imaging._scope = SimpleNamespace(
        illumination=SimpleNamespace(state_ch2color=lambda ch: {3: 'BF'}[ch]),
        _camera_driver=SimpleNamespace(
            get_model_name=lambda: 'simcam',
            timestamp_tick_frequency_hz=1_000_000_000,
            cam_image_handler=SimpleNamespace(get_last_chunks=get_last_chunks),
        ),
    )
    imaging._camera_cache_lock = threading.Lock()
    imaging._camera_cache = {'binning': 1}
    answer = {'gain_db': LIVE_GAIN_DB, 'exposure_ms': LIVE_EXPOSURE_MS} if live is None else live
    imaging.get_live_camera_settings = lambda: answer
    return imaging


def _record(imaging, chunks):
    return imaging._build_frame_record(
        chunks=chunks or {}, lit=frozenset({(3, 50.0)}), captured_at=GRABBED_AT, frames_summed=1
    )


def test_exposure_metadata_prefers_chunk_with_us_to_ms_conversion():
    record = _record(_imaging(), {'ExposureTime': 5000.0, 'Gain': 3.0})
    assert record.exposure_ms == 5.0, (
        'chunk ExposureTime (us) must win over the live read and convert '
        f'to ms; got {record.exposure_ms}'
    )


def test_gain_metadata_prefers_chunk_with_live_fallback():
    record = _record(_imaging(), {'ExposureTime': 5000.0, 'Gain': 3.0})
    assert record.gain_db == 3.0, f'chunk Gain must win over the live read; got {record.gain_db}'

    no_chunk = _record(_imaging(), None)
    assert no_chunk.exposure_ms == LIVE_EXPOSURE_MS
    assert no_chunk.gain_db == LIVE_GAIN_DB


def test_failed_live_read_omits_gain_exposure_keys():
    """A failed live read must never be written into saved metadata as if it
    were a real acquisition setting. get_live_camera_settings answers {} when
    no field's read just succeeded; unknown -> the record carries none."""
    record = _record(_imaging(live={}), None)
    assert record.exposure_ms is None, (
        f'failed exposure read must leave the record empty, not {record.exposure_ms}'
    )
    assert record.gain_db is None, (
        f'failed gain read must leave the record empty, not {record.gain_db}'
    )


def test_inactive_camera_zero_exposure_omitted():
    """Belt and braces: get_live_camera_settings can only return validated
    values by contract, but the record keeps its own validity guard. If a
    non-physical value (zero exposure, negative gain) ever slips through the
    live surface, the record still carries none -- a capture racing a
    disconnect must not fabricate exposure_time_ms=0.0 in frame metadata."""
    record = _record(_imaging(live={'exposure_ms': 0.0, 'gain_db': -1.0}), None)
    assert record.exposure_ms is None
    assert record.gain_db is None


def test_tiff_write_path_tolerates_omitted_gain_exposure():
    """The structured TIFF writer must treat the omitted keys as optional
    fields (same contract as the per-frame timestamps): a frame whose
    exposure/gain is unknown still SAVES, with those TIFF fields absent --
    one unreadable value must never become a lost frame."""
    import numpy as np

    from modules.image_utils import generate_tiff_data

    scope = SimpleNamespace(
        runtime_state=SimpleNamespace(
            get_objective_info=lambda objective_id: {'focal_length': 9.0}
        ),
        capabilities=SimpleNamespace(pixel_size_um=None, lens_focal_length_mm=None),
        diagnostics=SimpleNamespace(
            get_motor_info=lambda: {'serial_number': 'SN1', 'firmware_version': 'fw'}
        ),
        layer_identity=layer_record.UNRESOLVED,
    )
    metadata = generate_image_metadata(
        scope,
        channel='BF',
        plate_x_mm=0,
        plate_y_mm=0,
        stage_z_um=0,
        objective_id='4x Oly',
        frame_record=frame_record(exposure_ms=None, gain_db=None),
        labware=plate(),
        well_label=None,
    )
    metadata['significant_bits'] = 8  # write_tiff supplies this in production
    data = np.zeros((4, 4), dtype=np.uint8)
    tiff = generate_tiff_data(data, metadata=metadata, image_type='ome', color='BF')
    plane = tiff['metadata']['Plane']
    assert 'ExposureTime' not in plane
    assert 'Gain' not in plane


def test_chunk_provenance_fields_recorded():
    record = _record(
        _imaging(), {'ExposureTime': 5000.0, 'Gain': 3.0, 'Timestamp': 42, 'FrameID': 7}
    )
    assert record.camera_timestamp_ticks == 42
    assert record.camera_tick_hz == 1_000_000_000
    assert record.frame_id == 7


def test_chunk_read_not_duplicated():
    """The chunk is read once, by the capture, and handed to the record for
    gain/exposure + timestamp/frame-id; building the record does not read the
    camera's chunks a second time."""
    reads = []
    _record(_imaging(chunk_reads=reads), {'ExposureTime': 5000.0, 'Gain': 3.0})
    assert reads == [], f'building the record re-read the camera chunks {len(reads)}x'
