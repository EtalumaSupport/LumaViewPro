# Copyright Etaluma, Inc.
"""Regression test: a frame's gain/exposure record comes from the camera chunk.

The per-frame chunk carries the camera's ACTUAL ExposureTime + Gain for that
frame (the same values frame_validity checks the camera settled to). The
record used to be built when the file was written, re-reading gain/exposure
live and racing the next step's settings. The capture now builds the frame's
record beside the grab: from the grab-time chunk (microseconds -> ms for
exposure, dB for gain), falling back to a fresh read of the camera
(``read_exposure_ms`` / ``read_gain_db``) only when no chunk is present. The
save writes what the record says.

Driven behaviorally: a stub imaging API provides distinct chunk vs live
values so each test proves which source won.
"""

import datetime
import threading
from types import SimpleNamespace

from modules.image_save import generate_image_metadata
from modules.lumascope_api.camera import CameraAPI
from modules.lumascope_api.imaging import ImagingAPI
from tests.frame_records import frame_record, plate
from modules.layer_record import release_color


LIVE_EXPOSURE_MS = 33.0
LIVE_GAIN_DB = 4.5
GRABBED_AT = datetime.datetime(2026, 9, 29, 12, 0, 0)


def _imaging(chunk_reads=None):
    """An imaging API and its camera API, with just what building a frame's record touches.

    The camera's fresh reads answer ``LIVE_EXPOSURE_MS`` and ``LIVE_GAIN_DB``.
    chunk_reads: optional list that records each read of the camera's chunks.
    """

    def get_last_chunks():
        if chunk_reads is not None:
            chunk_reads.append(1)
        return None

    scope = SimpleNamespace(
        capabilities=SimpleNamespace(
            camera_model='simcam',
            camera_timestamp_tick_hz=1_000_000_000,
            camera_has_clock_latch=False,
        ),
        _camera_driver=SimpleNamespace(
            active=True,
            get_black_level=lambda: None,
            cam_image_handler=SimpleNamespace(get_last_chunks=get_last_chunks),
        ),
    )
    camera = object.__new__(CameraAPI)
    camera._scope = scope
    camera._camera_cache_lock = threading.Lock()
    camera._camera_cache_write_gen = {}
    camera._camera_cache = {'binning': 1}
    camera._state_lock = threading.Lock()
    camera._held_layer = None
    camera.read_exposure_ms = lambda: LIVE_EXPOSURE_MS
    camera.read_gain_db = lambda: LIVE_GAIN_DB
    imaging = object.__new__(ImagingAPI)
    imaging._scope = scope
    imaging._camera = camera
    # The reducer's account of the frame just made: one frame, 12 bits.
    imaging._state_lock = threading.Lock()
    imaging._last_frame_summing = (1, 12)
    return imaging


def _record(imaging, chunks):
    return imaging._build_frame_record(
        chunks=chunks or {},
        lit=frozenset({(3, 50.0)}),
        captured_at=GRABBED_AT,
        judgments=(),
        saturated_fraction=None,
        dark=None,
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


def test_tiff_write_path_tolerates_omitted_gain_exposure(sim_scope):
    """The structured TIFF writer must treat the omitted keys as optional
    fields (same contract as the per-frame timestamps): a frame whose
    exposure/gain is unknown still SAVES, with those TIFF fields absent --
    one unreadable value must never become a lost frame."""
    import numpy as np

    from modules.image_utils import generate_tiff_data

    metadata = generate_image_metadata(
        sim_scope,
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
    tiff = generate_tiff_data(data, metadata=metadata, image_type='ome', color=release_color('BF'))
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
