# Copyright Etaluma, Inc.
"""Regression test: frame gain/exposure metadata comes from the camera chunk.

The per-frame chunk carries the camera's ACTUAL ExposureTime + Gain for that
frame (the same values frame_validity checks the camera settled to). Image
metadata previously re-read gain/exposure LIVE per frame -- redundant, and
racing the next step's settings. generate_image_metadata now sources them
from the grab-time chunk (microseconds -> ms for exposure, dB for gain),
falling back to the live getter only when no chunk is present.

Driven behaviorally: a stub scope provides distinct chunk vs live values so
each test proves which source won.
"""

from types import SimpleNamespace

from modules.image_save import generate_image_metadata


LIVE_EXPOSURE_MS = 33.0
LIVE_GAIN_DB = 4.5


def _metadata_scope(chunks, chunk_reads=None):
    """Stub scope covering exactly what generate_image_metadata touches.

    chunks: the dict the camera handler returns (None = no chunk support).
    chunk_reads: optional list that records each get_last_chunks() call.
    """

    def get_last_chunks():
        if chunk_reads is not None:
            chunk_reads.append(1)
        return chunks

    runtime_state = SimpleNamespace(
        _objective={'focal_length': 9.0},
        _labware=SimpleNamespace(config={'rows': 8, 'columns': 12, 'standard': 'SBS'}),
        _stage_offset={'x': 0, 'y': 0},
        _coordinate_transformer=SimpleNamespace(
            stage_to_plate=lambda **kwargs: (1.0, 2.0)
        ),
        get_well_label=lambda: 'A1',
    )
    return SimpleNamespace(
        runtime_state=runtime_state,
        imaging=SimpleNamespace(
            _binning_size=1,
            get_exposure_time=lambda: LIVE_EXPOSURE_MS,
            get_gain=lambda: LIVE_GAIN_DB,
        ),
        diagnostics=SimpleNamespace(
            get_microscope_model=lambda: 'LS720-SIM',
            get_motor_info=lambda: {'serial_number': 'SN1', 'firmware_version': 'fw'},
            get_camera_info=lambda: {'model': 'simcam'},
        ),
        illumination=SimpleNamespace(get_led_ma=lambda color: 100.0),
        _camera_driver=SimpleNamespace(
            cam_image_handler=SimpleNamespace(get_last_chunks=get_last_chunks),
            timestamp_tick_frequency_hz=1_000_000_000,
        ),
    )


def test_exposure_metadata_prefers_chunk_with_us_to_ms_conversion():
    scope = _metadata_scope({'ExposureTime': 5000.0, 'Gain': 3.0})
    metadata = generate_image_metadata(scope, color='BF', x=0, y=0, z=0)
    assert metadata['exposure_time_ms'] == 5.0, (
        'chunk ExposureTime (us) must win over the live read and convert '
        f'to ms; got {metadata["exposure_time_ms"]}'
    )


def test_gain_metadata_prefers_chunk_with_live_fallback():
    scope = _metadata_scope({'ExposureTime': 5000.0, 'Gain': 3.0})
    metadata = generate_image_metadata(scope, color='BF', x=0, y=0, z=0)
    assert metadata['gain_db'] == 3.0, (
        f'chunk Gain must win over the live read; got {metadata["gain_db"]}'
    )

    no_chunk = generate_image_metadata(
        _metadata_scope(None), color='BF', x=0, y=0, z=0
    )
    assert no_chunk['exposure_time_ms'] == LIVE_EXPOSURE_MS
    assert no_chunk['gain_db'] == LIVE_GAIN_DB


def test_chunk_provenance_fields_recorded():
    scope = _metadata_scope(
        {'ExposureTime': 5000.0, 'Gain': 3.0, 'Timestamp': 42, 'FrameID': 7}
    )
    metadata = generate_image_metadata(scope, color='BF', x=0, y=0, z=0)
    assert metadata['timestamp_camera_ticks'] == 42
    assert metadata['timestamp_camera_tick_hz'] == 1_000_000_000
    assert metadata['frame_id'] == 7


def test_chunk_read_not_duplicated():
    """The chunk is read once and reused for gain/exposure + timestamp/
    frame-id (was two separate get_last_chunks() calls)."""
    reads = []
    scope = _metadata_scope({'ExposureTime': 5000.0, 'Gain': 3.0}, chunk_reads=reads)
    generate_image_metadata(scope, color='BF', x=0, y=0, z=0)
    assert len(reads) == 1, (
        f'get_last_chunks() must run once per metadata build; ran {len(reads)}x'
    )
