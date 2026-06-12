# Copyright Etaluma, Inc.
"""Regression test: frame gain/exposure metadata comes from the camera chunk.

The per-frame chunk carries the camera's ACTUAL ExposureTime + Gain for that
frame (the same values frame_validity checks the camera settled to). Image
metadata previously re-read gain/exposure LIVE per frame -- redundant, and
racing the next step's settings. generate_image_metadata now sources them
from the grab-time chunk (microseconds -> ms for exposure, dB for gain),
falling back to the live getter only when no chunk is present.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _metadata_fn_body():
    src = (REPO_ROOT / 'modules' / 'image_save.py').read_text()
    start = src.find('def generate_image_metadata(')
    assert start != -1, 'generate_image_metadata not found'
    end = src.find('\ndef ', start + 1)
    return src[start:end] if end != -1 else src[start:]


def test_exposure_metadata_prefers_chunk_with_us_to_ms_conversion():
    body = _metadata_fn_body()
    assert "chunks.get('ExposureTime')" in body, (
        'exposure metadata must read the chunk ExposureTime'
    )
    assert '/ 1000.0' in body, (
        'chunk ExposureTime is in microseconds; it must convert to ms'
    )
    assert 'else scope.imaging.get_exposure_time()' in body, (
        'the live exposure read must be the fallback only (no chunk present)'
    )


def test_gain_metadata_prefers_chunk_with_live_fallback():
    body = _metadata_fn_body()
    assert "chunks.get('Gain')" in body, 'gain metadata must read the chunk Gain'
    assert 'else scope.imaging.get_gain()' in body, (
        'the live gain read must be the fallback only (no chunk present)'
    )


def test_chunk_read_not_duplicated():
    """The chunk is read once and reused for gain/exposure + timestamp/frame-id
    (was two separate get_last_chunks() calls)."""
    body = _metadata_fn_body()
    assert body.count('get_last_chunks()') == 1, (
        'get_last_chunks() should be called once per metadata build, not twice'
    )
