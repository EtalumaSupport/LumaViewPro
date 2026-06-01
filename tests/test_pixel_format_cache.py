# Copyright Etaluma, Inc.
"""Regression test: get_pixel_format() serves from a cache, not a live read.

get_camera_info() is called once per saved frame to stamp image metadata,
and it reads the camera's PixelFormat. PixelFormat only changes through
set_pixel_format(), so the pylon and ids drivers cache it (refreshed in the
setter, cleared on disconnect) instead of hitting the SDK node map on every
capture.

Source-scan guards: the SDK-coupled camera drivers can't be unit-constructed
(the base ctor auto-connects against a real SDK), so these assert the cache
is consulted in the getter and refreshed in the setter -- catching a future
revert to an unconditional live read.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _func_body(rel, func):
    src = (REPO_ROOT / rel).read_text()
    start = src.find(f'def {func}(')
    assert start != -1, f'{func} not found in {rel}'
    body = src[start : start + 1500]
    end = body.find('\n    def ', 1)
    return body if end == -1 else body[:end]


class TestPylonPixelFormatCache:
    def test_getter_consults_cache(self):
        body = _func_body('drivers/pyloncamera.py', 'get_pixel_format')
        assert '_pixel_format_cache' in body, (
            'pylon get_pixel_format must serve from _pixel_format_cache before '
            'a live GenICam node read (called once per saved frame)'
        )

    def test_setter_refreshes_cache(self):
        body = _func_body('drivers/pyloncamera.py', 'set_pixel_format')
        assert '_pixel_format_cache' in body, (
            'pylon set_pixel_format must refresh _pixel_format_cache so the '
            'getter never serves a stale format'
        )


class TestIdsPixelFormatCache:
    def test_getter_consults_cache(self):
        body = _func_body('drivers/idscamera.py', 'get_pixel_format')
        assert '_pixel_format_cache' in body, (
            'ids get_pixel_format must serve from _pixel_format_cache before '
            'a live node-map read'
        )

    def test_setter_refreshes_cache(self):
        body = _func_body('drivers/idscamera.py', 'set_pixel_format')
        assert '_pixel_format_cache' in body, (
            'ids set_pixel_format must refresh _pixel_format_cache'
        )
