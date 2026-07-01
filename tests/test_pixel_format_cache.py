# Copyright Etaluma, Inc.
"""Regression test: get_pixel_format() serves from a cache, not a live read.

get_camera_info() is called once per saved frame to stamp image metadata,
and it reads the camera's PixelFormat. PixelFormat only changes through
set_pixel_format(), so the pylon and ids drivers cache it (refreshed in the
setter, cleared on disconnect) instead of hitting the SDK node map on every
capture.

Behavioral since the typed pypylon stub + bare-driver builders landed:
each test drives the real getter/setter and watches the node map.
"""

from unittest.mock import MagicMock

from tests.camera_fakes import bare_ids_camera, bare_pylon_camera


class TestPylonPixelFormatCache:
    def test_getter_consults_cache(self):
        cam = bare_pylon_camera()
        cam._pixel_format_cache = 'Mono12'
        assert cam.get_pixel_format() == 'Mono12'
        cam.active.PixelFormat.GetValue.assert_not_called()

    def test_getter_populates_cache_on_first_live_read(self):
        cam = bare_pylon_camera()
        cam._pixel_format_cache = None
        cam.active.PixelFormat.GetValue.return_value = 'Mono8'
        assert cam.get_pixel_format() == 'Mono8'
        assert cam._pixel_format_cache == 'Mono8'

    def test_setter_refreshes_cache(self):
        cam = bare_pylon_camera()
        cam._pixel_format_cache = None
        cam.get_supported_pixel_formats = lambda: ['Mono12']
        cam.active.PixelFormat.GetValue.return_value = 'Mono8'
        assert cam.set_pixel_format('Mono12') is True
        assert cam._pixel_format_cache == 'Mono12'


class TestIdsPixelFormatCache:
    def test_getter_consults_cache(self):
        cam = bare_ids_camera()
        cam._pixel_format_cache = 'Mono8'
        assert cam.get_pixel_format() == 'Mono8'
        cam.remote_nodemap.FindNode.assert_not_called()

    def test_setter_refreshes_cache(self):
        cam = bare_ids_camera()
        cam._pixel_format_cache = None
        cam._resolve_logical_format = lambda fmt: 'Mono8'
        assert cam.set_pixel_format('Mono8') is True
        assert cam._pixel_format_cache == 'Mono8'

    def test_cache_updates_before_grab_restart(self):
        # update_camera_config()'s __exit__ restarts grabbing. The cache must be
        # written INSIDE the guard (before the restart), or frames flow under the
        # new format while get_pixel_format()/get_camera_info() still report the
        # old cached value. Drive a real stop/start bounce and capture the cache
        # value at the instant grabbing restarts.
        cam = bare_ids_camera()
        cam._pixel_format_cache = 'Mono8'
        cam._resolve_logical_format = lambda fmt: 'Mono12'
        cam.is_grabbing = lambda: True  # force update_camera_config to bounce
        cam.stop_grabbing = MagicMock()
        seen = {}

        def _spy_start():
            seen['cache_at_restart'] = cam._pixel_format_cache

        cam.start_grabbing = _spy_start
        assert cam.set_pixel_format('Mono12') is True
        assert seen['cache_at_restart'] == 'Mono12'  # already updated pre-restart
        assert cam._pixel_format_cache == 'Mono12'
