# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pylon's grab restarts after a format change with the new format already cached.

``set_pixel_format`` wrote the cache after ``update_camera_config()`` had
restarted the grab, so the restart read the old format: every format change
logged ``Grabbing: pixel_format=Mono8`` after ``SetValue('Mono12')``, and
back. The IDS driver sets its cache inside the config guard for the same
reason.
"""

from tests.camera_fakes import bare_pylon_camera


def test_the_grab_restarts_with_the_new_format_cached():
    cam = bare_pylon_camera()
    cam.active.PixelFormat.GetSymbolics.return_value = ('Mono8', 'Mono12')
    cam.active.PixelFormat.GetValue.return_value = 'Mono8'
    cam._pixel_format_cache = 'Mono8'
    cam.active.IsGrabbing.return_value = True
    cam.stop_grabbing = lambda: None
    seen_at_restart = []
    cam.start_grabbing = lambda *a, **k: seen_at_restart.append(cam.get_pixel_format())

    assert cam.set_pixel_format('Mono12') is True

    assert seen_at_restart == ['Mono12']
