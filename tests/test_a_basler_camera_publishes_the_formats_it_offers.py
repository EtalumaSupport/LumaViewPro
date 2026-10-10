# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler camera's capabilities list the pixel formats the camera offers.

Capabilities publish the profile's `pixel_formats` (REST and the SDK read
it). On a Basler body that list was the static profile entry's, so a
daA3840-45um advertised Mono10 / Mono10p, which the camera does not offer.
The connect-time query now records the camera's own list, as the IDS driver
does; a list the camera cannot report keeps the documented one.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from drivers.camera_profiles import lookup_profile
from modules.layer_record import UNRESOLVED
from modules.scope_capabilities import ScopeCapabilities
from tests.camera_fakes import bare_pylon_camera


def _connected_dart(offered):
    cam = bare_pylon_camera()
    cam.profile = lookup_profile('daA3840-45um')
    cam.active.PixelFormat.GetSymbolics.return_value = offered
    cam._query_dynamic_capabilities()
    return cam


def _capabilities(camera):
    motion = MagicMock()
    motion.detect_present_axes.return_value = ()
    motion.motorconfig = None
    led = MagicMock()
    led.available_channels.return_value = ()
    led.supports_firmware_stim.return_value = False
    return ScopeCapabilities.from_drivers(
        motion=motion, led=led, camera=camera, layer_identity=UNRESOLVED, scope_models={}
    )


def test_the_profile_and_capabilities_list_what_the_camera_offers():
    cam = _connected_dart(('Mono8', 'Mono12'))
    assert cam.profile.pixel_formats == ['Mono8', 'Mono12']
    assert _capabilities(cam).camera_pixel_formats == ('Mono8', 'Mono12')


def test_an_unreadable_list_keeps_the_documented_formats():
    cam = _connected_dart(())
    assert cam.profile.pixel_formats == ['Mono8', 'Mono12', 'Mono12p']
