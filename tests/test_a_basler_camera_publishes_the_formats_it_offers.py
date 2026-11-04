# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler camera's capabilities list the pixel formats the camera offers.

Capabilities publish the profile's `pixel_formats` (REST and the SDK read
it). On a Basler body that list was the static profile entry's, so a
daA3840-45um advertised Mono10 / Mono10p, which the camera does not offer.
The connect-time query now records the camera's own list, as the IDS driver
does; a list the camera cannot report keeps the documented one.
"""

from __future__ import annotations

from drivers.camera_profiles import lookup_profile
from tests.camera_fakes import bare_pylon_camera
from tests.scope_fakes import capabilities_of


def _connected_dart(offered):
    cam = bare_pylon_camera()
    cam.profile = lookup_profile('daA3840-45um')
    cam.active.PixelFormat.GetSymbolics.return_value = offered
    cam._query_dynamic_capabilities()
    return cam


def test_the_profile_and_capabilities_list_what_the_camera_offers():
    cam = _connected_dart(('Mono8', 'Mono12'))
    assert cam.profile.pixel_formats == ['Mono8', 'Mono12']
    assert capabilities_of(cam).camera_pixel_formats == ('Mono8', 'Mono12')


def test_an_unreadable_list_keeps_the_documented_formats():
    cam = _connected_dart(())
    assert cam.profile.pixel_formats == ['Mono8', 'Mono12', 'Mono12p']
