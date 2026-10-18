# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The bit depth a client reads is the camera's, and the buffer width is named as such.

Before, ``capabilities.native_bit_depth`` was the only published depth and
was the buffers' width (16 for a Mono12 camera, whose data is 12 bits), so a
client scaling a frame by it was off by four bits, and the depth of the
frames the camera delivers now was not published at all. Now
``buffer_bit_depth`` is the width, ``camera_max_significant_bits`` the
deepest data depth the camera delivers, and ``imaging.significant_bits`` the
current frames' depth, following ``camera.set_image_mode``; with no camera it is
refused rather than answered with the buffer width.
"""

from __future__ import annotations

import pytest

from modules.image_mode import IMAGE_MODE_8BIT, IMAGE_MODE_12BIT_SCIENTIFIC
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def ls850t(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850T'), simulate=True
    )
    yield session
    session.shutdown()


def test_the_capabilities_name_the_deepest_depth_and_the_buffer_width(ls850t):
    caps = ls850t.scope.capabilities

    assert caps.camera_max_significant_bits == 12
    assert caps.buffer_bit_depth == 16


def test_the_current_depth_follows_the_image_mode(ls850t):
    ls850t.camera.set_image_mode(IMAGE_MODE_8BIT)
    assert ls850t.scope.imaging.significant_bits == 8

    ls850t.camera.set_image_mode(IMAGE_MODE_12BIT_SCIENTIFIC)
    assert ls850t.scope.imaging.significant_bits == 12
