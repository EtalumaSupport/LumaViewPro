# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LS460 is an LS560 with brightfield only: no fluorescence channel.

The LS460, LS560 and LS620 present identically to LumaViewPro -- the same
FX2 board, no motor board -- so the user declares which one it is, and the
model catalogue (``data/scopes.json``) says what that model has. A declared
LS460 is brought up as an LS560 is, with the LS560's lens and frame, and
offers BF alone: its LED for 488 nm is refused as not on this scope.
"""

from __future__ import annotations

import dataclasses

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.scope_capabilities import FrameSize
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED1


def _capabilities(tmp_path, microscope):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope=microscope),
        simulate=True,
        warn_pre_release=False,
    )
    return s, s.scope.capabilities


def test_a_declared_ls460_is_an_ls560_that_lights_brightfield_alone(tmp_path):
    ls560, ls560_caps = _capabilities(tmp_path / '560', 'LS560')
    ls560.shutdown()
    ls460, caps = _capabilities(tmp_path / '460', 'LS460')
    try:
        assert caps.led_colors == ('BF',)
        assert caps == dataclasses.replace(ls560_caps, led_colors=caps.led_colors)
        assert (caps.axes, caps.lens_focal_length_mm, caps.camera_limits.frame_max) == (
            (),
            32.3,
            FrameSize(1700, 1700),
        )
        with pytest.raises(HardwareCommandRefusedError) as refused:
            ls460.scope.illumination.led_on(LED1, 10.0)
        assert refused.value.reason == 'not_on_this_scope'
    finally:
        ls460.shutdown()
