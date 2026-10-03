# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LS560 image's scale follows its 32.3 mm lens.

The model catalogue gave the LS560 the LS620's 47.8 mm lens, so every LS560
image recorded a pixel size and field of view 1.48 times too small. The
LS460 and LS560 lens is 32.3 mm (Eric, 2026-10-02), the reason LumaView
Classic capped their window at 1700.
"""

from __future__ import annotations

import pytest

import modules.common_utils as common_utils
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def ls560(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS560'),
        simulate=True,
        warn_pre_release=False,
    )
    yield s
    s.shutdown()
    s.scope.disconnect()


def test_the_ls560_lens_is_32_3_mm(ls560):
    assert ls560.scope.capabilities.lens_focal_length_mm == 32.3


def test_an_ls560_image_records_the_pixel_size_its_lens_gives(ls560):
    # 2.2 um pixels behind the 32.3 mm lens with a 9 mm (20x) objective.
    um_per_pixel = common_utils.get_pixel_size(
        focal_length=9.0, binning_size=1, capabilities=ls560.scope.capabilities
    )
    assert um_per_pixel == pytest.approx(2.2 / (32.3 / 9.0))
