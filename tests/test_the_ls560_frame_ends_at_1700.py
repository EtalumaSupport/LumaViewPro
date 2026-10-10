# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LS560's frame ends at 1700, where its lens's image ends.

LumaView Classic capped the LS560 and LS460 window at 1700: their lens has a
shorter focal length than the LS620's, so its image covers less of the
sensor. The model catalogue declares it (``MaxFrame``), the scope's frame
maximum is the smaller of it and the camera's, and a frame above it is
refused at the API, naming the range.
"""

from __future__ import annotations

import pytest

from modules.exceptions import CameraSettingOutOfRangeError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _session(tmp_path, microscope):
    return ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope=microscope,
            binning={'size': '1x1'},
            frame={'width': 1000, 'height': 1000},
        ),
        simulate=True,
        warn_pre_release=False,
    )


def test_the_ls560_refuses_a_frame_above_1700_and_takes_1700(tmp_path):
    s = _session(tmp_path, 'LS560')
    try:
        imaging = s.scope.imaging
        # The simulated FX2 sensor delivers 1900 x 1900; the lens bounds both.
        assert s.scope.capabilities.camera_max_frame_size == (1700, 1700)
        with pytest.raises(CameraSettingOutOfRangeError, match='1700'):
            imaging.set_frame_size(1704, 1000)
        assert imaging.set_frame_size(1700, 1000) == {'width': 1700, 'height': 1000}
    finally:
        s.shutdown()
        s.scope.disconnect()


def test_a_model_that_declares_no_maximum_is_bounded_by_its_camera(tmp_path):
    s = _session(tmp_path, 'LS620')
    try:
        assert s.scope.capabilities.camera_max_frame_size == (1900, 1900)
        assert s.scope.imaging.set_frame_size(1900, 1000) == {'width': 1900, 'height': 1000}
    finally:
        s.shutdown()
        s.scope.disconnect()
