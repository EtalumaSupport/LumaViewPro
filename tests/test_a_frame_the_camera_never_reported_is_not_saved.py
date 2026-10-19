# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame the camera never reported is not saved as 0x0.

The camera cache once seeded its frame as 0x0, meaning "not read", and
bring-up, a binning change and the camera restore saved it as the delivered
one, so a camera that never answered a frame-size read left the settings at
a 0x0 frame (reproduced on the simulator, settings-leaf ranges pass 1). The
held frame is None until the camera reports one (``CameraAPI.frame_size``),
and each saves no frame while it is unknown.
"""

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def silent_frame_session(tmp_path, monkeypatch):
    # A stand-in by design: the simulator has no fault for a camera that
    # answers no frame-size read, and bring-up reads it inside create(),
    # before scope.simulation can inject one (a simulator gap).
    monkeypatch.setattr(SimulatedCamera, 'get_frame_size', lambda self: {})
    monkeypatch.setattr(SimulatedCamera, 'set_frame_size', lambda self, *a, **k: False)
    settings = complete_settings(live_folder=str(tmp_path))
    # Copied first: create prepares the dict it is handed in place.
    saved_frame = dict(settings['frame'])
    s = ScopeSession.create(settings, simulate=True)
    yield s, saved_frame
    s.shutdown()


def test_bring_up_and_a_binning_change_keep_the_saved_frame(silent_frame_session):
    session, saved_frame = silent_frame_session
    assert saved_frame['width'] > 0
    assert session.settings['frame'] == saved_frame

    size = session.scope.capabilities.camera_binning_sizes[-1]
    with pytest.raises(CameraSettingRejected):
        session.scope.camera.set_binning(size)

    assert session.settings['binning']['size'] == f'{size}x{size}'
    assert session.settings['frame'] == saved_frame
