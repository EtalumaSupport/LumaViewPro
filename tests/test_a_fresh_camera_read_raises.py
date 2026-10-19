# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A fresh camera read asks the camera, and raises when the camera does not answer.

``scope.camera.read_*`` reads the setting from the camera now. A read the
camera did not answer raises ``CameraReadFailedError`` naming the setting;
it never answers the last value as if it were new. None means only that
there is no camera, or that this camera has no such setting. A saved frame
says what the instrument did, so a capture whose gain or exposure cannot be
read from its frame or from the camera fails, and saves nothing.

Run on the simulated LS850, its reads failed through
``scope.simulation.refuse_camera_read``.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from drivers.exceptions import CameraReadFailedError
from modules.lumascope_api.camera import CameraAPI
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# (the read, the simulator's node for it, a failed-read answer its driver gives)
_READS = (
    ('read_gain_db', 'gain', 'sentinel'),
    ('read_exposure_ms', 'exposure', 'sentinel'),
    ('read_binning', 'binning', 'sentinel'),
    ('read_frame_size', 'frame_size', 'sentinel'),
    ('read_pixel_format', 'pixel_format', 'raise'),
    ('read_black_level', 'black_level', 'raise'),
    ('read_frame_rate', 'frame_rate', 'raise'),
)


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


def test_every_fresh_read_is_in_the_sweep():
    """A ``read_*`` added to the camera is swept below, or this names it."""
    reads = {name for name in vars(CameraAPI) if name.startswith('read_')}
    assert reads == {read for read, _node, _answer in _READS}


def test_every_fresh_read_raises_naming_its_setting_when_the_camera_does_not_answer(sim_scope):
    camera = sim_scope.camera
    unraised = []
    for read, node, answer in _READS:
        sim_scope.simulation.refuse_camera_read(node, answer)
        try:
            getattr(camera, read)()
        except CameraReadFailedError as failed:
            if node.replace('_', ' ') not in str(failed):
                unraised.append(f'{read}: the refusal does not name the {node}: {failed}')
        else:
            unraised.append(f'{read} answered after a failed read')
        # The next read is answered: the refusal was one read.
        if getattr(camera, read)() is None:
            unraised.append(f'{read} answered None from an answering camera')
    assert unraised == [], '\n'.join(unraised)


def test_with_no_camera_every_fresh_read_answers_none(diagnostic_scope):
    answered = {read: getattr(diagnostic_scope.camera, read)() for read, _node, _answer in _READS}
    assert answered == dict.fromkeys(answered)


def test_a_chunkless_still_whose_exposure_read_fails_saves_nothing(session, monkeypatch, tmp_path):
    """The IDS and FX2 path: their frames carry no exposure, so the camera is read."""
    driver = session.scope._camera_driver
    real = driver.grab_new_capture
    # a stand-in by design: the simulated camera always stamps a Basler body's
    # chunks; a chunkless simulated camera is the simulator's gap
    monkeypatch.setattr(
        driver,
        'grab_new_capture',
        lambda timeout_s: (
            frame if (frame := real(timeout_s)) is None else dataclasses.replace(frame, chunks=None)
        ),
    )
    session.scope.simulation.refuse_camera_read('exposure', 'raise', count=10)

    with pytest.raises(CameraReadFailedError) as failed:
        session.manual_capture.capture(false_color_on=False).result(timeout=30)

    assert 'exposure' in str(failed.value)
    assert [name for _dir, _dirs, files in os.walk(tmp_path) for name in files] == []
