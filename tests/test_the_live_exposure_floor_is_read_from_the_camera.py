# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera's live exposure floor is read from the camera, and a failed read raises.

``scope.camera.read_exposure_ms_min()`` answers the shortest exposure the
camera takes now: on IDS bodies it rises above the floor read at connect
(``capabilities.camera_limits.exposure_ms_min``) as the camera's other
settings change. None means no camera; a read that fails on a connected
camera is a different fact, so it raises, and a caller that sizes an
exposure sweep from the answer never mistakes it for an absent camera or a
floor. It replaced ``get_camera_profile_info``'s ``exposure_min_ms``.
"""

import pytest

from drivers.exceptions import CameraReadFailedError


def test_a_connected_camera_answers_its_floor(sim_scope):
    # The simulated camera declares 14 us.
    assert sim_scope.camera.read_exposure_ms_min() == pytest.approx(0.014)


def test_no_camera_answers_none(sim_scope):
    sim_scope._camera_driver.active = False
    assert sim_scope.camera.read_exposure_ms_min() is None


def test_a_failed_read_raises(sim_scope, monkeypatch):
    # A stand-in by design: the simulated camera's floor read cannot fail;
    # the IDS driver's failing read is pinned in test_ids_driver.py.
    def unreadable():
        raise CameraReadFailedError('Minimum exposure read failed: node unreadable')

    monkeypatch.setattr(sim_scope._camera_driver, 'get_min_exposure', unreadable)
    with pytest.raises(CameraReadFailedError, match='node unreadable'):
        sim_scope.camera.read_exposure_ms_min()
