# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated pattern's focus score is highest at the focal Z and falls on both sides.

The cells and the Ronchi ruling were drawn through the objective's in-focus
resolution and the focus target was not: its one-pixel grid lines scored
higher under Vollath F4 slightly blurred than sharp, so its score peaked
about half a depth of field off focus at 20x and 40x, and the simulator
could not show an autofocus landing at focus. Every drawing now goes
through the one in-focus resolution, at binning 1, and is binned after,
as the sensor sums it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules import autofocus_functions
from _lumascope.modules.scope_session import ScopeSession
from tests.af_drives import park_z
from tests.camera_fakes import grab_a_frame_made_after_now
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_FOCAL_Z_UM = 5000.0
# Off focus by these depths of field: beyond the defocus model's flat top,
# within a fifth of a depth of field at binning 1, where a frame is drawn
# with no defocus blur.
_OFF_FOCUS_DOF = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


@pytest.mark.parametrize('objective', ['20x Oly', '40x Meiji'])
def test_the_focus_target_scores_highest_at_the_focal_z(objective):
    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        scope = session.scope
        scope.imaging.enable_streaming()
        home_sim_scope(scope)
        camera = scope._camera_driver
        camera.set_test_pattern(enabled=True, pattern='focus_target')
        session.select_objective(objective)
        # A light at which the target does not clip: a clipped frame's
        # score is pulled off focus on the hardware as here.
        scope.illumination.led_on('BF', 2.0)
        camera.set_focal_z(_FOCAL_Z_UM)
        dof = camera._depth_of_field_func()

        def score_at(dz_dof: float) -> float:
            park_z(scope, _FOCAL_Z_UM + dz_dof * dof)
            image = scope.imaging.capture_still(accept_dark=True).image
            assert image.max() < np.iinfo(image.dtype).max, f'clipped at {dz_dof} DOF'
            return autofocus_functions.autofocus_score(image=image)

        at_focus = score_at(0.0)
        off_focus = {dz: score_at(dz) for dz in _OFF_FOCUS_DOF}
    finally:
        session.shutdown()

    higher = {dz: s for dz, s in off_focus.items() if s >= at_focus}
    assert not higher, (objective, at_focus, off_focus)


def test_a_binned_focus_target_is_the_unbinned_one_summed():
    """At binning 2 the target is the binning-1 frame binned: its features and blur are in sensor pixels."""
    cam = SimulatedCamera(width=256, height=192)
    cam.set_test_pattern(enabled=True, pattern='focus_target')
    cam.exposure_t(2.0)
    cam.gain(0.0)
    cam.open_and_start()
    try:
        unbinned = grab_a_frame_made_after_now(cam).image
        cam.set_binning_size(2)
        cam.set_frame_size(128, 96)
        binned = grab_a_frame_made_after_now(cam).image
    finally:
        cam.stop_grabbing()

    assert unbinned.max() < 255, 'the unbinned frame clipped'
    summed = cv2.resize(unbinned, (128, 96), interpolation=cv2.INTER_AREA)
    # The light scales each frame after it is binned, a level apart at most.
    assert np.abs(binned.astype(int) - summed.astype(int)).max() <= 1
