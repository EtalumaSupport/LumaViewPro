# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated camera can stall: its frames stop while it stays connected and grabbing.

A real camera's link or grab loop can stall without the device being
removed -- the camera reports itself connected and streaming, and no frame
arrives. The simulator has to be able to do the same, so that what LVP does
about a stalled stream can be shown and tested without hardware. The stall
is asked for with a launch argument (``--sim-camera-stall=AFTER,FOR``), which
reaches the camera through the session and the scope's construction.
"""

from __future__ import annotations

import threading
import time

import pytest

from _lumascope.drivers.simulated_camera import SimulatedCamera, SimulatedStall
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.camera_fakes import grab_a_frame_made_after_now
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings


def _count_over(cam, seconds):
    before = cam.frames_delivered
    time.sleep(seconds)
    return cam.frames_delivered - before


class TestAStallIsRefusedWhereItCannotHappen:
    @pytest.mark.parametrize('after_s,for_s', [(-1.0, 1.0), (0.0, 0.0), (1.0, -2.0)])
    def test_a_stall_that_cannot_happen_is_refused_when_built(self, after_s, for_s):
        with pytest.raises(ValueError, match='lasts more than 0 s'):
            SimulatedStall(after_s=after_s, for_s=for_s)

    def test_a_real_scope_refuses_a_stall(self):
        with pytest.raises(ArgumentRefusedError) as refused:
            build_scope(simulate=False, sim_camera_stall=SimulatedStall(0.0, 1.0))
        assert (refused.value.reason, refused.value.argument) == (
            'needs_simulated_scope',
            'sim_camera_stall',
        )

    def test_a_scope_simulated_with_an_fx2_refuses_a_stall(self):
        with pytest.raises(ArgumentRefusedError) as refused:
            build_scope(simulate=True, sim_model='LS620', sim_camera_stall=SimulatedStall(0.0, 1.0))
        assert (refused.value.reason, refused.value.argument) == (
            'needs_simulated_camera',
            'sim_camera_stall',
        )

    def test_a_scope_simulated_without_its_camera_refuses_each_of_the_cameras_options(self):
        # Each is the simulated camera's; refused before anything starts.
        before = set(threading.enumerate())
        refused = {}
        for argument, value in (
            ('sim_camera_stall', SimulatedStall(0.0, 1.0)),
            ('sim_focal_spread_um', 25.0),
            ('sim_conversion_gain_mode', True),
        ):
            with pytest.raises(ArgumentRefusedError) as raised:
                build_scope(simulate=True, sim_camera_present=False, **{argument: value})
            refused[argument] = (raised.value.reason, raised.value.argument)
        assert refused == {
            'sim_camera_stall': ('needs_simulated_camera', 'sim_camera_stall'),
            'sim_focal_spread_um': ('needs_simulated_camera', 'sim_focal_spread_um'),
            'sim_conversion_gain_mode': ('needs_simulated_camera', 'sim_conversion_gain_mode'),
        }
        assert [t for t in threading.enumerate() if t not in before] == []

    def test_a_real_scope_refuses_a_camera_left_off_the_bus(self):
        with pytest.raises(ArgumentRefusedError) as refused:
            build_scope(simulate=False, sim_camera_present=False)
        assert (refused.value.reason, refused.value.argument) == (
            'needs_simulated_scope',
            'sim_camera_present',
        )


class TestTheStreamStopsAndResumes:
    def test_frames_stop_for_the_stall_and_resume_after_it(self):
        cam = SimulatedCamera(width=48, height=24)
        cam.exposure_t(1.0)  # the delivery ceiling: frames well inside every window below
        cam.open_and_start()
        grab_a_frame_made_after_now(cam)
        cam.hold_frames(SimulatedStall(after_s=0.3, for_s=0.6))
        assert _count_over(cam, 0.2) > 0, 'frames stopped before the stall began'
        time.sleep(0.15)  # into the stall
        assert _count_over(cam, 0.3) == 0, 'a frame arrived during the stall'
        assert cam.is_grabbing() and cam.is_connected(), (
            'a stall is frames stopping, not the camera going away'
        )
        time.sleep(0.2)  # past the end of the stall
        assert _count_over(cam, 0.2) > 0, 'frames did not resume after the stall'

    def test_a_scope_built_with_a_stall_stops_its_cameras_frames(self):
        scope = build_scope(
            simulate=True, sim_model='LS850T', sim_camera_stall=SimulatedStall(0.0, 30.0)
        )
        cam = scope._camera_driver
        cam.open_and_start()
        assert cam.is_grabbing()
        assert _count_over(cam, 0.5) == 0, (
            'the stall given at construction did not reach the camera'
        )


def test_a_session_created_with_a_stall_stops_its_cameras_frames(tmp_path):
    # The session's door: the record's stall reaches the camera the
    # session's scope is built with, through bring-up's own stream start.
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
        simulated=SimulatedScope(camera_stall=SimulatedStall(0.0, 30.0)),
        warn_pre_release=False,
    )
    try:
        cam = session.scope._camera_driver
        assert cam.is_grabbing(), 'bring-up did not start the stream'
        assert _count_over(cam, 0.5) == 0, 'the stall given to create did not reach the camera'
    finally:
        session.shutdown()
