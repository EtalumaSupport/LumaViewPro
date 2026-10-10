# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus score comes from a frame made at the Z it is recorded at.

The sweep's captures told frame validity to ignore Z motion, so a frame
already on its way when a step's move went out was scored and recorded at
the new Z: every point of the curve could carry the previous step's score.
Here the simulated camera delivers each frame a while after it reads Z, as
a real camera's pipeline does, so every move finds a frame in flight; a
sweep that scores it finds its focus a step away from the true one.
"""

import contextlib
import time

import pytest

from modules.autofocus_thread import AutofocusThread
from modules.scope_session import ScopeSession
from tests.af_drives import park_z
from tests.protocol_drives import held_run_claim, run_identity
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

FOCAL_Z_UM = 2000.0
# How long a frame takes to arrive after the camera has read Z: long next to
# a simulated Z step, so a move always finds a frame in flight.
FRAME_DELIVERY_DELAY_S = 0.06


def _autofocus(session) -> float:
    """Run one autofocus through the run's autofocus thread; return its focus Z."""
    runner = session.create_protocol_runner()
    af = runner.sequenced_capture_runner._autofocus_runner
    thread = AutofocusThread(afe=af)
    thread.start()
    try:
        future = thread.run_autofocus(
            run=run_identity('autofocus'),
            objective_id=session.scope.runtime_state.get_available_objectives()[0],
            led_color='BF',
            led_illumination=50.0,
            led_lease=session.scope.illumination.acquire_led_lease(
                'protocol', claim=held_run_claim()
            ),
        )
        return future.result(timeout=60)
    finally:
        thread.stop(timeout=2.0)


@contextlib.contextmanager
def _session_over_the_focus_target():
    """A homed simulated session streaming a focus target, Z parked above its focus."""
    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        session.scope.imaging.start_streaming()
        home_sim_scope(session.scope)
        camera = session.scope._camera_driver
        camera.set_test_pattern(enabled=True, pattern='focus_target')
        camera.set_focal_z(FOCAL_Z_UM)
        park_z(session.scope, 3000.0)
        yield session
    finally:
        session.shutdown()


@pytest.mark.slow
def test_the_sweep_finds_the_focus_with_a_frame_in_flight_at_every_move():
    with _session_over_the_focus_target() as session:
        camera = session.scope._camera_driver
        render = camera._generate_image

        def delivered_late():
            image = render()
            time.sleep(FRAME_DELIVERY_DELAY_S)
            return image

        camera._generate_image = delivered_late

        focus = _autofocus(session)

    assert focus == pytest.approx(FOCAL_Z_UM, abs=50.0)


def test_the_scored_frame_is_drained_past_the_step_s_z_move():
    """Every autofocus grab waits out every pending write, its own Z step
    included: the frame a score is taken on was made after the move.

    The last capture of a sweep follows that step's Z move, which leaves
    'z_move' pending for its skip count, so the frame the step scored is
    one the capture drained past it. A sweep that grabbed without the
    drain (a bare get_image), or that excluded Z motion from it, records
    no drained frame here.
    """
    with _session_over_the_focus_target() as session:
        # Bright enough that no frame is dark, so the sweep makes no
        # same-Z dark retry and its last capture is a step's; at the
        # simulator's 40 fps delivery ceiling it costs no time.
        session.scope.imaging.set_exposure_ms(20.0)
        _autofocus(session)
        info = session.scope.imaging.last_capture_info
        z_skip = session.scope.imaging.frame_validity.SKIP_FRAMES['z_move']

    assert info is not None, 'the sweep never captured through capture_and_wait'
    assert info['drained'] >= z_skip, (
        f'the last scored frame was not drained past its Z move: {info}'
    )
