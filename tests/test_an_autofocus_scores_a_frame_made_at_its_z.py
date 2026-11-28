# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus score comes from a frame made at the Z it is recorded at.

The sweep's captures told frame validity to ignore Z motion, so a frame
already on its way when a step's move went out was scored and recorded at
the new Z: every point of the curve could carry the previous step's score.
The simulated camera takes Z when a frame's exposure starts and has frames
in its pipeline when a move goes out, as a real camera does, so every move
finds a frame in flight; a sweep that scores it finds its focus a step
away from the true one.
"""

import contextlib
import logging
import re

import pytest

from _lumascope.modules.autofocus_thread import AutofocusThread
from _lumascope.modules.frame_validity import FrameValiditySource
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.run_outcome import EndingLatch
from tests.af_drives import park_z
from tests.protocol_drives import held_run_claim, run_identity
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

FOCAL_Z_UM = 2000.0


def _autofocus(session) -> float:
    """Run one autofocus through the run's autofocus thread; return its focus Z."""
    runner = session.protocol_runner()
    af = runner.session._sequenced_capture_runner._autofocus_runner
    thread = AutofocusThread(afe=af)
    thread.start()
    try:
        future = thread.run_autofocus(
            run=run_identity(),
            objective_id=next(iter(session.scope.objectives)),
            center_um=session.scope.motion.get_current_position('Z'),
            led_color='BF',
            led_illumination=5.0,
            run_ending=EndingLatch(),
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
        session.scope.imaging.enable_streaming()
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
        focus = _autofocus(session)

    assert focus == pytest.approx(FOCAL_Z_UM, abs=50.0)


def test_every_scored_frame_counts_for_its_step_s_z_move(monkeypatch):
    """Every autofocus capture waits out every pending write, its own Z step
    included: each frame a score is taken on counts for the step's Z move
    (its record's ``z_move`` judgment), so it was made after the move.

    Whoever counted the move's frames -- the capture's drain, or the
    frames' delivery -- the judgment is the frame's own. A sweep whose
    captures excluded Z motion keeps the frame in flight at each move,
    which does not count for it.
    """
    stills = []
    with _session_over_the_focus_target() as session:
        capture_still = session.scope.imaging.capture_still

        def recorded_capture(**kwargs):
            still = capture_still(**kwargs)
            stills.append(still)
            return still

        monkeypatch.setattr(session.scope.imaging, 'capture_still', recorded_capture)
        _autofocus(session)

    z_moves = [
        judgment
        for still in stills
        for judgment in still.record.judgments
        if judgment.change.source == FrameValiditySource.Z_MOVE
    ]
    assert len(z_moves) == len(stills) > 0, 'a scored frame was judged against no Z move'
    uncounted = [judgment for judgment in z_moves if not judgment.counts]
    assert not uncounted, f'frames scored before their Z move was met: {uncounted[:3]}'


def test_every_scored_point_states_its_frame_s_saturated_and_masked_shares(caplog):
    """Each point of a sweep logs the saturated share of the frame it
    scored, the capture's own measurement (``FrameRecord.saturated_fraction``),
    and the share of the scored region the score's mask left out
    (``masked_fraction``): a clipped frame scores differently, so a reader
    of ``autofocus.log`` can tell a jump in the curve that saturation made,
    and whether the score itself saw it.
    """
    with _session_over_the_focus_target() as session:
        session.scope.camera.set_exposure_ms(2.0)
        with caplog.at_level(logging.INFO, logger='LVP.autofocus'):
            _autofocus(session)

    points = [m for m in caplog.messages if m.lstrip().startswith('Z=')]
    assert points, 'the sweep logged no scored point'
    unstated = [m for m in points if not re.search(r' sat=\d+\.\d% masked=\d+\.\d%$', m)]
    assert not unstated, f'points without their saturated and masked shares: {unstated[:3]}'
