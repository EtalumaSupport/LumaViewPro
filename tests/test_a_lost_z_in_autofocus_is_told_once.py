# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Z lost during autofocus is told by its fault and by the run's ending, nothing more.

One Z move that stalled mid-sweep used to be told four times: the stall, then
autofocus renaming it an unexpected error, then autofocus's restore that
could not move a Z the stall had lost, then the run's ending. The stall is
reported where the motion monitor finds it; autofocus passes every typed
outcome on as itself, and says nothing when it ends on a lost Z, whose loss
was told and on which the run ends.

The stall is the simulated motor board's own: the sweep's third frame arms a
hold on its next Z move, which never arrives, and the motion monitor gives
the axis up.
"""

import threading

import pytest

from modules.exceptions import (
    AutofocusFailedError,
    AutofocusZNotRestoredError,
    MoveFaultError,
    RunFailedError,
)
from tests.test_composite_run_e2e import headless_settings, open_composite_session

RESULT_TIMEOUT_S = 60.0


@pytest.fixture
def af_session(tmp_path):
    settings = headless_settings(tmp_path, acquiring=('BF',))
    # A protocol built from these focuses each step from the same Z.
    settings['BF']['autofocus'] = True
    settings['BF']['focus'] = 3000.0
    with open_composite_session(settings) as (session, runner):
        # Clear of the travel limit the home leaves Z at, over a scene with a
        # focus there, so the sweep reaches its frames.
        session.scope.motion.move_z(3000.0)
        camera = session.scope._camera_driver
        camera.set_test_pattern(enabled=True, pattern='focus_target')
        camera.set_focal_z(3000.0)
        yield session, runner


def _stall_the_sweeps_z(monkeypatch, session):
    # a stand-in by design: the simulator holds an axis's next move, and only
    # the sweep knows when its next move is; its third frame arms the hold.
    imaging_type = type(session.scope.imaging)
    real = imaging_type.capture_and_wait
    frames = []

    def arming(imaging, *args, **kwargs):
        if threading.current_thread().name == 'autofocus_thread':
            frames.append(None)
            if len(frames) == 3:
                session.scope.simulation.hold_axis('Z', 0.5)
        return real(imaging, *args, **kwargs)

    monkeypatch.setattr(imaging_type, 'capture_and_wait', arming)
    return frames


def _record_reports(monkeypatch):
    from modules.notification_center import notifications

    reported = []
    real = notifications.report_outcome

    def recorded(outcome, *args, **kwargs):
        reported.append(outcome)
        return real(outcome, *args, **kwargs)

    monkeypatch.setattr(notifications, 'report_outcome', recorded)
    return reported


def _told(reported):
    return [(type(r).__name__, getattr(r, 'reason', None)) for r in reported]


def _assert_told_by_the_stall_and_the_ending(reported, outcome):
    assert (outcome.status, outcome.reason) == ('failed', 'position_lost'), outcome
    assert _told(reported) == [
        ('MoveFaultError', 'stalled'),
        ('RunFailedError', 'position_lost'),
    ], _told(reported)
    assert not [
        r for r in reported if isinstance(r, (AutofocusFailedError, AutofocusZNotRestoredError))
    ]
    stall, ending = reported
    assert isinstance(stall, MoveFaultError) and isinstance(ending, RunFailedError)
    assert 'Home the scope' in str(ending)


def test_a_single_autofocus_that_loses_z_is_told_by_the_stall_and_the_ending(
    af_session, monkeypatch
):
    session, runner = af_session
    frames = _stall_the_sweeps_z(monkeypatch, session)
    reported = _record_reports(monkeypatch)

    outcome = runner.start_autofocus(layer='BF').result(timeout=RESULT_TIMEOUT_S)

    assert len(frames) >= 3, 'the sweep never reached the frame that arms the stall'
    _assert_told_by_the_stall_and_the_ending(reported, outcome)


def test_a_protocol_step_whose_autofocus_loses_z_is_told_by_the_stall_and_the_ending(
    af_session, monkeypatch, tmp_path
):
    session, runner = af_session
    protocol = session.new_protocol()
    while protocol.num_steps() > 1:
        protocol.delete_step(protocol.num_steps() - 1)
    assert protocol.step_row(idx=0)['Auto_Focus']
    frames = _stall_the_sweeps_z(monkeypatch, session)
    reported = _record_reports(monkeypatch)

    outcome = runner.start_single_scan(
        protocol, parent_dir=str(tmp_path / 'run'), enable_image_saving=True
    ).result(timeout=RESULT_TIMEOUT_S)

    assert len(frames) >= 3, 'the sweep never reached the frame that arms the stall'
    _assert_told_by_the_stall_and_the_ending(reported, outcome)
