# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every run that ends tells the person how it ended.

The notification centre hides a post whose (category, title) it showed in the
last ten seconds, on the premise that the post repeats something the person
was just told. A run's ending is told once per run, so a second same-titled
ending is another run's, and the window hid it as a repeat. Seen on the bench:
a scan stopped, then an autofocus run stopped 8 s later, and the second
"Protocol Stopped" was logged ``not shown to the user (suppressed: dedup)``.

Each test drives two runs of one kind inside the window, through the
production path on the simulated scope, and counts what was shown. Every
other post keeps its window: an ordinary post after an ending, inside it, is
held back, and one run's repeated failure is held to one popup by the run's
own latch.
"""

from __future__ import annotations

import pytest

import modules.autofocus_functions as autofocus_functions
import modules.manual_recording as manual_recording_module
from modules.activity_claim import ActivityClaim
from modules.exceptions import ProtocolNotSavedError, RunEndedByStopNotice
from modules.layer_record import release_colors
from modules.manual_recording import ManualRecordingController
from modules.notification_center import NotificationCenter, Severity
from modules.protocol import Protocol
from modules.scheduler import ThreadingTimerScheduler
from modules.scope_session import ScopeSession
from tests.scope_fakes import bind_settings_like_a_session, home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_a_run_reports_what_it_captured import _no_frame, _run, _two_steps
from tests.test_a_stop_press_ends_the_live_run import (
    _HOLD_WAIT_S,
    _steps,
    start_an_autofocus_held_mid_sweep,
)
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_composite_run_failures import _fail_these_channels
from tests.test_run_refusal_contract import _build_real_protocol
from tests.stock_channels import LED0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _shown(centre_posts, *, title=None, reason=None):
    return [
        n
        for n in centre_posts
        if n.shown
        and (title is None or n.title == title)
        and (reason is None or n.reason == reason)
    ]


def _stopped_mid_move(session, tmp_path):
    held = session.scope.simulation.hold_axis('X')
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"
    session.scope.motion.stop_motion()
    return run.result(timeout=COMPLETION_TIMEOUT)


def test_a_scan_stopped_then_an_autofocus_stopped_both_say_so(session, tmp_path, centre_posts):
    """The bench case: a scan stopped, then an autofocus run stopped seconds later."""
    scan = _stopped_mid_move(session, tmp_path / 'scan')
    session.scope.simulation.clear()
    autofocus = start_an_autofocus_held_mid_sweep(session)
    session.scope.motion.stop_motion()
    focused = autofocus.result(timeout=COMPLETION_TIMEOUT)

    assert (scan.reason, focused.reason) == ('stage_stopped', 'stage_stopped')
    assert len(_shown(centre_posts, title='Protocol Stopped')) == 2


def test_two_runs_short_of_captures_both_say_so(tmp_path, centre_posts):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope._camera_driver.grab_new_capture = _no_frame
        first = _run(runner, tmp_path / 'first', _two_steps())
        second = _run(runner, tmp_path / 'second', _two_steps())

    assert (first.reason, second.reason) == ('captures_failed', 'captures_failed')
    assert len(_shown(centre_posts, title='Run Incomplete')) == 2


def test_two_runs_that_could_not_start_both_say_so(session, tmp_path, centre_posts, monkeypatch):
    def _refused(self, file_path):
        # As Protocol.to_file answers a write the file system refused.
        raise ProtocolNotSavedError(file_path, OSError(28, 'No space left on device'))

    monkeypatch.setattr(Protocol, 'to_file', _refused)
    runner = session.create_protocol_runner()
    endings = [
        runner.start_single_scan(
            protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path / name)
        ).result(timeout=COMPLETION_TIMEOUT)
        for name in ('first', 'second')
    ]

    assert [(e.status, e.reason) for e in endings] == [
        ('failed_at_start', 'run_dir_init_failed')
    ] * 2
    assert len(_shown(centre_posts, reason='run_dir_init_failed')) == 2


def test_two_autofocus_runs_that_did_not_focus_both_say_so(session, centre_posts, monkeypatch):
    # Every frame scores zero: the sweep chooses no focus.
    monkeypatch.setattr(autofocus_functions, 'autofocus_score', lambda **kwargs: 0.0)
    # Mid-travel, so each sweep's window lies inside Z's travel.
    session.scope.motion.move_z(5000.0)
    runner = session.create_protocol_runner()
    endings = [runner.start_autofocus('BF').result(timeout=COMPLETION_TIMEOUT) for _ in range(2)]

    assert [(e.status, e.reason) for e in endings] == [('failed', 'autofocus_failed')] * 2
    assert len(_shown(centre_posts, title='Autofocus Failed')) == 2


def test_two_composites_missing_a_channel_both_say_so(tmp_path, centre_posts):
    """The merge's own failure, then the run's shortfall, for each run."""
    from modules.exceptions import CaptureError

    settings = headless_settings(tmp_path, acquiring=('BF', LED0))
    with open_composite_session(settings) as (session, runner):
        for name in ('first', 'second'):
            with pytest.raises(CaptureError):
                runner.run_composite(
                    sequence_name=name,
                    parent_dir=str(tmp_path),
                    events=_fail_these_channels(session.scope._camera_driver, ('BF', LED0), {LED0}),
                )

    assert len(_shown(centre_posts, title='Composite Failed')) == 2
    assert len(_shown(centre_posts, title='Run Incomplete')) == 2


def test_two_composites_whose_merge_raised_both_say_so(tmp_path, centre_posts, monkeypatch):
    """A raise from outside the root, told under the merge's "Composite Failed"."""
    from modules.composite_generation import CompositeGeneration
    from modules.exceptions import CaptureError

    def _explode(self, path, **kwargs):
        raise RuntimeError('synthetic merge crash')

    monkeypatch.setattr(CompositeGeneration, 'load_folder', _explode)
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        for name in ('first', 'second'):
            with pytest.raises(CaptureError):
                runner.run_composite(sequence_name=name, parent_dir=str(tmp_path))

    assert len(_shown(centre_posts, title='Composite Failed')) == 2


def _failed_mid_run(session, tmp_path, *, faults=1):
    held = session.scope.simulation.hold_axis('X')
    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol(_steps(60.0)), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"
    for _ in range(faults):
        session.sequenced_capture_runner.abort_run_fatal('camera_failure', 'The camera failed.')
    session.scope.motion.stop_motion()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)
    session.scope.simulation.clear()
    return outcome


def test_two_runs_that_failed_mid_run_both_say_so(session, tmp_path, centre_posts):
    endings = [_failed_mid_run(session, tmp_path / name) for name in ('first', 'second')]

    assert [(e.status, e.reason) for e in endings] == [('failed', 'camera_failure')] * 2
    assert len(_shown(centre_posts, reason='camera_failure')) == 2


def test_one_run_whose_fault_repeats_says_so_once(session, tmp_path, centre_posts):
    """The run's latch, not the window, holds one run's failure to one popup."""
    ending = _failed_mid_run(session, tmp_path, faults=5)

    assert (ending.status, ending.reason) == ('failed', 'camera_failure')
    assert len(_shown(centre_posts, reason='camera_failure')) == 1


def test_two_recordings_stopped_by_a_full_disk_both_say_so(
    sim_scope, tmp_path, centre_posts, monkeypatch, post_processing
):
    bind_settings_like_a_session(sim_scope, objective_id='4x Oly')
    scheduler = ThreadingTimerScheduler(name_prefix='TestRunEndingShown')
    controller = ManualRecordingController(
        scope=sim_scope,
        settings=complete_settings(
            live_folder=str(tmp_path),
            video_as_frames=True,
            video={'max_fps': 0, 'max_duration_s': 60, 'timestamp_overlay': False},
        ),
        shown_layer=lambda: 'BF',
        channel_colors=release_colors,
        activity_claim=ActivityClaim(),
        scheduler=scheduler,
        post_processing=post_processing,
    )
    # The start's own probe finds room; every probe while recording finds
    # the floor crossed, so each recording stops at its first.
    monkeypatch.setattr(
        manual_recording_module,
        'check_disk_space_ok',
        lambda *_: (not controller.is_recording, 100.0),
    )
    try:
        ends = [controller.start().result(timeout=COMPLETION_TIMEOUT) for _ in range(2)]
    finally:
        scheduler.shutdown()

    assert ends == ['disk_floor', 'disk_floor']
    assert len(_shown(centre_posts, title='Recording Stopped -- Disk Almost Full')) == 2


def test_an_ordinary_post_inside_the_window_after_an_ending_is_held_back():
    """A shown ending advances the window, as a solicited post does."""
    centre = NotificationCenter()
    ended = RunEndedByStopNotice()
    centre.report_outcome(ended, solicited=False, category='Protocol', ends_a_run=True)

    shown = centre.notice('Protocol', ended.title, 'A repeat that is not a run ending.')

    assert shown is False


def test_an_ending_is_still_muted_on_an_unattended_run():
    """The window is what an ending is exempt from; the unattended mute still holds."""
    centre = NotificationCenter()
    told = []
    centre.add_listener(told.append, min_severity=Severity.DEBUG)
    centre.open_run_scope(attended=False)

    centre.report_outcome(
        RunEndedByStopNotice(), solicited=False, category='Protocol', ends_a_run=True
    )

    assert [n.shown for n in told] == [False]
