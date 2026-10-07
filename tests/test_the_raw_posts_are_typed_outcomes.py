# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The notices, run endings and autofocus faults reach a subscriber typed.

Before, each was a direct post to the notification centre: a subscriber got
it as ``unclassified``, with no reason code, a level the poster chose, and
-- for the two run endings that must reach an unattended user -- a ``fatal``
flag passed by hand. Now each is an exception type reported once through
``report_outcome``, which reads its kind, title, words, reason and fatality
from the type. The guard that no direct post remains is
``tests/guards/test_only_the_reporter_posts_below_the_gui.py``.
"""

from __future__ import annotations

import datetime

import pytest

import modules.exceptions as exc
import modules.notification_center as nc
from modules.notification_center import NotificationCenter, OutcomeKind, Severity
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def centre(monkeypatch):
    # A centre of its own: the shared one's dedup window remembers what
    # earlier tests posted.
    centre = NotificationCenter()
    monkeypatch.setattr(nc, 'notifications', centre)
    return centre


def _heard(centre):
    heard = []
    centre.add_listener(heard.append, min_severity=Severity.DEBUG)
    return heard


_NOTICES = [
    (exc.ExposureAtMaximumNotice(200.0), 'exposure_at_maximum'),
    (exc.ExposureAtMinimumNotice(0.01, 0.05), 'exposure_at_minimum'),
    (exc.CapturePositionNotRecordedNotice(), 'position_not_recorded'),
    (
        exc.RecordingPositionNotRecordedNotice(unknown_axes=['X', 'Z'], has_plate=False),
        'position_not_recorded',
    ),
    (
        exc.DuplicateCaptureFilenamesNotice(colliding_steps=4, shared_names=2),
        'duplicate_capture_filenames',
    ),
    (exc.SlowFileWritesNotice(), 'slow_file_writes'),
    (
        exc.ProtocolStepsInvalidNotice(
            errors=['Step 1 (a): Exposure must be more than 0 ms, got 0.0']
        ),
        'protocol_steps_invalid',
    ),
    (
        exc.SingleScanNotice(
            period=datetime.timedelta(minutes=10), duration=datetime.timedelta(minutes=5)
        ),
        'single_scan',
    ),
    (exc.HyperstacksSavingNotice(), 'hyperstacks_saving'),
    (
        exc.HyperstacksSavedNotice(
            {'new_count': 3, 'output_root': '/data/run', 'accounting_note': ''}
        ),
        'hyperstacks_saved',
    ),
]

_FAULTS = [
    (exc.AutoGainNotSettledError(), 'auto_gain_not_settled', False),
    (exc.AutofocusFailedError('flat_focus_curve'), 'flat_focus_curve', False),
    (exc.AutofocusFailedError('unexpected_error'), 'unexpected_error', False),
    (exc.AutofocusZNotRestoredError(z_lost=True), 'z_position_lost', False),
    (exc.AutofocusZNotRestoredError(z_lost=False), 'z_left_at_search_position', False),
    (
        exc.CompositeFailedError('The run has no directory to merge from.', 'no_run_dir'),
        'no_run_dir',
        False,
    ),
    (
        exc.RunFailedToStartError(reason='start_failed', title='Run failed to start', message='m'),
        'start_failed',
        False,
    ),
    (
        exc.RunFailedError(
            reason='motion_timeout', title='Protocol Error -- Motion Timeout', message='m'
        ),
        'motion_timeout',
        True,
    ),
]


@pytest.mark.parametrize(('notice', 'reason'), _NOTICES, ids=lambda v: type(v).__name__)
def test_a_notice_arrives_as_a_notice_with_its_reason_and_is_shown_once(centre, notice, reason):
    heard = _heard(centre)

    centre.report_outcome(notice, solicited=False, category='Test')
    centre.report_outcome(notice, solicited=False, category='Test')

    assert [(n.kind, n.severity, n.title, n.message, n.reason, n.shown) for n in heard] == [
        (OutcomeKind.NOTICE, Severity.NOTICE, notice.title, str(notice), reason, True)
    ]


@pytest.mark.parametrize(('fault', 'reason', 'fatal'), _FAULTS, ids=lambda v: type(v).__name__)
def test_a_fault_arrives_as_a_fault_in_its_own_words_with_its_reason(centre, fault, reason, fatal):
    heard = _heard(centre)

    centre.report_outcome(fault, solicited=False, category='Test')

    assert [(n.kind, n.title, n.message, n.reason, n.fatal) for n in heard] == [
        (OutcomeKind.FAULT, fault.title, str(fault), reason, fatal)
    ]
    assert heard[0].severity is (Severity.CRITICAL if fatal else Severity.ERROR)


def test_a_failed_run_is_shown_through_an_unattended_runs_mute_and_a_failed_start_is_not(centre):
    heard = _heard(centre)
    centre.open_run_scope(attended=False)

    centre.report_outcome(
        exc.RunFailedError(reason='disk_space_critical', title='Disk Space Critical', message='m'),
        solicited=False,
        category='FileIO',
    )
    centre.report_outcome(
        exc.RunFailedToStartError(reason='start_failed', title='Run failed to start', message='m'),
        solicited=False,
        category='Protocol',
    )

    assert [(n.title, n.shown) for n in heard] == [
        ('Disk Space Critical', True),
        ('Run failed to start', False),
    ]


class TestTheAutoGainLimitsAreShown:
    @pytest.fixture
    def session(self, tmp_path, centre, monkeypatch):
        # Imaging binds the centre when it is imported.
        monkeypatch.setattr('modules.lumascope_api.imaging.notifications', centre)
        s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            yield s
        finally:
            s.shutdown()

    def _lock(self, state, *, resume_after_capture=True):
        from modules.lumascope_api.imaging import AutoGainLock

        return AutoGainLock(
            state=state,
            exposure_ms=0.01,
            gain_db=0.0,
            floor_ms=0.05,
            ceiling_ms=200.0,
            resume_after_capture=resume_after_capture,
        )

    def test_a_live_view_lock_at_either_limit_is_a_shown_notice(self, session):
        from modules.lumascope_api.imaging import AutoGainConvergence

        heard = []
        session.add_outcome_listener(heard.append)
        imaging = session.scope.imaging

        imaging._notify_auto_gain_outcome(self._lock(AutoGainConvergence.MAXED))
        imaging._notify_auto_gain_outcome(self._lock(AutoGainConvergence.AT_MINIMUM))
        imaging._notify_auto_gain_outcome(self._lock(AutoGainConvergence.FAILED))

        assert [(n.kind, n.title, n.shown) for n in heard] == [
            (OutcomeKind.NOTICE, 'Exposure at the maximum', True),
            (OutcomeKind.NOTICE, 'Exposure at the minimum', True),
            (OutcomeKind.FAULT, 'Auto-gain did not settle', True),
        ]

    def test_a_protocol_steps_lock_tells_no_one(self, session):
        from modules.lumascope_api.imaging import AutoGainConvergence

        heard = []
        session.add_outcome_listener(heard.append)
        imaging = session.scope.imaging

        imaging._notify_auto_gain_outcome(
            self._lock(AutoGainConvergence.MAXED, resume_after_capture=False)
        )

        assert heard == []


def test_an_exposure_exactly_at_the_floor_is_not_called_below_it():
    # The lock calls an exposure at the floor AT_MINIMUM (<=); on the
    # simulator a fluorescence channel at its 1 ms slider minimum is
    # exactly the 1 ms floor, and the words must not say "below" it.
    words = str(exc.ExposureAtMinimumNotice(1.0, 1.0))

    assert 'settled at 1 ms, at or below the 1 ms usable floor' in words
