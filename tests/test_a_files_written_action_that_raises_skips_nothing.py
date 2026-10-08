# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each thing a run does once its files are written happens, whatever raises.

When a run's last write lands, the run completes its record, sends
files_written, and tells the Session the drain is over; run_ended goes out
just before. They ran in sequence with one catch around the lot, so a raise
in one skipped the rest -- a caller never told its files were done, a
Session that went on reading the drain as live -- and the raise was only
logged. Each is now contained on its own, and reported.
"""

import time

import modules.protocol_cleanup as protocol_cleanup
import modules.run_events as run_events
from modules.notification_center import notifications
from modules.run_events import RunEvents
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0


class _SendError(RuntimeError):
    pass


def test_a_raising_run_ended_send_still_tells_the_files_and_ends_the_drain(tmp_path, monkeypatch):
    reported = []
    real_report = notifications.report_outcome

    def _report(ex, **kw):
        if isinstance(ex, _SendError):
            reported.append(ex)
            return None
        return real_report(ex, **kw)

    monkeypatch.setattr(notifications, 'report_outcome', _report)

    deliver = protocol_cleanup.deliver

    def _cannot_send_run_ended(handler, event, *args, **kwargs):
        if event != 'run_ended':
            return deliver(handler, event, *args, **kwargs)

        def _refused(func, timeout=0):
            raise _SendError('run_ended could not be sent')

        # The dispatcher refuses this one delivery; deliver's own path for a
        # refusal is what is under test.
        schedule = run_events.schedule_ui
        run_events.schedule_ui = _refused
        try:
            return deliver(handler, event, *args, **kwargs)
        finally:
            run_events.schedule_ui = schedule

    monkeypatch.setattr(protocol_cleanup, 'deliver', _cannot_send_run_ended)
    files = []
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        outcome = runner.run_single_scan(
            protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
            parent_dir=str(tmp_path / 'runs'),
            events=RunEvents(
                run_ended=lambda outcome, run_dir, protocol: None,
                files_written=lambda run_dir, written: files.append(written),
            ),
        )
        # A run_ended that could not be sent is told all the same: the
        # wait does not run out its bound on a delivery that never comes.
        assert outcome.wait(timeout_s=WAIT_S) is not None
        deadline = time.monotonic() + WAIT_S
        while not files and time.monotonic() < deadline:
            time.sleep(0.02)

        assert files == ['written'], 'files_written never came after run_ended raised'
        assert not session.protocol_files_draining
    assert reported, 'the raise was not reported'


def test_the_batch_reports_a_completion_action_that_raises_and_still_completes(monkeypatch):
    from unittest.mock import MagicMock

    from modules.protocol_image_writer import RunWriteBatch

    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda ex, **kw: reported.append(ex))
    batch = RunWriteBatch(MagicMock())

    def _raising(outcome):
        raise _SendError('the completion action fell over')

    batch.close()
    batch.when_complete(_raising)

    assert batch.wait_complete(0.1)
    assert not batch.draining
    [ex] = reported
    assert isinstance(ex, _SendError)


def test_a_files_written_handler_that_fails_is_reported_as_itself(monkeypatch):
    """The files are told on their own; a handler that fails is reported once,
    as the exception it is, under its event's name -- never as a cleanup
    failure that says anything of the run's images."""
    import pathlib

    reported = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda ex, *a, **k: reported.append((ex, k)),
    )
    failure = _SendError('files_written fell over')

    def _raising(run_dir, files):
        raise failure

    run_events.deliver(_raising, 'files_written', pathlib.Path('run'), 'written')

    [(ex, kwargs)] = reported
    assert ex is failure
    assert kwargs == {'solicited': False, 'category': 'files_written'}
