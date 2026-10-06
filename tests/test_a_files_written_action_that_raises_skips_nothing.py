# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each thing a run does once its files are written happens, whatever raises.

When a run's last write lands, the run completes its record, sends
run_complete if cleanup never did, sends files_complete, and tells the
Session the drain is over. They ran in sequence with one catch around the
lot, so a raise in one skipped the rest -- a caller never told its files
were done, a Session that went on reading the drain as live -- and the raise
was only logged. Each is now contained on its own, and reported.
"""

import contextlib
import time

import modules.protocol_cleanup as protocol_cleanup
from modules.notification_center import notifications
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0


class _SendError(RuntimeError):
    pass


def test_a_raising_run_complete_send_still_tells_the_files_and_ends_the_drain(
    tmp_path, monkeypatch
):
    reported = []
    real_report = notifications.report_outcome

    def _report(ex, **kw):
        if isinstance(ex, _SendError):
            reported.append(ex)
            return None
        return real_report(ex, **kw)

    monkeypatch.setattr(notifications, 'report_outcome', _report)

    schedule = protocol_cleanup._schedule_cleanup_ui

    def _cannot_send_run_complete(func, step_label, *args, **kwargs):
        if step_label == 'Run-complete callback':
            raise _SendError('run_complete could not be sent')
        return schedule(func, step_label, *args, **kwargs)

    monkeypatch.setattr(protocol_cleanup, '_schedule_cleanup_ui', _cannot_send_run_complete)
    files = []
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        outcome = runner.run_single_scan(
            protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
            parent_dir=str(tmp_path / 'runs'),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            callbacks={
                'run_complete': lambda **kw: None,
                'files_complete': lambda **kw: files.append(kw['files']),
            },
        )
        # A run_complete that could not be sent is told all the same: the
        # wait does not run out its bound on a delivery that never comes.
        assert outcome.wait(timeout_s=WAIT_S) is not None
        deadline = time.monotonic() + WAIT_S
        while not files and time.monotonic() < deadline:
            time.sleep(0.02)

        assert files == ['written'], 'files_complete never came after run_complete raised'
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


def test_a_callback_that_fails_after_the_run_does_not_say_the_images_were_saved(monkeypatch):
    """The files are told on their own; a run whose save failed is incomplete,
    and a failed step after it must not tell the user its images were saved."""
    import threading

    shown = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda ex, *a, **k: shown.append(('Protocol', ex.title, str(ex))),
    )
    monkeypatch.setattr(protocol_cleanup, '_schedule_ui', lambda fn, *a, **k: fn(0))
    summary_sent = threading.Event()
    summary_sent.set()

    def _raising(dt):
        raise _SendError('files_complete fell over')

    protocol_cleanup._schedule_cleanup_ui(
        _raising, 'Files-complete callback', [], summary_sent, contextlib.nullcontext()
    )

    [(_category, _title, message)] = shown
    assert 'Files-complete callback' in message
    assert 'saved' not in message, message
