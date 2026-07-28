# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for protocol write-queue back-pressure (#740, commit 1).

The bounded file-IO queue used to DROP a capture (PROTOCOL_QUEUE_FULL) when
the single write worker fell behind -- already-grabbed frames silently never
reached disk. The contract is now back-pressure: the capture path BLOCKS
until a slot frees, pacing the run to disk drain, and a worker that stops
retiring tasks entirely is declared wedged -- a loud fatal abort naming the
stuck task instead of an unbounded silent wait.

All tests drive the production classes: a real SequentialIOExecutor with a
small bounded protocol queue and an event-gated wedge task (no sleeps for
synchronization; bounded waits only where the assertion IS "it blocks").
"""

import pathlib
import threading

import numpy as np
import pytest

import modules.protocol_image_writer as piw
from modules.notification_center import Severity, notifications
from modules.sequential_io_executor import (
    IOTask,
    PROTOCOL_ENQUEUED,
    SequentialIOExecutor,
)
from tests.test_audit_fixes import _bare_protocol_writer, _protocol_step


@pytest.fixture
def bounded_executor():
    """Started executor with a 2-slot bounded protocol queue, in session."""
    ex = SequentialIOExecutor(name='TEST_BP', protocol_queue_maxsize=2)
    ex.start()
    ex.protocol_start()
    yield ex
    ex.shutdown(wait=True)


def _park_worker(ex):
    """Occupy the worker with an event-gated wedge task; returns the release
    event once the worker is provably parked (so queue fills are race-free)."""
    started = threading.Event()
    release = threading.Event()

    def _wedge():
        started.set()
        release.wait(timeout=60)

    assert ex.protocol_put(IOTask(action=_wedge)) is PROTOCOL_ENQUEUED
    assert started.wait(2), 'worker never picked up the wedge task'
    return release


def test_backpressure_blocks_instead_of_dropping(bounded_executor):
    """A submit against a full queue BLOCKS until a slot frees -- no
    PROTOCOL_QUEUE_FULL, no drop row -- and the task then runs.
    Pre-fix: protocol_put returned PROTOCOL_QUEUE_FULL immediately."""
    ex = bounded_executor
    release = _park_worker(ex)
    assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED
    assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED

    ran = threading.Event()
    submitted = threading.Event()
    results = []

    def _submit():
        results.append(
            ex.protocol_put_wait(
                IOTask(action=ran.set),
                should_abort=lambda: False,
                stall_timeout_s=30.0,
            )
        )
        submitted.set()

    t = threading.Thread(target=_submit, daemon=True)
    t.start()
    assert not submitted.wait(0.6), 'submit returned against a full queue instead of blocking'

    release.set()
    assert submitted.wait(5), 'submit never completed after the queue drained'
    assert results == [PROTOCOL_ENQUEUED]
    assert ran.wait(5), 'the blocked-then-enqueued task never ran'
    assert ex.protocol_dropped_count() == 0


def test_writer_capture_paces_to_full_queue_without_drop_row(tmp_path, monkeypatch):
    """ProtocolImageWriter.capture against a full real queue: blocks, then
    returns True with the write executed and NO capture_failed_queue_full
    row. Pre-fix: recorded the drop row and returned False immediately."""
    from unittest.mock import MagicMock

    ex = SequentialIOExecutor(name='TEST_BP_WRITER', protocol_queue_maxsize=2)
    ex.start()
    ex.protocol_start()
    saved = threading.Event()
    monkeypatch.setattr(
        piw,
        'save_image',
        lambda *a, **k: saved.set() or pathlib.Path(tmp_path / 'frame.tiff'),
    )
    record = MagicMock()
    writer = _bare_protocol_writer(file_io_executor=ex, execution_record=record)
    scope = writer._scope
    scope.motion.has_turret.return_value = False
    scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
    scope.imaging.capture_frame_depth.return_value = 8
    protocol = MagicMock()
    protocol.capture_root.return_value = ''

    try:
        release = _park_worker(ex)
        assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED
        assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED

        done = threading.Event()
        results = []

        def _capture():
            results.append(
                writer.capture(
                    save_folder=tmp_path,
                    step=_protocol_step(),
                    output_format='TIFF',
                    protocol=protocol,
                )
            )
            done.set()

        t = threading.Thread(target=_capture, daemon=True)
        t.start()
        assert not done.wait(0.6), 'capture returned against a full queue instead of pacing'

        release.set()
        assert done.wait(10), 'capture never completed after the queue drained'
        assert results == [True]
        assert saved.wait(10), 'the paced write never reached save_image'
        dropped_rows = [
            c
            for c in record.add_step.call_args_list
            if c.kwargs.get('capture_result_file_name') == 'capture_failed_queue_full'
        ]
        assert dropped_rows == [], 'a paced write must not be recorded as a queue-full drop'
        assert ex.protocol_dropped_count() == 0
    finally:
        ex.shutdown(wait=True)


def test_blocked_submit_honors_abort_promptly():
    """An abort signalled while a submit is blocked unblocks it within one
    poll interval; the return is the cancelled outcome (None), not a wedge."""
    ex = SequentialIOExecutor(name='TEST_BP_ABORT', protocol_queue_maxsize=1)
    ex.start()
    ex.protocol_start()
    try:
        release = _park_worker(ex)
        assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED

        aborted = threading.Event()
        submitted = threading.Event()
        results = []

        def _submit():
            results.append(
                ex.protocol_put_wait(
                    IOTask(action=lambda: None),
                    should_abort=aborted.is_set,
                    stall_timeout_s=30.0,
                )
            )
            submitted.set()

        t = threading.Thread(target=_submit, daemon=True)
        t.start()
        assert not submitted.wait(0.6), 'submit returned before any abort was signalled'

        aborted.set()
        assert submitted.wait(1.0), 'abort did not unblock the waiting submit within a poll'
        assert results == [None]
        assert ex.protocol_dropped_count() == 0
        release.set()
    finally:
        ex.shutdown(wait=True)


def test_wedged_writer_declares_stall_notifies_and_aborts(tmp_path, monkeypatch):
    """A worker that never retires anything past the stall budget: capture
    returns False, a fatal 'File Writer Stalled' notification names the
    stuck task, the lost capture is recorded as writer_stalled, and the run
    is aborted."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(piw, '_WRITE_STALL_FATAL_S', 0.4)

    ex = SequentialIOExecutor(name='TEST_BP_WEDGE', protocol_queue_maxsize=1)
    ex.start()
    ex.protocol_start()
    aborts = []
    record = MagicMock()
    writer = _bare_protocol_writer(
        file_io_executor=ex,
        execution_record=record,
        abort_fn=lambda: aborts.append(1),
    )
    scope = writer._scope
    scope.motion.has_turret.return_value = False
    scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
    scope.imaging.capture_frame_depth.return_value = 8
    protocol = MagicMock()
    protocol.capture_root.return_value = ''

    fired = []
    # remove_listener unregisters by identity, so the exact same callable
    # object must be handed to both calls.
    listener = fired.append
    notifications.add_listener(listener, min_severity=Severity.CRITICAL)
    try:
        release = _park_worker(ex)
        assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED

        result = writer.capture(
            save_folder=tmp_path,
            step=_protocol_step(),
            output_format='TIFF',
            protocol=protocol,
        )

        assert result is False, 'a wedged writer must fail the capture, not fake success'
        assert aborts == [1], 'a wedged writer must abort the run'
        stall_notes = [n for n in fired if n.title == 'File Writer Stalled']
        assert len(stall_notes) == 1, f'expected one fatal stall notification, saw {fired}'
        assert stall_notes[0].fatal, 'the stall popup must be fatal (mid-run popup class)'
        stalled_rows = [
            c
            for c in record.add_step.call_args_list
            if c.kwargs.get('capture_result_file_name') == 'writer_stalled'
        ]
        assert len(stalled_rows) == 1, 'the lost capture must be recorded as writer_stalled'
        release.set()
    finally:
        notifications.remove_listener(listener)
        ex.shutdown(wait=True)
