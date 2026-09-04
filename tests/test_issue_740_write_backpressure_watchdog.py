# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for protocol write-queue back-pressure + wedge watchdog (#740).

The bounded file-IO queue used to DROP a capture (PROTOCOL_QUEUE_FULL) when
the single write worker fell behind -- already-grabbed frames silently never
reached disk. The contract is now back-pressure: the capture path BLOCKS
until a slot frees, pacing the run to disk drain, and a worker that stops
retiring tasks entirely is declared wedged -- a loud fatal abort naming the
stuck task instead of an unbounded silent wait. The post-run half: a wedged
writer used to hold every "please wait - files are still being written" gate
closed forever; drain-stall detection now distinguishes wedged from slow,
and recovery discards pending writes and replaces the stuck worker.

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
    scope.capabilities.has_turret = False
    scope.imaging._capture_and_wait_impl.return_value = np.zeros((4, 4), dtype=np.uint8)
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

    monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', 0.4)

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
    scope.capabilities.has_turret = False
    scope.imaging._capture_and_wait_impl.return_value = np.zeros((4, 4), dtype=np.uint8)
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


def test_drain_stalled_and_recovery_unlocks_and_replaces_worker():
    """Post-run lockout half: a wedged worker keeps is_protocol_queue_active
    True forever. protocol_drain_stalled distinguishes that wedge from a
    healthy drain; recover_wedged_protocol_queue drains pending writes,
    clears the lockout gate, releases the deferred protocol-complete
    callback, and a replacement worker serves fresh tasks. The abandoned
    worker's late completion must touch no executor state."""
    import time as _time

    ex = SequentialIOExecutor(name='TEST_BP_RECOVER', protocol_queue_maxsize=4)
    ex.start()
    ex.protocol_start()
    fired = []
    listener = fired.append
    notifications.add_listener(listener, min_severity=Severity.WARNING)
    started = threading.Event()
    release = threading.Event()
    try:

        def _wedge_then_raise():
            started.set()
            release.wait(timeout=60)
            raise RuntimeError('stuck write finally failed')

        assert ex.protocol_put(IOTask(action=_wedge_then_raise)) is PROTOCOL_ENQUEUED
        assert started.wait(2), 'worker never picked up the wedge task'
        assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED

        # Post-run shape: cleanup deferred files_complete and signalled drain.
        deferred = threading.Event()
        ex.set_protocol_complete_callback(deferred.set)
        ex.protocol_finish_then_end()

        assert not ex.protocol_drain_stalled(3600.0), (
            'an in-flight task under its stall threshold must read as draining, not wedged'
        )
        deadline = _time.monotonic() + 2.0
        while not ex.protocol_drain_stalled(0.05) and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert ex.protocol_drain_stalled(0.05), 'the aging in-flight task never read as stalled'
        assert ex.is_protocol_queue_active(), 'precondition: the lockout gate is held'

        orphan_thread = ex._worker_thread
        ex.recover_wedged_protocol_queue()

        assert not ex.is_protocol_queue_active(), 'recovery must clear the lockout gate'
        assert deferred.wait(2), 'recovery must release the deferred protocol-complete callback'

        ran = threading.Event()
        ex.put(IOTask(action=ran.set))
        assert ran.wait(2), 'replacement worker must serve normal-queue tasks'

        # Let the abandoned worker finish (and raise); its guarded epilogue
        # must not fire a stale task-failure popup or clobber the
        # replacement's state.
        release.set()
        orphan_thread.join(2)
        assert not orphan_thread.is_alive(), 'abandoned worker must exit after its stuck call'
        assert not any('task failed' in n.title for n in fired), (
            f'abandoned worker fired a stale task-failure popup: {fired}'
        )
        assert not ex.is_protocol_queue_active(), (
            'abandoned worker clobbered executor state after recovery'
        )
    finally:
        release.set()
        notifications.remove_listener(listener)
        ex.shutdown(wait=True)


def test_backpressure_blocked_time_accumulates_and_resets_per_run():
    """The demand-relative slow-disk signal: time spent blocked waiting for
    a write slot accumulates across a run and resets at protocol_start."""
    ex = SequentialIOExecutor(name='TEST_BP_SLOW', protocol_queue_maxsize=1)
    ex.start()
    ex.protocol_start()
    try:
        release = _park_worker(ex)
        assert ex.protocol_put(IOTask(action=lambda: None)) is PROTOCOL_ENQUEUED

        aborted = threading.Event()
        submitted = threading.Event()

        def _submit():
            ex.protocol_put_wait(
                IOTask(action=lambda: None),
                should_abort=aborted.is_set,
                stall_timeout_s=30.0,
            )
            submitted.set()

        t = threading.Thread(target=_submit, daemon=True)
        t.start()
        assert not submitted.wait(0.7), 'submit returned against a full queue instead of blocking'
        aborted.set()
        assert submitted.wait(1.0)

        assert ex.protocol_backpressure_blocked_s() >= 0.5, (
            'blocked-enqueue time must accumulate while waiting for a slot'
        )
        release.set()
        ex.protocol_start()
        assert ex.protocol_backpressure_blocked_s() == 0.0, (
            'a fresh run must not inherit the previous run blocked-wait total'
        )
    finally:
        ex.shutdown(wait=True)


def test_run_cleanup_surfaces_slow_write_warning(monkeypatch):
    """Run-end summary fires the sustained-slow-write warning when the run's
    blocked-wait total crossed the threshold, and stays silent otherwise."""
    from modules.notification_center import notifications as nc_notifications
    from modules.protocol_cleanup import run_cleanup
    from tests.test_audit_fixes import _run_cleanup_kwargs

    captured = []
    monkeypatch.setattr(nc_notifications, 'warning', lambda *a, **k: captured.append(a))

    kwargs = _run_cleanup_kwargs()
    kwargs['file_io_executor'].protocol_backpressure_blocked_s.return_value = 45.0
    run_cleanup(**kwargs)
    assert any(a[1] == 'Very Slow File Writes' for a in captured), (
        f'45s of blocked writes must surface the slow-write warning; saw {captured}'
    )

    captured.clear()
    run_cleanup(**_run_cleanup_kwargs())
    assert not any(a[1] == 'Very Slow File Writes' for a in captured), (
        'a run with no blocked-wait must not warn about slow writes'
    )


def test_prepare_refusal_names_stalled_writer(tmp_path):
    """The runner's pre-run gate distinguishes a wedged writer (recover,
    retrying is useless) from a healthy drain (wait and retry)."""
    from unittest.mock import MagicMock

    import pytest as _pytest

    from modules.exceptions import ProtocolRunRefusedError
    from modules.image_mode import ImageCaptureConfig
    from modules.sequenced_capture_runner import SequencedCaptureRunMode
    from tests.protocol_drives import autofocus_snapshot
    from tests.test_audit_fixes import _make_capture_runner

    runner = _make_capture_runner()
    runner.file_io_executor.is_protocol_queue_active.return_value = True
    runner.file_io_executor.protocol_drain_stalled.return_value = True
    runner.file_io_executor.describe_running_task.return_value = (
        "write_capture 'B2_BF' 45s in flight"
    )

    def _prepare():
        runner.prepare(
            protocol=MagicMock(),
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='t',
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            autogain_settings={},
            parent_dir=tmp_path,
            autofocus_snapshot=autofocus_snapshot(),
        )

    with _pytest.raises(ProtocolRunRefusedError) as excinfo:
        _prepare()
    assert excinfo.value.reason == 'files_writing_stalled'
    assert 'B2_BF' in excinfo.value.message, 'the refusal must name the stuck write'

    runner.file_io_executor.protocol_drain_stalled.return_value = False
    with _pytest.raises(ProtocolRunRefusedError) as excinfo:
        _prepare()
    assert excinfo.value.reason == 'files_writing'


def test_session_recover_file_writer_passthrough():
    """L2 parity: a Session holding an executor bundle can recover a
    wedged writer; a session with neither bundle nor file-io handle
    reports False."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from modules.scope_session import ScopeSession

    bundle = SimpleNamespace(file_io_executor=MagicMock(), protocol_thread=MagicMock())
    session = ScopeSession(
        settings={},
        scope=MagicMock(),
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
        executor_bundle=bundle,
    )
    assert session.recover_file_writer() is True
    bundle.file_io_executor.recover_wedged_protocol_queue.assert_called_once()

    gui_hosted = ScopeSession(
        settings={},
        scope=MagicMock(),
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
    )
    assert gui_hosted.recover_file_writer() is False


def test_blank_labware_has_no_wells_and_fabricates_no_index():
    """The zero-well Blank plate used to clip every position to well index
    (-1, -1) -- rendered as label '@0' in filenames/metadata and a bogus
    selected-well ring at plate origin. A zero-well plate now has no well
    index at all: has_wells() gates consumers, labels are empty (omitted
    downstream), and asking for an index is an error, not a fabrication."""
    import pytest as _pytest

    from modules.labware_loader import WellPlateLoader

    loader = WellPlateLoader()
    blank = loader.get_plate('Blank')

    assert blank.has_wells() is False
    assert blank.get_well_label(x=10.0, y=10.0) == ''
    with _pytest.raises(ValueError):
        blank.get_well_index(10.0, 10.0)
    assert blank.get_positions_with_labels() == []

    # A real plate is unaffected.
    plate = loader.get_plate('6 well microplate')
    assert plate.has_wells() is True
    i, j = plate.get_well_index(*plate.get_well_position(0, 0))
    assert (i, j) == (0, 0)
    assert plate.get_well_label(*plate.get_well_position(0, 0)) == 'A1'
