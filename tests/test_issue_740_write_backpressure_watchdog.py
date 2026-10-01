# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for protocol write back-pressure + wedge watchdog (#740).

The bounded file-IO queue used to DROP a capture (PROTOCOL_QUEUE_FULL) when
the single write worker fell behind -- already-grabbed frames silently never
reached disk. The contract is now back-pressure: the capture path BLOCKS
until the run's write backlog has room, pacing the run to disk drain, and a
worker that stops retiring writes entirely is declared wedged -- a loud
fatal abort naming the stuck write instead of an unbounded silent wait. The
post-run half: a wedged writer used to hold every "please wait - files are
still being written" gate closed forever; the run's write batch
distinguishes wedged from slow, and recovery gives up on the run's
outstanding writes, counting them, and replaces the stuck worker -- it
clears no queue.

All tests drive the production classes: a real SequentialIOExecutor, the
run's RunWriteBatch with its backlog bound shrunk, and an event-gated wedge
task (no sleeps for synchronization; bounded waits only where the assertion
IS "it blocks").
"""

import pathlib
import threading

import numpy as np
import pytest

import modules.protocol_image_writer as piw
from modules.notification_center import Severity, notifications
from modules.protocol_image_writer import RunWriteBatch
from modules.sequential_io_executor import (
    ENQUEUED,
    IOTask,
    SequentialIOExecutor,
)
from tests.test_audit_fixes import _bare_protocol_writer, _protocol_step


@pytest.fixture
def file_lane():
    """A started file lane."""
    ex = SequentialIOExecutor(name='TEST_BP')
    ex.start()
    yield ex
    ex.shutdown(wait=True)


def _park_worker(ex):
    """Occupy the worker with an event-gated wedge task; returns the release
    event once the worker is provably parked (so backlog fills are race-free)."""
    started = threading.Event()
    release = threading.Event()

    def _wedge():
        started.set()
        release.wait(timeout=60)

    assert ex.put(IOTask(action=_wedge)) is ENQUEUED
    assert started.wait(2), 'worker never picked up the wedge task'
    return release


def _fill(batch, count):
    """Take ``count`` places in the run's backlog with writes queued behind
    the parked worker."""
    for _ in range(count):
        assert batch.submit(lambda: None, {}, what='A filler', pace_until=None) is ENQUEUED


def test_backpressure_blocks_instead_of_dropping(file_lane, monkeypatch):
    """A submit against a full backlog BLOCKS until a write lands -- no
    drop -- and the write then runs.
    Pre-fix: protocol_put returned PROTOCOL_QUEUE_FULL immediately."""
    monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 2)
    ex = file_lane
    batch = RunWriteBatch(ex)
    release = _park_worker(ex)
    _fill(batch, 2)

    ran = threading.Event()
    submitted = threading.Event()
    results = []

    def _submit():
        results.append(batch.submit(ran.set, {}, what='The image', pace_until=lambda: False))
        submitted.set()

    t = threading.Thread(target=_submit, daemon=True)
    t.start()
    assert not submitted.wait(0.6), 'submit returned against a full backlog instead of blocking'

    release.set()
    assert submitted.wait(5), 'submit never completed after the backlog drained'
    assert results == [ENQUEUED]
    assert ran.wait(5), 'the blocked-then-enqueued write never ran'


def test_writer_capture_paces_to_full_queue_without_drop_row(tmp_path, monkeypatch):
    """ProtocolImageWriter.capture against a full real backlog: blocks, then
    returns True with the write executed and NO capture_failed_queue_full
    row. Pre-fix: recorded the drop row and returned False immediately."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 2)
    ex = SequentialIOExecutor(name='TEST_BP_WRITER')
    ex.start()
    saved = threading.Event()
    monkeypatch.setattr(
        piw,
        'save_image',
        lambda *a, **k: saved.set() or pathlib.Path(tmp_path / 'frame.tiff'),
    )
    record = MagicMock()
    writer = _bare_protocol_writer(write_batch=RunWriteBatch(ex), execution_record=record)
    scope = writer._scope
    # The objective the frame is taken with, read at capture.
    scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
    scope.capabilities.has_turret = False
    scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
    scope.imaging.capture_frame_depth.return_value = 8
    protocol = MagicMock()
    protocol.capture_root.return_value = ''

    try:
        release = _park_worker(ex)
        _fill(writer._write_batch, 2)

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
        assert not done.wait(0.6), 'capture returned against a full backlog instead of pacing'

        release.set()
        assert done.wait(10), 'capture never completed after the backlog drained'
        assert results == [True]
        assert saved.wait(10), 'the paced write never reached save_image'
        dropped_rows = [
            c
            for c in record.add_step.call_args_list
            if c.kwargs.get('capture_result_file_name') == 'capture_failed_queue_full'
        ]
        assert dropped_rows == [], 'a paced write must not be recorded as a queue-full drop'
    finally:
        ex.shutdown(wait=True)


def test_blocked_submit_honors_abort_promptly(file_lane, monkeypatch):
    """An abort signalled while a submit is blocked unblocks it within one
    poll interval, and the frame is handed over -- never dropped, and not a
    wedge -- so it is written once the backlog drains."""
    monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
    ex = file_lane
    batch = RunWriteBatch(ex)
    release = _park_worker(ex)
    _fill(batch, 1)

    aborted = threading.Event()
    submitted = threading.Event()
    ran = threading.Event()
    results = []

    def _submit():
        results.append(batch.submit(ran.set, {}, what='The image', pace_until=aborted.is_set))
        submitted.set()

    t = threading.Thread(target=_submit, daemon=True)
    t.start()
    try:
        assert not submitted.wait(0.6), 'submit returned before any abort was signalled'

        aborted.set()
        assert submitted.wait(1.0), 'abort did not unblock the waiting submit within a poll'
        assert results == [ENQUEUED], 'an aborted run must hand its captured frame over'
    finally:
        release.set()
    assert ran.wait(5), 'the frame handed over at the abort was never written'


def test_wedged_writer_declares_stall_notifies_and_aborts(tmp_path, monkeypatch):
    """A worker that never retires anything past the stall budget: capture
    returns False, a fatal 'File Writer Stalled' notification names the
    stuck task, the lost capture is recorded as writer_stalled, and the run
    is aborted."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', 0.4)
    monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)

    ex = SequentialIOExecutor(name='TEST_BP_WEDGE')
    ex.start()
    aborts = []
    record = MagicMock()
    writer = _bare_protocol_writer(
        write_batch=RunWriteBatch(ex),
        execution_record=record,
        abort_fn=lambda: aborts.append(1),
    )
    scope = writer._scope
    # The objective the frame is taken with, read at capture.
    scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
    scope.capabilities.has_turret = False
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
        _fill(writer._write_batch, 1)

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


def test_recovery_abandons_the_stuck_writes_and_replaces_the_worker():
    """Post-run lockout half: a wedged worker keeps the finished run's
    files draining forever. The batch's stall answer distinguishes that
    wedge from a healthy drain; recovery gives up on the run's outstanding
    writes, counting them, which completes the batch as abandoned and
    clears the lockout; the stuck worker is replaced, and nothing queued
    on the lane is discarded -- the replacement serves it. The abandoned
    worker's late completion counts nothing and touches no executor
    state."""
    import time as _time

    ex = SequentialIOExecutor(name='TEST_BP_RECOVER')
    ex.start()
    batch = RunWriteBatch(ex)
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

        assert (
            batch.submit(_wedge_then_raise, {}, what='The stuck image', pace_until=None) is ENQUEUED
        )
        assert started.wait(2), 'worker never picked up the wedge task'
        queued_ran = threading.Event()
        assert ex.put(IOTask(action=queued_ran.set)) is ENQUEUED

        # Post-run shape: the run's cleanup closed its writes.
        outcomes = []
        batch.close(outcomes.append)

        assert not batch.stalled(3600.0), (
            'an in-flight write under its stall threshold must read as draining, not wedged'
        )
        deadline = _time.monotonic() + 2.0
        while not batch.stalled(0.05) and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert batch.stalled(0.05), 'the aging in-flight write never read as stalled'
        assert batch.draining, 'precondition: the lockout gate is held'

        orphan_thread = ex._worker_thread
        abandoned = batch.abandon('File writer recovery')
        ex.replace_stuck_worker()

        assert abandoned == 1, 'recovery must count the write it gave up on'
        assert not batch.draining, 'recovery must clear the lockout gate'
        assert outcomes == ['incomplete'], "the run's files must end incomplete, never written"
        assert queued_ran.wait(2), 'recovery must not discard work queued behind the stuck write'

        ran = threading.Event()
        ex.put(IOTask(action=ran.set))
        assert ran.wait(2), 'replacement worker must serve normal-queue tasks'

        # Let the abandoned worker finish (and raise); its guarded epilogue
        # must not fire a stale task-failure popup or clobber the
        # replacement's state, and its write must not count again.
        release.set()
        orphan_thread.join(2)
        assert not orphan_thread.is_alive(), 'abandoned worker must exit after its stuck call'
        assert not any('task failed' in n.title for n in fired), (
            f'abandoned worker fired a stale task-failure popup: {fired}'
        )
        assert batch.pending == 0 and batch.outcome == 'incomplete', (
            'the abandoned write counted again when its stuck call returned'
        )
        assert outcomes == ['incomplete'], 'the batch completed twice'
    finally:
        release.set()
        notifications.remove_listener(listener)
        ex.shutdown(wait=True)


def test_backpressure_blocked_time_accumulates_and_resets_per_run(file_lane, monkeypatch):
    """The demand-relative slow-disk signal: time spent blocked waiting for
    room in the backlog accumulates across a run, and the next run's batch
    starts at zero."""
    monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', 1)
    ex = file_lane
    batch = RunWriteBatch(ex)
    release = _park_worker(ex)
    try:
        _fill(batch, 1)

        aborted = threading.Event()
        submitted = threading.Event()

        def _submit():
            batch.submit(lambda: None, {}, what='The image', pace_until=aborted.is_set)
            submitted.set()

        t = threading.Thread(target=_submit, daemon=True)
        t.start()
        assert not submitted.wait(0.7), 'submit returned against a full backlog instead of blocking'
        aborted.set()
        assert submitted.wait(1.0)

        assert batch.blocked_s >= 0.5, 'blocked-enqueue time must accumulate while waiting for room'
    finally:
        release.set()
    assert RunWriteBatch(ex).blocked_s == 0.0, (
        'a fresh run must not inherit the previous run blocked-wait total'
    )


def test_run_cleanup_surfaces_slow_write_warning(monkeypatch):
    """Run-end summary fires the sustained-slow-write warning when the run's
    blocked-wait total crossed the threshold, and stays silent otherwise."""
    from unittest.mock import MagicMock

    from modules.notification_center import notifications as nc_notifications
    from modules.protocol_cleanup import run_cleanup
    from tests.test_audit_fixes import _run_cleanup_kwargs

    captured = []
    monkeypatch.setattr(nc_notifications, 'warning', lambda *a, **k: captured.append(a))

    slow_batch = MagicMock(spec=RunWriteBatch, blocked_s=45.0, pending=0)
    run_cleanup(**_run_cleanup_kwargs(write_batch=slow_batch))
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
    lane = runner.file_io_executor
    lane.put.return_value = ENQUEUED
    lane.in_flight_task_stalled.return_value = True
    lane.describe_running_task.return_value = "write_capture 'B2_BF' 45s in flight"
    # The last run ended with one write still to land: its files are draining.
    last_run = RunWriteBatch(lane)
    last_run.submit(lambda: None, {}, what='The image B2_BF', pace_until=None)
    last_run.close(lambda outcome: None)
    runner._write_batch = last_run

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

    lane.in_flight_task_stalled.return_value = False
    with _pytest.raises(ProtocolRunRefusedError) as excinfo:
        _prepare()
    assert excinfo.value.reason == 'files_writing'


def test_session_recover_file_writer_passthrough():
    """L2 parity: a Session recovers a wedged writer -- the last run's
    stuck writes given up on and counted, the FILE lane's worker replaced
    -- and refuses when nothing is stuck, since recovery would lose images
    for nothing."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from modules.exceptions import FileWriterNotStuckError
    from modules.scope_session import ScopeSession

    bundle = SimpleNamespace(
        file_io_executor=MagicMock(),
        post_processing_executor=MagicMock(),
        protocol_thread=MagicMock(),
    )
    session = ScopeSession(settings={}, scope=MagicMock(), executor_bundle=bundle)

    with pytest.raises(FileWriterNotStuckError) as refused:
        session.recover_file_writer()
    assert refused.value.reason == 'file_writer_not_stuck'
    bundle.file_io_executor.replace_stuck_worker.assert_not_called()

    lane = bundle.file_io_executor
    lane.put.return_value = ENQUEUED
    lane.in_flight_task_stalled.return_value = True
    stuck = RunWriteBatch(lane)
    stuck.submit(lambda: None, {}, what='The image', pace_until=None)
    stuck.close(lambda outcome: None)
    session.sequenced_capture_runner._write_batch = stuck

    assert session.recover_file_writer() == 1
    bundle.file_io_executor.replace_stuck_worker.assert_called_once()
    assert stuck.outcome == 'incomplete'


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
