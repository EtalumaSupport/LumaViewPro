# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: a fatal run abort must darken the sample immediately and
must never block on the fault that caused it.

Bench-measured failure shape: the wedge abort wrote the lost capture's
record row synchronously to the save target that had just been declared
dead, blocking the protocol thread for the whole OS timeout (~60 s on a
dead SMB share) before the abort -- and the LED-off behind it -- could
run. The lit channel illuminated the sample the entire time, and would
indefinitely on a share that never times out.

Contract now: every run-killing fault routes through one funnel that
aborts (closes the step-lighting gates), sets the run's fatal flag,
force-darkens via the direct driver path, and only then notifies; the
execution record latches dead so no further row write can block; cleanup
ASSERTS dark on fatal (forces the end policy to OFF) rather than trusting
the user's end policy; the step-boundary hold is gated on the fatal flag
so a cross-thread fatal cannot be undone by a completing capture.
"""

import ast
import datetime
import pathlib
import threading
from unittest.mock import MagicMock

from modules.lumascope_api.illumination import LedEndPolicy, LedTransition
from modules.exceptions import RecordIncompleteError
from modules.notification_center import notifications
from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.protocol_state_machine import ProtocolState
from modules.protocol_cleanup import RunCompleteNotice
from modules.protocol_image_writer import RunWriteBatch
from tests.protocol_drives import autofocus_snapshot

from tests.test_audit_fixes import _bare_protocol_writer
from tests.test_protocol_modules import _FakeExecutor

_MODULES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'modules'


# ---------------------------------------------------------------------------
# Funnel: abort -> fatal flag -> force_off -> critical, record only after
# ---------------------------------------------------------------------------


from modules.run_outcome import RunEnding
from tests.scope_fakes import swap_lanes


def test_wedge_funnel_order_abort_then_dark_then_notify(monkeypatch):
    order = []
    fatal_event = threading.Event()
    orig_set = fatal_event.set
    fatal_event.set = lambda: (order.append('fatal'), orig_set())[1]

    # The writer is stuck: the backlog is full (a bound of 0 is full at
    # once) past the stall budget, and the write in flight is stalled, so
    # the run's batch answers the submit wedged.
    monkeypatch.setattr('modules.protocol_image_writer.WRITE_BACKLOG_BOUND', 0)
    monkeypatch.setattr('modules.protocol_image_writer.WRITE_STALL_FATAL_S', 0.0)
    file_io_executor = MagicMock()
    file_io_executor.in_flight_task_stalled.return_value = True
    file_io_executor.describe_running_task.return_value = "write_capture 'x' 32s in flight"

    record = MagicMock()
    record.mark_target_unresponsive.side_effect = lambda: order.append('latch')
    record.add_step.side_effect = lambda **kw: order.append('row')

    writer = _bare_protocol_writer(
        write_batch=RunWriteBatch(file_io_executor),
        execution_record=record,
        abort_fn=lambda: order.append('abort'),
        fatal_abort_event=fatal_event,
    )
    writer._scope.illumination.force_off.side_effect = lambda: order.append('force_off')
    monkeypatch.setattr(notifications, 'critical', lambda *a, **k: order.append('critical'))

    submitted = writer._submit_write(
        kwargs={},
        step={'Name': 's'},
        step_index=0,
        scan_count=0,
        capture_time=datetime.datetime.now(),
        name='x',
    )

    assert submitted is False
    assert order == ['abort', 'fatal', 'force_off', 'critical', 'latch', 'row'], (
        'fatal abort must close the step gates and darken the sample BEFORE '
        'any notification or record write can run (or block): ' + repr(order)
    )
    assert fatal_event.is_set()


# ---------------------------------------------------------------------------
# Dead-target latch: add_step no-ops, complete() keeps its reconcile warning
# ---------------------------------------------------------------------------


def test_latched_record_writes_nothing_but_still_reconciles(tmp_path, monkeypatch):
    outfile = tmp_path / 'protocol_record.tsv'
    record = ProtocolExecutionRecord(protocol_file_loc=tmp_path / 'p.tsv', outfile=outfile)

    def row_kwargs(name):
        return {
            'capture_result_file_name': name,
            'step_name': name,
            'step_index': 0,
            'scan_count': 0,
            'timestamp': datetime.datetime.now(),
        }

    record.note_capture_attempt()
    record.add_step(**row_kwargs('before_latch'))
    size_before = outfile.stat().st_size

    record.mark_target_unresponsive()
    record.note_capture_attempt()
    record.add_step(**row_kwargs('after_latch'))
    assert outfile.stat().st_size == size_before, (
        'a latched record must not touch the filesystem -- the write against '
        'a declared-dead target is exactly the block being prevented'
    )

    warnings = []
    try:
        record.complete()
    except RecordIncompleteError as shortfall:
        warnings.append(shortfall)
    assert len(warnings) == 1, (
        'complete() must stay un-latched: it does no filesystem I/O and its '
        'reconcile warning is the only surviving report of the lost row'
    )


# ---------------------------------------------------------------------------
# Cleanup: fatal asserts dark; user Stop keeps the configured policy
# ---------------------------------------------------------------------------


def _run_cleanup_capture_led_ctx(*, forced_dark, leds_state_at_end):
    from modules.protocol_cleanup import run_cleanup

    applied = []
    state = [ProtocolState.RUNNING]
    scope = MagicMock()
    scope.illumination.color2ch.return_value = 1
    af_thread = MagicMock()
    af_thread.current_future = None
    callbacks = ProtocolCallbacks()
    ending = RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped')

    swap_lanes(scope, io=_FakeExecutor(), camera=_FakeExecutor())
    run_cleanup(
        get_state_fn=lambda: state[0],
        set_state_fn=lambda s: state.__setitem__(0, s),
        scan_in_progress=threading.Event(),
        forced_dark=forced_dark,
        leds_state_at_end=leds_state_at_end,
        original_led_states={'Blue': {'enabled': True, 'illumination_ma': 42.0}},
        autofocus_snapshot=autofocus_snapshot(states={}),
        saved_camera_state=None,
        return_to_position=None,
        scope=scope,
        callbacks=callbacks,
        apply_led_transition_fn=lambda transition, ctx: applied.append((transition, ctx)),
        default_move_fn=lambda **kw: None,
        cancel_scheduled_events_fn=lambda: None,
        autofocus_thread=af_thread,
        write_batch=RunWriteBatch(_FakeExecutor()),
        run_complete=RunCompleteNotice(callbacks, protocol=None, ending=ending, run_dir=None),
        ending=ending,
        record_cleanup_failures=lambda steps: None,
    )
    run_end = [ctx for t, ctx in applied if t is LedTransition.RUN_END]
    assert len(run_end) == 1
    return run_end[0]


def test_fatal_cleanup_asserts_dark_regardless_of_end_policy():
    ctx = _run_cleanup_capture_led_ctx(forced_dark=True, leds_state_at_end='return_to_original')
    assert ctx.end_policy is LedEndPolicy.OFF
    assert ctx.snapshot_lit == frozenset(), (
        'a fatal abort must ASSERT dark -- restoring pre-run channels after a '
        'run died on a fault would re-illuminate the sample'
    )


def test_nonfatal_cleanup_keeps_the_configured_end_policy():
    ctx = _run_cleanup_capture_led_ctx(forced_dark=False, leds_state_at_end='return_to_original')
    assert ctx.end_policy is LedEndPolicy.RETURN_TO_ORIGINAL
    assert ctx.snapshot_lit != frozenset(), 'user Stop keeps the configured restore'


# ---------------------------------------------------------------------------
# Source pins: the funnel is the only critical path; the boundary gate is
# keyed on the fatal flag, not the abort flag
# ---------------------------------------------------------------------------


def test_no_critical_notification_outside_the_fatal_funnel():
    src = (_MODULES_DIR / 'protocol_image_writer.py').read_text()
    tree = ast.parse(src)
    offenders = []
    funnel_range = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_abort_run_fatal':
            funnel_range = (node.lineno, node.end_lineno)
    assert funnel_range is not None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'critical'
            and not (funnel_range[0] <= node.lineno <= funnel_range[1])
        ):
            offenders.append(node.lineno)
    assert offenders == [], (
        'every run-killing critical must route through _abort_run_fatal so '
        'no fatal path can notify without first aborting and darkening: '
        f'bare criticals at lines {offenders}'
    )


def test_step_boundary_gate_keys_on_fatal_not_aborted():
    src = (_MODULES_DIR / 'protocol_step_runner.py').read_text()
    assert 'if completed and not p._fatal_abort_event.is_set():' in src, (
        'the STEP_BOUNDARY apply must be gated on the fatal flag; gating on '
        'the abort flag would skip the boundary EXTINGUISH on a user Stop '
        'and lengthen sample illumination'
    )
