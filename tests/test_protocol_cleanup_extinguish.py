# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Protocol cleanup darkens the LEDs when the run's end-state is undecided.

The RUN_END LED transition runs inside run_cleanup's fault-tolerant try:
a raise before it (autofocus unwind), a failure inside it, or a
cancellation of its queue task all leave it un-applied -- and the lease
release deliberately leaves LEDs as-is. Before this guarantee, every one
of those paths left the sample lit with no owner to turn it off. The
force-dark must be owner-blind: the lit channel can be recorded to an
autofocus child lease or to no owner at all (a channel already lit at
the commanded current records no ownership), so an owner-scoped darken
misses exactly the channels at issue.

The counter-cases are as load-bearing as the darkening: a cleanup whose
RUN_END applied must honor the user's end policy untouched, and a
cleanup that never owned the run's LEDs (double cleanup, early return)
must not darken a prior cleanup's restored end-state.
"""

import types
from unittest.mock import MagicMock

import pytest

from modules.lumascope_api.illumination import LedEndPolicy, LedState

import modules.sequenced_capture_runner as scr
from tests.protocol_drives import held_run_claim
from modules.run_outcome import PendingRunOutcome, RunEnding
from tests.scope_fakes import build_scope

LAYER = 'Blue'
ILLUMINATION_MA = 10.0


@pytest.fixture
def scope():
    s = build_scope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    return s


def _lit(scope):
    return scope.illumination.get_led_state(LAYER).state is LedState.LIT


def _make_runner_stub(scope, *, lease):
    """A stand-in SequencedCaptureRunner carrying only what _cleanup_inner touches.

    MagicMock base so run_cleanup's kwarg expressions (state fns, executor
    handles) resolve; the load-bearing slots are set explicitly because
    auto-created mock attributes are truthy and would defeat the
    lease-held gate. The stub's current run is the run its cleanup is for,
    so the pass owns the run; the batch close in cleanup's finally lands on
    an inert mock slot.
    """
    stub = MagicMock()
    run = scr.EngineRun(stub, PendingRunOutcome(), scr.RunWriteBatch(MagicMock()))
    stub.run_outcome = lambda: run
    stub._scope = scope
    stub._led_lease = lease
    stub._image_writer = None
    stub._is_run_live = lambda: True
    stub.LOGGER_NAME = 'TestCleanup'
    stub._start_hyperstack_build = lambda on_end: None
    stub._release_scan_led_lease = types.MethodType(
        scr.SequencedCaptureRunner._release_scan_led_lease, stub
    )
    stub._release_activity_claim = lambda: None
    return stub


def _run_cleanup_inner(stub, ending=None):
    if ending is None:
        ending = RunEnding('failed', 'run_loop_crashed', 'died')
    scr.SequencedCaptureRunner._cleanup_inner(stub, ending, stub.run_outcome(), [])


def test_run_cleanup_raise_darkens_before_release(scope, monkeypatch):
    # Lit before the lease exists, so no lease is recorded as having lit it:
    # the exact case a lit-by-lease-scoped darken misses.
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: lit before the fault'
    lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
    assert lease is not None

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(side_effect=RuntimeError('cleanup died')))
    stub = _make_runner_stub(scope, lease=lease)

    with pytest.raises(RuntimeError):
        _run_cleanup_inner(stub)

    assert not _lit(scope), (
        'a raise before the RUN_END transition leaves the end-state '
        'undecided; cleanup must darken before releasing the lease'
    )
    assert scope.illumination.led_lease_purpose is None, 'the lease must still release'


def test_run_cleanup_undecided_return_darkens(scope, monkeypatch):
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: lit before the fault'
    lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
    assert lease is not None

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(return_value=False))
    stub = _make_runner_stub(scope, lease=lease)

    _run_cleanup_inner(stub)

    assert not _lit(scope), (
        'a cancelled or failed RUN_END restore returns undecided; '
        'cleanup must darken before releasing the lease'
    )
    assert scope.illumination.led_lease_purpose is None


def test_cleanup_without_lease_does_not_darken(scope, monkeypatch):
    # Behavior-preservation guard: passes before and after the fix. The
    # early-return path (run already unwound) never owned the run's LEDs;
    # darkening here would kill a prior cleanup's restored end-state.
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope)

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(return_value=False))
    stub = _make_runner_stub(scope, lease=None)
    stub._is_run_live = lambda: False

    _run_cleanup_inner(stub)

    assert _lit(scope), 'a cleanup that never held the lease must leave the LED end-state alone'


def _cleanup_told(scope, monkeypatch, apply_led_transition_fn):
    """Run run_cleanup with this RUN_END transition; what it logged at INFO."""
    import threading

    import modules.protocol_cleanup as protocol_cleanup
    from modules.protocol_state_machine import ProtocolState

    told = []
    monkeypatch.setattr(
        protocol_cleanup.logger, 'info', lambda message, *a, **k: told.append(message)
    )
    applied = protocol_cleanup.run_cleanup(
        get_state_fn=lambda: ProtocolState.RUNNING,
        set_state_fn=lambda state: None,
        scan_in_progress=threading.Event(),
        forced_dark=False,
        led_end_policy=LedEndPolicy.OFF,
        original_led_states=(),
        saved_camera_state={},
        show_pre_run_layer_fn=None,
        return_to_position=None,
        scope=scope,
        apply_led_transition_fn=apply_led_transition_fn,
        default_move_fn=lambda *a, **k: None,
        cancel_scheduled_events_fn=lambda: None,
        autofocus_thread=None,
        write_batch=scr.RunWriteBatch(MagicMock()),
        ending=RunEnding('completed', 'completed', 'The run finished normally.'),
        record_cleanup_failures=lambda steps: None,
        failed_before=(),
    )
    return applied, told


def test_a_restore_that_did_not_apply_is_not_logged_as_complete(scope, monkeypatch):
    # A stand-in by design: the subject is the cleanup catching the run's LED
    # transition, and the transition is its interchangeable collaborator.
    from concurrent.futures import CancelledError

    for fault in (RuntimeError('LED board did not answer'), CancelledError()):

        def _fails(transition, ctx, fault=fault):
            raise fault

        applied, told = _cleanup_told(scope, monkeypatch, _fails)

        assert applied is False
        assert [line for line in told if 'LED' in line and 'complete' in line] == [], (
            f'a restore that raised {type(fault).__name__} must not be logged as complete'
        )


def test_a_restore_that_applied_is_logged_with_what_it_applied(scope, monkeypatch):
    applied, told = _cleanup_told(scope, monkeypatch, lambda transition, ctx: None)

    assert applied is True
    assert [line for line in told if 'LED end state applied' in line] != []
