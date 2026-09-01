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

import threading
import types
from unittest.mock import MagicMock

import pytest

import modules.sequenced_capture_runner as scr
from modules.lumascope_api import Lumascope

LAYER = 'Blue'
ILLUMINATION_MA = 10.0


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    return s


def _lit(scope):
    return scope.illumination.get_led_state(LAYER)['enabled']


def _make_runner_stub(scope, *, lease):
    """A stand-in SequencedCaptureRunner carrying only what _cleanup_inner touches.

    MagicMock base so run_cleanup's kwarg expressions (state fns, executor
    handles) resolve; the load-bearing slots are set explicitly because
    auto-created mock attributes are truthy and would defeat the
    lease-held gate.
    """
    stub = MagicMock()
    stub._scope = scope
    stub._led_lease = lease
    stub._image_writer = None
    stub._run_in_progress_event = threading.Event()
    stub._run_in_progress_event.set()
    stub.LOGGER_NAME = 'TestCleanup'
    stub._start_hyperstack_build = lambda: None
    stub._release_scan_led_lease = types.MethodType(
        scr.SequencedCaptureRunner._release_scan_led_lease, stub
    )
    stub._release_activity_claim = lambda: None
    return stub


def _run_cleanup_inner(stub, run_status='failed'):
    scr.SequencedCaptureRunner._cleanup_inner(stub, run_status)


def test_run_cleanup_raise_darkens_before_release(scope, monkeypatch):
    # Lit before the lease exists, so the channel carries no owner record:
    # the exact case an owner-scoped darken misses.
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: lit before the fault'
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert lease is not None

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(side_effect=RuntimeError('cleanup died')))
    stub = _make_runner_stub(scope, lease=lease)

    with pytest.raises(RuntimeError):
        _run_cleanup_inner(stub)

    assert not _lit(scope), (
        'a raise before the RUN_END transition leaves the end-state '
        'undecided; cleanup must darken before releasing the lease'
    )
    assert scope.illumination.led_lease_owner is None, 'the lease must still release'


def test_run_cleanup_undecided_return_darkens(scope, monkeypatch):
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope), 'precondition: lit before the fault'
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert lease is not None

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(return_value=False))
    stub = _make_runner_stub(scope, lease=lease)

    _run_cleanup_inner(stub)

    assert not _lit(scope), (
        'a cancelled or failed RUN_END restore returns undecided; '
        'cleanup must darken before releasing the lease'
    )
    assert scope.illumination.led_lease_owner is None


def test_decided_end_state_is_left_untouched(scope, monkeypatch):
    # Behavior-preservation guard: passes before and after the fix. When
    # RUN_END applied, the user's end policy owns the LEDs.
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope)
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert lease is not None

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(return_value=True))
    stub = _make_runner_stub(scope, lease=lease)

    _run_cleanup_inner(stub)

    assert _lit(scope), 'a decided end-state must not be overridden by a force-dark'
    assert scope.illumination.led_lease_owner is None


def test_cleanup_without_lease_does_not_darken(scope, monkeypatch):
    # Behavior-preservation guard: passes before and after the fix. The
    # early-return path (run already unwound) never owned the run's LEDs;
    # darkening here would kill a prior cleanup's restored end-state.
    scope.illumination._led_on_impl(LAYER, ILLUMINATION_MA)
    assert _lit(scope)

    monkeypatch.setattr(scr, 'run_cleanup', MagicMock(return_value=False))
    stub = _make_runner_stub(scope, lease=None)
    stub._run_in_progress_event.clear()

    _run_cleanup_inner(stub)

    assert _lit(scope), 'a cleanup that never held the lease must leave the LED end-state alone'
