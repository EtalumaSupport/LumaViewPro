# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for the LED ownership lease primitive.

The lease is the enforced layer above the advisory owner tags: while a lease
is held, only its owner may drive the LEDs, and a second owner's acquire is
refused. Autofocus running inside a protocol step takes a child lease the
parent must outlive. Release turns the owner's channels off by default so the
end-state is a property of the release. force_off is the unblockable bypass
for emergency / error paths; a lease wedged by a provably-dead owner is
reclaimed with evidence inside the next acquire, never reset by a caller.

These exercise the primitive in isolation -- no production caller acquires a
lease yet, so app behavior is unchanged. The run-boundary callers adopt it in
a later stage.
"""

import logging

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    yield s


def _lit(scope, ch):
    """Whether the channel at *ch* is currently lit (API source of truth)."""
    color = scope.illumination.ch2color(ch)
    return scope.illumination.led_enabled(color)


def test_acquire_when_unleased_returns_token(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert lease is not None
    assert lease.held
    assert scope.illumination.led_lease_owner == 'protocol'


def test_second_owner_acquire_refused(scope):
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    denied = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    assert denied is None
    assert scope.illumination.led_lease_owner == 'protocol'


def test_release_frees_lease_for_next_owner(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    lease.release()
    assert scope.illumination.led_lease_owner is None
    second = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    assert second is not None
    assert scope.illumination.led_lease_owner == 'autofocus'


def test_child_lease_acquired_only_by_holder(scope):
    parent = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    child = parent.acquire_child('autofocus', alive=lambda: True)
    assert child is not None
    # The innermost holder is now the active owner.
    assert scope.illumination.led_lease_owner == 'autofocus'


def test_child_acquire_with_stale_parent_refused(scope):
    parent = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    parent.release()
    child = parent.acquire_child('autofocus', alive=lambda: True)
    assert child is None
    assert scope.illumination.led_lease_owner is None


def test_child_release_returns_control_to_parent(scope):
    parent = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    child = parent.acquire_child('autofocus', alive=lambda: True)
    # Ownership is the observable: the innermost holder owns while it lives,
    # and the parent does not get control back until the child releases.
    assert scope.illumination.led_lease_owner == 'autofocus'
    child.release()
    assert scope.illumination.led_lease_owner == 'protocol'


def test_release_turns_owned_leds_off_by_default(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    assert _lit(scope, 0)
    lease.release()
    assert not _lit(scope, 0)


def test_release_leave_on_keeps_leds_lit(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    lease.release(leave_on=True)
    assert _lit(scope, 0)


def test_double_release_is_noop(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    lease.release()
    # A second release must not raise and must not turn off a LED a later
    # owner has since lit.
    second = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
    lease.release()
    assert _lit(scope, 0)
    assert scope.illumination.led_lease_owner == 'autofocus'
    assert second.held


def test_force_off_bypasses_lease_and_logs(scope, caplog):
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.force_off()
    assert not _lit(scope, 0)
    # The lease is left intact -- the holder still releases normally.
    assert scope.illumination.led_lease_owner == 'protocol'
    assert any('force_off bypassing held LED lease' in r.message for r in caplog.records)


def test_dead_owner_lease_is_reclaimed_by_next_acquire(scope):
    # A wedged lease from a dead owner must not lock out the next run. The
    # holder's liveness probe answering False is the evidence that lets the
    # next acquire reclaim the stack instead of being refused.
    holder_alive = {'value': True}
    wedged = scope.illumination.acquire_led_lease('protocol', alive=lambda: holder_alive['value'])
    assert wedged is not None
    holder_alive['value'] = False  # the owning run died without releasing

    nxt = scope.illumination.acquire_led_lease('next', alive=lambda: True)
    assert nxt is not None, 'a dead holder must not lock out the next acquire'
    assert scope.illumination.led_lease_owner == 'next'
    assert not wedged.held, 'the reclaimed lease must report not held'


def test_context_manager_acquires_and_releases(scope):
    with scope.illumination.acquire_led_lease('protocol', alive=lambda: True) as lease:
        assert lease.held
        assert scope.illumination.led_lease_owner == 'protocol'
    assert scope.illumination.led_lease_owner is None


def test_lease_violation_detects_external_writer(scope):
    assert scope.illumination._lease_violation('autofocus') is None  # unleased
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert scope.illumination._lease_violation('protocol') is None  # the holder
    assert scope.illumination._lease_violation('autofocus') == 'protocol'
    assert scope.illumination._lease_violation('') == 'protocol'  # bare UI click


def test_owner_emit_diff_does_not_self_violate(scope, caplog):
    # The lease holder driving its own diff clears other channels via an
    # owner-less off whose lease check is tagged with the holder; that must
    # NOT be flagged as a violation of the holder's own lease.
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination._emit_led_diff(frozenset({(3, 200.0)}), owner='protocol', block=False)
    assert not any('holds the lease' in r.message for r in caplog.records)


def test_external_led_on_during_lease_is_refused(scope, caplog):
    # A live UI write (empty owner) while a run owns the LEDs is rejected.
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.led_on(channel=0, mA=100, owner='')
    assert not _lit(scope, 0)
    assert any('refused' in r.message for r in caplog.records)


def test_external_led_off_during_lease_is_refused(scope):
    # The autofocus-LED-killed shape: a UI off must not turn off a channel a
    # run owns. The protocol lit the channel; a bare UI off is refused.
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    scope.illumination.led_off(channel=0, owner='')  # live UI off
    assert _lit(scope, 0)


def test_owner_write_during_own_lease_is_allowed(scope):
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    assert _lit(scope, 0)


def test_force_off_still_works_under_enforcement(scope):
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    scope.illumination.force_off()
    assert not _lit(scope, 0)


def test_leds_off_turns_off_during_a_held_lease(scope):
    # The app-shutdown / emergency path: leds_off is nuclear and must turn the
    # owner's lit channel off even while a run holds the lease, so closing the
    # app mid-run cannot leave an LED stuck on.
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    assert _lit(scope, 0)
    scope.illumination.leds_off()
    assert not _lit(scope, 0)
