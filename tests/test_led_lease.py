# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for the LED ownership lease primitive.

While a lease is held, only that lease object may drive the LEDs -- its
purpose label grants nothing -- and a second acquire is refused. Autofocus
running inside a protocol step takes a child lease the parent must outlive.
Release turns off the channels the lease lit by default so the end-state is a
property of the release. force_off is the unblockable bypass
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
    return scope.illumination.get_led_state(color)['enabled']


def test_acquire_when_unleased_returns_token(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert lease is not None
    assert lease.held
    assert scope.illumination.led_lease_purpose == 'protocol'


def test_second_owner_acquire_refused(scope):
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    denied = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    assert denied is None
    assert scope.illumination.led_lease_purpose == 'protocol'


def test_release_frees_lease_for_next_owner(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    lease.release()
    assert scope.illumination.led_lease_purpose is None
    second = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    assert second is not None
    assert scope.illumination.led_lease_purpose == 'autofocus'


def test_child_lease_acquired_only_by_holder(scope):
    parent = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    child = parent.acquire_child('autofocus', alive=lambda: True)
    assert child is not None
    # The innermost holder is now the active owner.
    assert scope.illumination.led_lease_purpose == 'autofocus'


def test_child_acquire_with_stale_parent_refused(scope):
    parent = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    parent.release()
    child = parent.acquire_child('autofocus', alive=lambda: True)
    assert child is None
    assert scope.illumination.led_lease_purpose is None


def test_child_release_returns_control_to_parent(scope):
    parent = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    child = parent.acquire_child('autofocus', alive=lambda: True)
    # Ownership is the observable: the innermost holder owns while it lives,
    # and the parent does not get control back until the child releases.
    assert scope.illumination.led_lease_purpose == 'autofocus'
    child.release()
    assert scope.illumination.led_lease_purpose == 'protocol'


def test_release_turns_owned_leds_off_by_default(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    assert _lit(scope, 0)
    lease.release()
    assert not _lit(scope, 0)


def test_release_leave_on_keeps_leds_lit(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    lease.release(leave_on=True)
    assert _lit(scope, 0)


def test_double_release_is_noop(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    lease.release()
    # A second release must not raise and must not turn off a LED a later
    # owner has since lit.
    second = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=second)
    lease.release()
    assert _lit(scope, 0)
    assert scope.illumination.led_lease_purpose == 'autofocus'
    assert second.held


def test_force_off_bypasses_lease_and_logs(scope, caplog):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.force_off()
    assert not _lit(scope, 0)
    # The lease is left intact -- the holder still releases normally.
    assert scope.illumination.led_lease_purpose == 'protocol'
    assert any(
        "force_off bypassing the held 'protocol' LED lease" in r.message for r in caplog.records
    )


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
    assert scope.illumination.led_lease_purpose == 'next'
    assert not wedged.held, 'the reclaimed lease must report not held'


def test_context_manager_acquires_and_releases(scope):
    with scope.illumination.acquire_led_lease('protocol', alive=lambda: True) as lease:
        assert lease.held
        assert scope.illumination.led_lease_purpose == 'protocol'
    assert scope.illumination.led_lease_purpose is None


def test_lease_violation_detects_external_writer(scope):
    # A lease from an earlier, finished operation: the "another writer" whose
    # object is not the active holder.
    stale = scope.illumination.acquire_led_lease('autofocus', alive=lambda: True)
    stale.release()
    assert scope.illumination._lease_violation(stale) is None  # unleased
    assert scope.illumination._lease_violation(None) is None  # unleased
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    assert scope.illumination._lease_violation(lease) is None  # the holder
    assert scope.illumination._lease_violation(stale) == 'protocol'
    assert scope.illumination._lease_violation(None) == 'protocol'  # bare UI click
    # The holder's purpose NAME is not a credential: only the object is.
    assert scope.illumination._lease_violation('protocol') == 'protocol'


def test_owner_emit_diff_does_not_self_violate(scope, caplog):
    # The lease holder driving its own diff clears other channels via an off
    # made as the holder's lease; that must NOT be flagged as a violation of
    # the holder's own lease.
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination._emit_led_diff(frozenset({(3, 200.0)}), lease=lease, block=False)
    assert not any('refused' in r.message for r in caplog.records)
    assert not _lit(scope, 0), 'the holder diff must clear the non-target channel'
    assert _lit(scope, 3), 'the holder diff must light its target'


def test_external_led_on_during_lease_is_refused(scope, caplog):
    # A live UI write (no lease) while a run holds the LEDs is rejected.
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.led_on(channel=0, illumination_ma=100)
    assert not _lit(scope, 0)
    assert any('refused' in r.message for r in caplog.records)


def test_external_led_off_during_lease_is_refused(scope):
    # The autofocus-LED-killed shape: a UI off must not turn off a channel a
    # run holds. The protocol lit the channel; a bare UI off is refused.
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    scope.illumination.led_off(channel=0)  # live UI off
    assert _lit(scope, 0)


def test_owner_write_during_own_lease_is_allowed(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    # Naming the holder's purpose is not holding the lease: refused.
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease='protocol')
    assert not _lit(scope, 0), 'a write naming the holder was permitted without its lease'
    # Writing as the lease object itself is permitted.
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    assert _lit(scope, 0)


def test_force_off_still_works_under_enforcement(scope):
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    scope.illumination.force_off()
    assert not _lit(scope, 0)


def test_leds_off_turns_off_during_a_held_lease(scope):
    # The app-shutdown / emergency path: leds_off is nuclear and must turn the
    # holder's lit channel off even while a run holds the lease, so closing the
    # app mid-run cannot leave an LED stuck on.
    lease = scope.illumination.acquire_led_lease('protocol', alive=lambda: True)
    scope.illumination._led_on_impl(channel=0, illumination_ma=100, _lease=lease)
    assert _lit(scope, 0)
    scope.illumination.leds_off()
    assert not _lit(scope, 0)
