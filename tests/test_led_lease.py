# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Unit tests for the LED ownership lease primitive.

The lease is the enforced layer above the advisory owner tags: while a lease
is held, only its owner may drive the LEDs, and a second owner's acquire is
refused. Autofocus running inside a protocol step takes a child lease the
parent must outlive. Release turns the owner's channels off by default so the
end-state is a property of the release. force_off is the unblockable bypass
for emergency / error paths; reset_led_leases frees a wedged lease on abort.

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
    lease = scope.illumination.acquire_led_lease('protocol')
    assert lease is not None
    assert lease.held
    assert scope.illumination.led_lease_owner == 'protocol'


def test_second_owner_acquire_refused(scope):
    scope.illumination.acquire_led_lease('protocol')
    denied = scope.illumination.acquire_led_lease('autofocus')
    assert denied is None
    assert scope.illumination.led_lease_owner == 'protocol'


def test_release_frees_lease_for_next_owner(scope):
    lease = scope.illumination.acquire_led_lease('protocol')
    lease.release()
    assert scope.illumination.led_lease_owner is None
    second = scope.illumination.acquire_led_lease('autofocus')
    assert second is not None
    assert scope.illumination.led_lease_owner == 'autofocus'


def test_child_lease_acquired_only_by_holder(scope):
    parent = scope.illumination.acquire_led_lease('protocol')
    child = parent.acquire_child('autofocus')
    assert child is not None
    # The innermost holder is now the active owner.
    assert scope.illumination.led_lease_owner == 'autofocus'


def test_child_acquire_with_stale_parent_refused(scope):
    parent = scope.illumination.acquire_led_lease('protocol')
    parent.release()
    child = parent.acquire_child('autofocus')
    assert child is None
    assert scope.illumination.led_lease_owner is None


def test_child_release_returns_control_to_parent(scope):
    parent = scope.illumination.acquire_led_lease('protocol')
    child = parent.acquire_child('autofocus')
    assert scope.illumination.led_write_allowed('autofocus') is True
    assert scope.illumination.led_write_allowed('protocol') is False
    child.release()
    assert scope.illumination.led_lease_owner == 'protocol'
    assert scope.illumination.led_write_allowed('protocol') is True


def test_led_write_allowed_open_when_unleased(scope):
    # No lease held: any owner, including a bare UI click (empty owner), may write.
    assert scope.illumination.led_write_allowed('autofocus') is True
    assert scope.illumination.led_write_allowed('') is True


def test_led_write_allowed_rejects_non_owner_when_leased(scope):
    scope.illumination.acquire_led_lease('protocol')
    assert scope.illumination.led_write_allowed('protocol') is True
    assert scope.illumination.led_write_allowed('autofocus') is False
    assert scope.illumination.led_write_allowed('') is False


def test_release_turns_owned_leds_off_by_default(scope):
    lease = scope.illumination.acquire_led_lease('protocol')
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    assert _lit(scope, 0)
    lease.release()
    assert not _lit(scope, 0)


def test_release_leave_on_keeps_leds_lit(scope):
    lease = scope.illumination.acquire_led_lease('protocol')
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    lease.release(leave_on=True)
    assert _lit(scope, 0)


def test_double_release_is_noop(scope):
    lease = scope.illumination.acquire_led_lease('protocol')
    lease.release()
    # A second release must not raise and must not turn off a LED a later
    # owner has since lit.
    second = scope.illumination.acquire_led_lease('autofocus')
    scope.illumination.led_on(channel=0, mA=100, owner='autofocus')
    lease.release()
    assert _lit(scope, 0)
    assert scope.illumination.led_lease_owner == 'autofocus'
    assert second.held


def test_force_off_bypasses_lease_and_logs(scope, caplog):
    scope.illumination.acquire_led_lease('protocol')
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.force_off()
    assert not _lit(scope, 0)
    # The lease is left intact -- the holder still releases normally.
    assert scope.illumination.led_lease_owner == 'protocol'
    assert any('force_off bypassing held LED lease' in r.message for r in caplog.records)


def test_reset_led_leases_frees_without_touching_leds(scope):
    scope.illumination.acquire_led_lease('protocol')
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    scope.illumination.reset_led_leases()
    # Lease gone, but the LED is physically untouched (caller decides via force_off).
    assert scope.illumination.led_lease_owner is None
    assert _lit(scope, 0)
    # The next run can acquire.
    assert scope.illumination.acquire_led_lease('autofocus') is not None


def test_context_manager_acquires_and_releases(scope):
    with scope.illumination.acquire_led_lease('protocol') as lease:
        assert lease.held
        assert scope.illumination.led_lease_owner == 'protocol'
    assert scope.illumination.led_lease_owner is None


def test_lease_violation_detects_external_writer(scope):
    assert scope.illumination._lease_violation('autofocus') is None  # unleased
    scope.illumination.acquire_led_lease('protocol')
    assert scope.illumination._lease_violation('protocol') is None  # the holder
    assert scope.illumination._lease_violation('autofocus') == 'protocol'
    assert scope.illumination._lease_violation('') == 'protocol'  # bare UI click


def test_owner_leds_exclusive_does_not_self_violate(scope, caplog):
    # The lease holder's own exclusive call clears other channels via an
    # internal owner-less off; that must NOT be flagged as a violation.
    scope.illumination.acquire_led_lease('protocol')
    scope.illumination.led_on(channel=0, mA=100, owner='protocol')
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.leds_exclusive(channel=3, mA=200, owner='protocol')
    assert not any('holds the lease' in r.message for r in caplog.records)


def test_external_write_during_lease_is_observed_and_applied(scope, caplog):
    # Shadow phase: an out-of-turn write is logged but still applied.
    scope.illumination.acquire_led_lease('protocol')
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        scope.illumination.led_on(channel=0, mA=100, owner='')
    assert _lit(scope, 0)
    assert any('holds the lease' in r.message for r in caplog.records)
