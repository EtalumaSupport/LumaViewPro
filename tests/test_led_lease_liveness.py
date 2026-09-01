# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Liveness arbitration for the LED ownership lease.

Every lease carries its owner's authoritative in-flight probe plus the
acquiring thread, so "the holder is stranded" is a provable fact instead of
an assumption a contender has to make. Contention is decided on the resource:

- A holder whose probe answers False is reclaimed inside the next acquire
  with the evidence logged.
- A LIVE holder refuses the contender, and the refused operation must refuse
  itself (autofocus aborts its own run) rather than proceed without
  illumination authority.
- The probe is mandatory and must answer True at acquire time, so a
  mis-ordered probe cannot create a window in which a live holder looks dead.
"""

import logging
import threading
from unittest.mock import MagicMock

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    yield s


def _af_runner(scope):
    """Real AutofocusRunner on the real simulated scope; the focus loop is
    stubbed out because these tests only exercise the lease-refusal path,
    which must abort before any focus iteration runs."""
    from modules.autofocus_runner import AutofocusRunner

    r = AutofocusRunner(
        scope=scope,
        camera_executor=MagicMock(),
        io_executor=MagicMock(),
        file_io_executor=MagicMock(),
    )
    r._objective_loader = MagicMock()
    r._objective_loader.get_objective_info.return_value = {
        'AF_range': 50.0,
        'AF_max': 10.0,
        'AF_min': 5.0,
    }
    return r


# ---------------------------------------------------------------------------
# The contract is unmissable: the probe is required and must answer True.
# ---------------------------------------------------------------------------


def test_acquire_without_alive_probe_raises_type_error(scope):
    with pytest.raises(TypeError):
        scope.illumination.acquire_led_lease('x')
    assert scope.illumination.led_lease_owner is None


def test_acquire_with_false_probe_raises_value_error(scope):
    # The probe must already answer True at acquire time: acquiring before
    # setting the in-flight fact would make this holder look stranded (and
    # reclaimable) from the moment it acquired.
    with pytest.raises(ValueError):
        scope.illumination.acquire_led_lease('x', alive=lambda: False)
    assert scope.illumination.led_lease_owner is None


# ---------------------------------------------------------------------------
# Thread identity is NOT liveness evidence: leases are acquired on caller
# threads (UI, scripts) while the work runs on persistent workers, so a dead
# acquiring thread must neither strand a live holder nor be needed to
# reclaim a dead one -- the probe is the sole evidence either way.
# ---------------------------------------------------------------------------


def test_dead_acquiring_thread_does_not_strand_a_live_holder(scope):
    ill = scope.illumination
    holder: dict = {}

    def _acquire_on_worker():
        holder['lease'] = ill.acquire_led_lease('worker', alive=lambda: True)

    t = threading.Thread(target=_acquire_on_worker)
    t.start()
    t.join()
    lease = holder['lease']
    assert lease is not None and lease.held

    contender = ill.acquire_led_lease('next', alive=lambda: True)
    assert contender is None, (
        'a holder whose probe answers True is LIVE even though its acquiring '
        'thread died -- the contender must be refused, not handed a reclaim'
    )
    assert ill.led_lease_owner == 'worker'
    assert lease.held
    lease.release(leave_on=False)


def test_dead_probe_reclaims_regardless_of_thread_state(scope, caplog):
    ill = scope.illumination

    # Acquire with a probe we can flip after the fact.
    probe_state = {'alive': True}
    lease = ill.acquire_led_lease('worker', alive=lambda: probe_state['alive'])
    assert lease is not None and lease.held
    probe_state['alive'] = False

    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        nxt = ill.acquire_led_lease('next', alive=lambda: True)

    assert nxt is not None, 'a dead-probe holder must not lock out the next acquire'
    assert ill.led_lease_owner == 'next'
    assert not lease.held, 'the reclaimed lease must report not held'
    reclaims = [
        r.getMessage() for r in caplog.records if 'reclaimed from stranded owner' in r.getMessage()
    ]
    assert any("'worker'" in m and 'liveness probe returned False' in m for m in reclaims), (
        f'the warning must name the dead owner and the probe evidence; got {reclaims}'
    )
    nxt.release(leave_on=False)


# ---------------------------------------------------------------------------
# Refused autofocus refuses itself: no sweep without illumination authority.
# ---------------------------------------------------------------------------


def test_refused_af_acquire_aborts_the_af_run(scope, monkeypatch):
    """Autofocus that cannot take the LED lease from a LIVE holder must abort
    its own run: AutofocusAborted raised, no focus sweep, no Z walk, camera
    state restored, in-progress flags cleared, the user notified -- and the
    live holder's lease and lit channel are untouched."""
    import modules.autofocus_runner as autofocus_runner_module
    from modules.exceptions import AutofocusAborted

    notified = []
    # error severity, not warning: the likeliest contention (a running
    # protocol) suppresses non-fatal popups, which would swallow this.
    monkeypatch.setattr(
        autofocus_runner_module.notifications,
        'error',
        lambda *args, **kwargs: notified.append(args),
    )

    ill = scope.illumination
    holder = ill.acquire_led_lease('protocol', alive=lambda: True)
    assert holder is not None
    ill.led_on(channel=ill.color2ch('Blue'), illumination_ma=120.0, owner='protocol')

    runner = _af_runner(scope)
    iterations = []
    runner._iterate = lambda: iterations.append(1)

    moves = []
    real_move = scope.motion.move_absolute

    def _recording_move(axis, pos, *args, **kwargs):
        moves.append((axis, pos))
        return real_move(axis, pos, *args, **kwargs)

    monkeypatch.setattr(scope.motion, 'move_absolute', _recording_move)

    pre_z = scope.motion.get_current_position('Z')
    pre_gain = scope.imaging.get_gain_db()
    pre_exposure = scope.imaging.get_exposure_ms()

    with pytest.raises(AutofocusAborted):
        runner.run(
            objective_id='objective-under-test',
            run_trigger_source='manual',
            abort_event=threading.Event(),
            led_color='Green',
            led_illumination=250.0,
        )

    assert iterations == [], 'no focus iteration may run without the LED lease'
    z_targets = [pos for axis, pos in moves if axis == 'Z']
    assert all(abs(pos - pre_z) < 1e-6 for pos in z_targets), (
        f'the stage must not walk the AF z range on a refused run; Z moves: {z_targets}'
    )
    assert scope.imaging.get_gain_db() == pre_gain, 'camera gain must be restored'
    assert scope.imaging.get_exposure_ms() == pre_exposure, 'camera exposure must be restored'
    assert scope.imaging.is_focusing is False
    assert not runner.in_progress(), 'the in-progress flag must clear on the refused run'
    assert any('Autofocus Did Not Start' in str(args) for args in notified), (
        f'the refused run must notify the user; got {notified}'
    )
    # The live holder is undisturbed: lease held, channel still lit, and the
    # AF channel was never lit.
    assert holder.held
    assert ill.led_lease_owner == 'protocol'
    assert ill.led_enabled('Blue'), "the holder's lit channel must survive the refused AF"
    assert not ill.led_enabled('Green'), 'the refused AF must not light its channel'
    holder.release(leave_on=False)
