"""Autofocus P01: a standalone autofocus hands back the light it found, stopped or finished.

A person watching a lit field clicks Autofocus; when the autofocus ends,
the field is lit as it was. The standalone autofocus is a run whose end
policy is to return the LEDs to their pre-run state, and the abort path once
darkened the channel anyway. Two runs, each started with the layer lit at
its saved current: one stopped partway through its sweep through the run's
handle, one left to finish. After each, the layer is read back from the
illumination API.

    python tests/capability/autofocus_p01_ends_with_the_light_it_found.py             # a simulated LS850T
    python tests/capability/autofocus_p01_ends_with_the_light_it_found.py --hardware  # the connected scope

On hardware the stage homes at bring-up, Z moves to the middle of its
travel, and Z sweeps twice; the layer's LED is lit throughout.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report
from tests.scope_fakes import channel_reading

LAYER = sys.argv[sys.argv.index('--layer') + 1] if '--layer' in sys.argv else 'BF'
RUN_TIMEOUT_S = 180.0


def _autofocus(session, runner, current_ma, *, stop):
    illumination = session.scope.illumination
    illumination.led_on(LAYER, current_ma)
    before = channel_reading(illumination, LAYER)
    leg = 'stopped' if stop else 'finished'
    figure(f'{leg}: {LAYER} before', before)

    handle = runner.start_autofocus(channel=LAYER)
    if stop:
        # Stopped inside the sweep, not during the run's setup or its move to
        # the step: half a second after the autofocus machinery says it is
        # focusing. No L2 member states that; `is_focusing` is the internal
        # flag the autofocus runner sets, read here only to time the stop.
        imaging = session.scope.imaging
        motion = session.scope.motion
        deadline = time.monotonic() + RUN_TIMEOUT_S
        while handle.progress().is_live and not imaging.is_focusing and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.5)
        check(f'{leg}: the autofocus was sweeping when stopped', imaging.is_focusing)
        figure(f'{leg}: Z position at stop um', motion.get_current_position('Z'))
        if check(f'{leg}: the run was live when stopped', handle.progress().is_live):
            handle.stop()
    outcome = handle.result(timeout=RUN_TIMEOUT_S)
    figure(f'{leg}: outcome', None if outcome is None else (outcome.status, outcome.reason))
    check(f'{leg}: the run ended', outcome is not None)

    after = channel_reading(illumination, LAYER)
    figure(f'{leg}: {LAYER} after', after)
    check(f'{leg}: {LAYER} is lit at its pre-run current', after == before, f'{before} -> {after}')


def main():
    if HARDWARE:
        session_cm = hardware_session()
    else:
        from contextlib import contextmanager

        @contextmanager
        def _sim():
            from tests.scope_fakes import TEST_TURRET_OBJECTIVES

            session, _live = make_session(
                'probe_af',
                home=True,
                microscope='LS850T',
                turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            )
            # The hardware bring-up turns the turret to slot 1; the simulated
            # session's home leaves it in no known slot.
            session.scope.motion.home('T')
            session.scope.motion.move_turret(1)
            # Bench-paced Z, so the stopped leg's stop lands inside the sweep.
            session.scope._motion_driver.set_timing_mode('realistic')
            try:
                yield session, session.create_protocol_runner()
            finally:
                session.shutdown()

        session_cm = _sim()

    with session_cm as (session, runner):
        # Bring-up leaves Z at its home, the bottom of travel, where every
        # autofocus window reaches past the floor and the run is refused
        # before it sweeps. The middle of travel has room for any
        # objective's window.
        motion = session.scope.motion
        z_limits = motion.get_axis_limits('Z')
        motion.move_z((z_limits.min_um + z_limits.max_um) / 2)
        figure('Z start um', motion.get_current_position('Z'))
        current_ma = float(session.settings[LAYER]['illumination_ma'])
        figure(f'{LAYER} saved current mA', current_ma)
        try:
            _autofocus(session, runner, current_ma, stop=True)
            _autofocus(session, runner, current_ma, stop=False)
        finally:
            session.scope.illumination.leds_off()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
