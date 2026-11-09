"""Motion P14: a run stopped mid-move ends with the stage at rest, and how long that takes.

A person presses Stop while a run's move is in flight. The run's cleanup
waits for every axis to read stopped before the run ends
(`protocol_cleanup.run_cleanup`, "Wait for the stage to stop"), so a caller
told the run is over can start the next at once. That wait's bound,
PERFORMANCE_BUDGETS.md row `run_end_stage_settle_wait_s`, was set from no
bench reading; this probe is that reading, on X during a scan's move and on
Z inside an autofocus sweep.

Three times are read from the Stop, which is the run's own `stop()`, the
call the GUI's Stop button makes:

1. The motor at rest: the last change of the axis's actual position (the
   board's ACTUAL_R, read by the motion monitor itself, so the probe adds
   no traffic to the board). On field firmware, which has no STOP command,
   the stage decelerates past where it was and drives back, so this is
   later than the STOP itself.
2. The API reading the stage stopped: the first time after the Stop that
   `motion.is_any_axis_moving()` (an in-memory read) answers False, and
   every change of it until the run ends. A scan's return to its start
   moves the stage again after the halt, so the scan leg can read moving,
   stopped and moving again.
3. The run's end: when the run's handle answers.

The checks: the axis was moving when the Stop was asked, the run ended, no
axis reads moving when it did, and the axis's position did not change after
it.

    python tests/capability/motion_p14_a_stopped_run_ends_with_the_stage_at_rest.py             # a simulated LS850T
    python tests/capability/motion_p14_a_stopped_run_ends_with_the_stage_at_rest.py --hardware  # the connected scope

On hardware the stage homes at bring-up. The scan leg moves Z to 0, its
lowest, then runs one BF step SPAN_MM to one side of where the home left the
stage and one SPAN_MM to the other, at that Z, and returns to the start when
it ends, as a scan started from the GUI does; the Stop lands on the first
move. The autofocus leg moves Z to the middle of its travel, as
autofocus_p01 does, and is stopped inside its sweep. The BF LED is lit by
each run.
"""

import contextlib
import sys
import threading
import time
import traceback

from harness import HARDWARE, SCRATCH, check, figure, hardware_session, make_session, report
from motion_p11_stop_halts_and_progress_reads import (
    _actual_reads,
    _arrive,
    _refuse_unless_z_is_down,
)

from _lumascope.modules.coord_transformations import PlateLocation

LAYER = 'BF'
# Within P11's SPAN_UM of where the home left the stage, nowhere near an end.
SPAN_MM = 15.0
RUN_TIMEOUT_S = 180.0
# How far into the scan's first move the Stop is asked, once the stage reads
# moving; that move is SPAN_MM long, so 0.3 s lands well inside it.
SCAN_INTO_THE_MOVE_S = 0.3
# How long the readings run on after the run's end, to see nothing moves.
AFTER_THE_END_S = 1.0
SAMPLE_S = 0.005


@contextlib.contextmanager
def _moving_flag(motion):
    """Every change of `is_any_axis_moving()`, as (monotonic time, value), sampled every SAMPLE_S."""
    changes = []
    done = threading.Event()

    def sample():
        last = None
        while not done.is_set():
            value = motion.is_any_axis_moving()
            if value != last:
                changes.append((time.monotonic(), value))
                last = value
            done.wait(SAMPLE_S)

    thread = threading.Thread(target=sample, name='p14-moving-flag', daemon=True)
    thread.start()
    try:
        yield changes
    finally:
        done.set()
        thread.join(timeout=1.0)


def _last_change(samples, until):
    """The time of the first read of the value `samples` held from then to `until`."""
    held = [(t, v) for t, v in samples if t <= until]
    if not held:
        return None
    final = held[-1][1]
    at = held[-1][0]
    for t, v in reversed(held):
        if v != final:
            break
        at = t
    return at


def _stop_and_read(session, handle, axis, leg, ready, into_the_move_s):
    """Stop `handle` `into_the_move_s` after `ready()` says its move is in flight; figure the three times."""
    motion = session.scope.motion
    driver = motion._driver
    with _actual_reads(driver) as reads, _moving_flag(motion) as flag:
        deadline = time.monotonic() + RUN_TIMEOUT_S
        while handle.progress().is_live and not ready() and time.monotonic() < deadline:
            time.sleep(SAMPLE_S)
        time.sleep(into_the_move_s)
        moving = motion.is_any_axis_moving()
        figure(f'{leg}: {axis} at the Stop um', motion.get_current_position(axis))
        if not check(f'{leg}: the stage was moving when the Stop was asked', moving):
            handle.stop()
            handle.result(timeout=RUN_TIMEOUT_S)
            return
        if not check(f'{leg}: the run was live when the Stop was asked', handle.progress().is_live):
            return
        stopped_at = time.monotonic()
        handle.stop()
        outcome = handle.result(timeout=RUN_TIMEOUT_S)
        ended_at = time.monotonic()
        moving_at_end = motion.is_any_axis_moving()
        time.sleep(AFTER_THE_END_S)
        after = time.monotonic()

    axis_reads = [(t, v) for t, a, v in list(reads) if a == axis and t >= stopped_at]
    rest_at = _last_change(axis_reads, after)
    idle = [t for t, v in flag if t >= stopped_at and v is False]
    figure(f'{leg}: outcome', None if outcome is None else (outcome.status, outcome.reason))
    figure(
        f'{leg}: the moving flag after the Stop, s from it',
        [(round(t - stopped_at, 3), v) for t, v in flag if t >= stopped_at],
    )
    figure(
        f'{leg}: Stop to the API reading stopped s',
        round(idle[0] - stopped_at, 3) if idle else None,
    )
    figure(
        f'{leg}: Stop to the {axis} motor at rest s',
        round(rest_at - stopped_at, 3) if rest_at else None,
    )
    figure(f'{leg}: Stop to the run end s', round(ended_at - stopped_at, 3))
    figure(f'{leg}: {axis} reads after the Stop', len(axis_reads))
    if axis_reads:
        # ACTUAL_R is in microsteps; the driver's own conversion gives um.
        figure(
            f'{leg}: {axis} at rest um', round(driver.steps_to_position(axis, axis_reads[-1][1]), 1)
        )
    check(f'{leg}: the run ended', outcome is not None)
    check(f'{leg}: no axis reads moving when the run ends', not moving_at_end)
    check(
        f'{leg}: {axis} did not move after the run ended',
        rest_at is not None and rest_at <= ended_at,
        f'last change {None if rest_at is None else round(rest_at - ended_at, 3)} s after the end',
    )


def _scan_leg(session, runner):
    """A two-step scan across X, stopped on its first move."""
    motion = session.scope.motion
    _arrive(motion, motion.start_move_z(0.0), timeout_s=30.0)
    _refuse_unless_z_is_down(motion, 'X')
    start = session.get_current_plate_position()
    figure('scan: start (plate mm, z um)', start)

    import _lumascope.modules.config_helpers as config_helpers

    # The protocol's plate is the session's, so the steps' plate positions
    # are the live position's frame; its own steps are replaced by two.
    cfg = config_helpers.get_sequenced_capture_config_from_settings(
        session.settings,
        objective_helper=session.scope.objective_helper,
        wellplate_loader=session.scope.wellplate_loader,
        current_z=start.z_um,
        tiling='1x1',
        use_zstacking=False,
    )
    cfg['channel_configs'] = {LAYER: cfg['channel_configs'][LAYER]}
    cfg['channel_configs'][LAYER]['acquire'] = 'image'
    protocols = session.scope.protocols
    protocol = protocols.create_protocol(input_config=cfg)
    while protocol.num_steps() > 0:
        protocols.delete_step(protocol, 0)
    for dx in (SPAN_MM, -SPAN_MM):
        at = PlateLocation(x_mm=start.x_mm + dx, y_mm=start.y_mm, z_um=start.z_um)
        session.add_step(protocol, channels=[LAYER], at=at)
    figure('scan: steps (plate X mm)', list(protocol.steps()['X']))

    handle = runner.start_single_scan(
        protocol,
        sequence_name='p14_scan',
        parent_dir=str(SCRATCH / 'p14_runs'),
        return_to_position=start,
    )
    _stop_and_read(session, handle, 'X', 'scan', motion.is_any_axis_moving, SCAN_INTO_THE_MOVE_S)
    figure('scan: X, Y back at the start (plate mm)', session.get_current_plate_position())


def _autofocus_leg(session, runner):
    """A standalone autofocus, stopped inside its sweep."""
    motion = session.scope.motion
    z_limits = motion.get_axis_limits('Z')
    _arrive(motion, motion.start_move_z((z_limits.min_um + z_limits.max_um) / 2), timeout_s=30.0)
    figure('autofocus: Z start um', motion.get_current_position('Z'))
    handle = runner.start_autofocus(channel=LAYER)
    imaging = session.scope.imaging
    # `is_focusing` is the internal flag the autofocus runner sets, read here
    # only to time the Stop into the sweep, as autofocus_p01 does; no L2
    # member states it. The sweep's Z steps are short and the stage holds
    # still for a frame between them, so the Stop is asked the moment a step
    # is seen moving.
    _stop_and_read(
        session,
        handle,
        'Z',
        'autofocus',
        lambda: imaging.is_focusing and motion.is_any_axis_moving(),
        0.0,
    )


def main():
    if HARDWARE:
        session_cm = hardware_session()
    else:

        @contextlib.contextmanager
        def _sim():
            from tests.scope_fakes import TEST_TURRET_OBJECTIVES

            session, _live = make_session(
                'probe_run_stop',
                home=True,
                microscope='LS850T',
                simulator_tier='realistic',
                turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            )
            # The hardware bring-up turns the turret to slot 1; the simulated
            # session's home leaves it in no known slot.
            session.scope.motion.home('T')
            session.scope.motion.move_turret(1)
            try:
                yield session, session.protocol_runner()
            finally:
                session.shutdown()

        session_cm = _sim()

    with session_cm as (session, runner):
        figure('board answers STOP', session.scope.motion._driver.supports_motor_stop())
        try:
            _scan_leg(session, runner)
            _autofocus_leg(session, runner)
        finally:
            session.scope.illumination.leds_off()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
