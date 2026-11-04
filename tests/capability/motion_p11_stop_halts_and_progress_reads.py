"""Motion P11: a Stop halts the stage on this firmware, and how long a moving axis's position holds still.

Two measurements the stall-bound plan (Firmware docs/API_CONVENTION_W2B_PLAN_2026-10-14.md,
section 9) gates its pushes on:

1. A Stop partway through an X move leaves X short of its target, at
   rest, on whatever firmware the board runs. On the field firmware, which
   has no STOP command, the driver halts each axis by writing its actual
   position as its target; the stage decelerates past where it was read and
   drives back. The probe reads X's actual position after the Stop until it
   settles and reports how far it ran past and where it came to rest.
2. While an axis moves, the longest span its actual microstep reading holds
   one value before the move ends. A move is judged stalled after 2 s with
   no change, so every span here must be well under that. X and Y moves
   and, on a scope with a turret, one slot change. A hold comes from a
   move's ramps and its slowest speed, which every move has, so the moves
   are short.

The readings are the motion monitor's own: each read the monitor makes of an
axis's actual position (the board's ACTUAL_R register) is recorded with its
time, so the probe adds no traffic to the board while it measures.

    python tests/capability/motion_p11_stop_halts_and_progress_reads.py             # a simulated LS850T
    python tests/capability/motion_p11_stop_halts_and_progress_reads.py --hardware  # the connected scope

On hardware the stage homes at bring-up; then Z goes to 0, its lowest, and
stays there for every X and Y move, which stay within SPAN_UM of where the
home left the stage, nowhere near an end: an objective raised while the
stage crosses its travel runs into the stage (an LS850T, 2026-10-14). A move
that has not arrived within MOVE_TIMEOUT_S is stopped at once. Z itself is
not moved here; its holds are the simulator's.
"""

import contextlib
import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report
from tests.motion_moves import start_axis_move

STALL_WINDOW_S = 2.0
MOVE_TIMEOUT_S = 5.0
SPAN_UM = 20000.0
# Where the home left X and Y; every move stays within SPAN_UM of it.
CENTRE: dict[str, float] = {}
SETTLE_READ_S = 2.0


@contextlib.contextmanager
def _actual_reads(driver):
    """Every ACTUAL_R read the driver makes, as (monotonic time, axis, microsteps)."""
    reads = []
    lock = threading.Lock()
    real = driver._read_register

    def recorded(register, axis, *args, **kwargs):
        value = real(register, axis, *args, **kwargs)
        if register == 'ACTUAL_R':
            with lock:
                reads.append((time.monotonic(), axis, value))
        return value

    driver._read_register = recorded
    try:
        yield reads
    finally:
        driver._read_register = real


def _longest_hold_s(samples):
    """The longest time one reading held before the reading the move ended on."""
    if len(samples) < 2:
        return 0.0
    final = samples[-1][1]
    end = next(i for i, (_t, v) in enumerate(samples) if v == final)
    longest, start = 0.0, 0
    for i in range(1, end + 1):
        if samples[i][1] != samples[start][1]:
            longest = max(longest, samples[i][0] - samples[start][0])
            start = i
    return longest


def _refuse_unless_z_is_down(motion, axis):
    z = motion.get_current_position('Z')
    if axis in ('X', 'Y') and not (z is not None and abs(z) < 1.0):
        raise RuntimeError(f'refusing to move {axis}: Z is at {z}, not 0')


def _arrive(motion, handle, timeout_s=MOVE_TIMEOUT_S):
    """Wait for a move; one that has not arrived in ``timeout_s`` is stopped at once."""
    try:
        return handle.result(timeout=timeout_s)
    except TimeoutError:
        motion.stop_motion()
        raise


def _hold_while_moving(motion, reads, axis, target):
    """Drive ``axis`` to ``target`` and return the longest hold of its reading on the way."""
    _refuse_unless_z_is_down(motion, axis)
    start = time.monotonic()
    _arrive(motion, start_axis_move(motion, axis, target))
    took = time.monotonic() - start
    samples = [(t, v) for t, a, v in list(reads) if a == axis and t >= start]
    return _longest_hold_s(samples), took, len(samples)


def _progress_holds(session, reads):
    motion = session.scope.motion
    for axis in ('X', 'Y'):
        if motion.get_axis_limits(axis) is None:
            continue
        centre = CENTRE[axis]
        for target in (centre - SPAN_UM, centre + SPAN_UM, centre):
            hold, took, n = _hold_while_moving(motion, reads, axis, target)
            figure(
                f'{axis} to {target:.0f}: longest hold s / move s / reads',
                (round(hold, 3), round(took, 2), n),
            )
            check(
                f'{axis} to {target:.0f}: the reading never held near the stall window',
                hold < STALL_WINDOW_S / 2,
                f'{hold:.3f} s',
            )
    slot = motion.turret_status().slot
    if slot is not None:
        start = time.monotonic()
        other = 2 if slot == 1 else 1
        motion.move_turret(other)
        took = time.monotonic() - start
        samples = [(t, v) for t, a, v in list(reads) if a == 'T' and t >= start]
        hold = _longest_hold_s(samples)
        figure(
            f'turret {slot} to {other}: longest hold s / change s / reads',
            (round(hold, 3), round(took, 2), len(samples)),
        )
        check(
            'the turret reading never held near the stall window',
            hold < STALL_WINDOW_S / 2,
            f'{hold:.3f} s',
        )


def _stop_mid_move(session):
    motion = session.scope.motion
    driver = motion._driver
    figure('board answers STOP', driver.supports_motor_stop())
    low, target = CENTRE['X'] - SPAN_UM, CENTRE['X'] + SPAN_UM
    _refuse_unless_z_is_down(motion, 'X')
    _arrive(motion, motion.start_move_xy(x_um=low))
    part_way = low + 0.3 * (target - low)
    reached = threading.Event()

    def seen(axis, position):
        if axis == 'X' and position.position_um is not None and position.position_um > part_way:
            reached.set()

    motion.add_position_listener(seen)
    try:
        handle = motion.start_move_xy(x_um=target)
        if not check('X got part of the way', reached.wait(MOVE_TIMEOUT_S)):
            motion.stop_motion()
            raise RuntimeError('X never got part of the way; stopped')
        asked_at = driver.current_pos('X')
        motion.stop_motion()
    finally:
        motion.remove_position_listener(seen)
    try:
        handle.result(timeout=MOVE_TIMEOUT_S)
        ended = 'arrived'
    except Exception as e:
        ended = getattr(e, 'reason', type(e).__name__)
    trace = []
    until = time.monotonic() + SETTLE_READ_S
    while time.monotonic() < until:
        trace.append(driver.current_pos('X'))
        time.sleep(0.02)
    rest = trace[-1]
    figure('X when the Stop was asked um', round(asked_at, 1))
    figure('X furthest past it um', round(max(trace) - asked_at, 1))
    figure('X at rest um', round(rest, 1))
    figure('the move ended', ended)
    check('the move ended stopped', ended == 'stopped', ended)
    check(
        'X came to rest short of its target', rest < target - 1000.0, f'{rest:.0f} of {target:.0f}'
    )
    check('X is at rest', abs(trace[-1] - trace[-10]) < 1.0, f'{trace[-10]:.1f} -> {trace[-1]:.1f}')


def main():
    if HARDWARE:
        session_cm = hardware_session()
    else:

        @contextlib.contextmanager
        def _sim():
            session, _live = make_session(
                'probe_stop', home=True, microscope='LS850T', simulator_tier='realistic'
            )
            try:
                yield session, None
            finally:
                session.shutdown()

        session_cm = _sim()

    with session_cm as (session, _runner):
        motion = session.scope.motion
        _arrive(motion, motion.start_move_z(0.0), timeout_s=30.0)
        check('Z is at 0 before any X or Y move', abs(motion.get_current_position('Z')) < 1.0)
        CENTRE.update({a: motion.get_current_position(a) for a in ('X', 'Y')})
        figure('X, Y where the home left them um', (round(CENTRE['X']), round(CENTRE['Y'])))
        with _actual_reads(session.scope.motion._driver) as reads:
            _progress_holds(session, reads)
        _stop_mid_move(session)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
