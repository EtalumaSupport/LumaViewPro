"""Motion P13: a move arrives where the board says it is, a later move supersedes it, and the overshoot leg's time.

The arrival plan's bench row. Through the motion API, on a homed scope:

1. The overshoot leg at its longest: Z from the top of its travel down to
   the lowest target that still takes the leg (just above the bottom by the
   board's backlash), overshoot on. The leg is bounded by
   ``OVERSHOOT_LEG_TIMEOUT_S``, derived on the simulator; this measures the
   move whole, and the leg itself is read off ``serial.log`` (the time
   between the leg's ``TARGET_WZ`` and the target's).
2. X, Y and Z moves, each waited: the position the API holds against a
   fresh board read (``get_actual_position``).
3. A retarget mid-move on X: the first move ends ``'superseded'``, the
   second arrives at its own target.
4. The same on Z with overshoot, the retarget sent while the first move is
   on its leg.

    python tests/capability/motion_p13_arrival.py             # a simulated LS850T
    python tests/capability/motion_p13_arrival.py --hardware  # the connected scope

Z travels its whole configured range; nothing is lit and nothing is saved.
On the simulator the probe runs on the realistic tier, whose moves take time.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

from modules.exceptions import MoveStoppedError
from tests.motion_moves import axis_move, start_axis_move

# How far below the top, and above the lowest leg target, the full-travel
# move starts and ends, um: inside the travel, so no limit refuses it.
_MARGIN_UM = 50.0


def _timed_move(m, axis, position, **kwargs):
    start = time.perf_counter()
    axis_move(m, axis, position, **kwargs)
    return round(time.perf_counter() - start, 3)


def _held_against_the_board(m, axis):
    held = m.get_current_position(axis)
    board = m.get_actual_position(axis)
    figure(f'{axis} held vs board (um)', {'held': held, 'board': board})
    check(
        f"{axis}: the position the API holds is the board's", abs(held - board) < 1.0, (held, board)
    )


def _retarget(m, axis, first, second, **kwargs):
    # The second is started once the board has taken the first, while the
    # first still has its axis.
    one = start_axis_move(m, axis, first, **kwargs)
    check(f'{axis}: the first move is still under way at the retarget', not one.done())
    two = start_axis_move(m, axis, second, **kwargs)
    try:
        one.result()
        check(f'{axis}: the first move is superseded', False, 'it arrived')
    except MoveStoppedError as e:
        check(f'{axis}: the first move is superseded', e.reason == 'superseded', e.reason)
    check(f'{axis}: the second move arrives', two.result() is None)
    landed = m.get_actual_position(axis)
    figure(f'{axis} retarget landed (um)', {'target': second, 'board': landed})
    check(f'{axis}: the board is at the second target', abs(landed - second) < 1.0, landed)


def _measure(session):
    m = session.scope.motion
    z = m.get_axis_limits('Z')
    backlash = m._driver.backlash_um()
    figure('Z limits (um)', (z.min_um, z.max_um))
    figure('Z backlash (um)', backlash)

    top = z.max_um - _MARGIN_UM
    lowest_with_leg = backlash + 50 + _MARGIN_UM
    figure('full-travel move (um)', (top, lowest_with_leg))
    figure('move to the top, s', _timed_move(m, 'Z', top))
    figure(
        'full-travel downward move with overshoot, s',
        _timed_move(m, 'Z', lowest_with_leg, overshoot_enabled=True),
    )
    check('Z arrived at the bottom target', abs(m.get_actual_position('Z') - lowest_with_leg) < 1.0)

    x, y = m.get_axis_limits('X'), m.get_axis_limits('Y')
    x_mid, y_mid = (x.min_um + x.max_um) / 2, (y.min_um + y.max_um) / 2
    for axis, position in (('X', x_mid), ('Y', y_mid), ('Z', (z.min_um + z.max_um) / 2)):
        axis_move(m, axis, position)
        _held_against_the_board(m, axis)

    _retarget(m, 'X', x.max_um - _MARGIN_UM, x_mid + 2000.0)
    # Up first, so the retarget's first move is a downward one with a leg.
    m.move_z(top)
    _retarget(m, 'Z', lowest_with_leg, (z.min_um + z.max_um) / 2, overshoot_enabled=True)


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
        return
    # The realistic tier: its moves take time, so a retarget lands mid-move.
    session, _live = make_session(
        'probe_arrival', home=True, microscope='LS850T', simulator_tier='realistic'
    )
    try:
        _measure(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
