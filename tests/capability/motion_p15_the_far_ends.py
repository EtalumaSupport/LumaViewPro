"""Motion P15: whether each axis reaches its far limit clear of its far switch.

The travel margin plan (``EL0940_TRAVEL_MARGIN_PLAN_2026-11-02.md``). On a
homed scope with no plate and Z at 0, each of X and Y is jogged toward its
far limit as ``axis_limits`` reports it: 1 mm steps from 10 mm short of
it, then 0.1 mm steps over its last 2 mm. After each step the probe reads the
axis's far (right) limit switch and the board's position. Z is then driven to
its far limit, with X and Y at their middles.

Run at Stage A, when the limit was the stated travel (120 / 80 mm), this
found the far switches (Firmware
``docs/reference/bench/el0940/2026-11-11_far_end_positions.md``). With the
limit a margin inside the travel, every axis should end ``none_by_the_limit``.

An axis's sweep ends at the first of:

- ``far_flag``: the far switch reads engaged after a step;
- ``stall``: a step ends ``'stalled'`` (the far-input stop or contact holds
  the motor short of its target); the position is read, and the stage is
  homed before going on;
- ``fault``: a step ends in any other ``MoveFaultError``; recorded, then
  homed;
- ``aborted``: the operator's Ctrl-C; the stage is stopped and the probe ends;
- ``none_by_the_limit``: the axis reached its far limit clear.

The record, per axis, is the figure ``<axis> far end``: how the sweep ended,
where, and every step's target, board position and switches.

    python tests/capability/motion_p15_the_far_ends.py             # a simulated LS850T
    python tests/capability/motion_p15_the_far_ends.py --hardware  # the connected scope

At the bench (``/hardware-bench-workflow``): watch the stage, and press Ctrl-C
on any sign or sound of contact. Nothing is lit and nothing is saved.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

from _lumascope.modules.exceptions import MoveFaultError
from tests.motion_moves import start_axis_move

# The sweep, um below the axis's far limit: coarse steps from the
# start, 0.1 mm steps over the last stretch.
_START_BELOW_UM = 10_000.0
_FINE_BELOW_UM = 2_000.0
_COARSE_STEP_UM = 1_000.0
_FINE_STEP_UM = 100.0

# How long one wait on a move lasts before the probe looks again, s: short,
# so a Ctrl-C reaches the probe mid-move on every host (a lock's untimed
# wait does not take one on Windows).
_WAIT_SLICE_S = 0.25


class _AbortedError(Exception):
    """The operator stopped the sweep."""


def _targets(far_um):
    """The sweep's targets, um, ascending, the last the far limit itself."""
    fine_from = far_um - _FINE_BELOW_UM
    position = far_um - _START_BELOW_UM
    targets = []
    while position < fine_from:
        targets.append(position)
        position += _COARSE_STEP_UM
    steps = round(_FINE_BELOW_UM / _FINE_STEP_UM)
    targets.extend(fine_from + i * _FINE_STEP_UM for i in range(steps + 1))
    return [round(t, 1) for t in targets]


def _move(m, axis, position):
    """Move ``axis`` to ``position`` and wait for it; a Ctrl-C stops the stage."""
    operation = start_axis_move(m, axis, position)
    try:
        while True:
            try:
                return operation.result(timeout=_WAIT_SLICE_S)
            except TimeoutError:
                continue
    except KeyboardInterrupt:
        m.stop_motion()
        raise _AbortedError from None


def _home(m):
    """Home every axis, then put Z at 0 again."""
    m.home('ALL')
    _move(m, 'Z', 0.0)


def _sweep(m, axis):
    """Jog ``axis`` to its far limit; the record of how it ended."""
    far_um = m.axis_limits(axis).max_um
    steps = []
    record = {'far_limit_um': far_um, 'ended_by': 'none_by_the_limit', 'at_um': None}
    for target in _targets(far_um):
        try:
            _move(m, axis, target)
        except MoveFaultError as e:
            record['ended_by'] = 'stall' if e.reason == 'stalled' else 'fault'
            record['fault_reason'] = e.reason
            record['at_um'] = m.read_actual_position(axis)
            record['switches'] = m.read_limit_switches(axis)
            record['target_um'] = target
            print(
                f'{axis}: {e.reason} at {record["at_um"]} um (target {target}); homing', flush=True
            )
            _home(m)
            break
        board = m.read_actual_position(axis)
        left, right = m.read_limit_switches(axis)
        steps.append({'target_um': target, 'board_um': board, 'left': left, 'right': right})
        print(f'{axis} {target:9.1f} um: board {board} um, switches ({left}, {right})', flush=True)
        if right:
            record['ended_by'] = 'far_flag'
            record['at_um'] = board
            break
    else:
        record['at_um'] = steps[-1]['board_um']
    record['steps'] = steps
    return record


def _measure(session):
    m = session.scope.motion
    limits = {axis: m.axis_limits(axis) for axis in ('X', 'Y', 'Z')}
    figure('limits (um)', {a: (lim.min_um, lim.max_um) for a, lim in limits.items()})
    if HARDWARE:
        input(
            'Remove any plate or vessel from the stage, keep a hand near Ctrl-C, '
            'and press Enter to start the sweep: '
        )
    _move(m, 'Z', 0.0)
    middle = {a: (limits[a].min_um + limits[a].max_um) / 2 for a in ('X', 'Y')}
    _move(m, 'X', middle['X'])
    _move(m, 'Y', middle['Y'])
    for axis in ('X', 'Y'):
        record = figure(f'{axis} far end', _sweep(m, axis))
        print(f'{axis}: ended by {record["ended_by"]} at {record["at_um"]} um', flush=True)
        # The axis goes back to its middle before the other sweeps, so the
        # stage never stands at a far corner.
        _move(m, axis, middle[axis])
        check(
            f'{axis}: the stage knows its position after the sweep',
            axis not in m.axes_without_position(),
        )
        check(f'{axis}: reached its far limit clear', record['ended_by'] == 'none_by_the_limit')
    z_far = limits['Z'].max_um
    _move(m, 'Z', z_far)
    z_record = figure(
        'Z far end',
        {
            'far_limit_um': z_far,
            'board_um': m.read_actual_position('Z'),
            'switches': m.read_limit_switches('Z'),
        },
    )
    _move(m, 'Z', 0.0)
    check('Z: reached its far limit clear', z_record['switches'][1] == 0)


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
        return
    session, _live = make_session('probe_far_ends', home=True, microscope='LS850T')
    try:
        _measure(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except _AbortedError:
        print('ABORTED by the operator: the stage was stopped.', flush=True)
        check('the sweep ran to its end', False, 'aborted')
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
