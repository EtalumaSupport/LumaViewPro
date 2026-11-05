"""Autofocus P02: Autofocus All Steps on a centre stack focuses it once, around its reference slice.

#829's bench check. A z-stack layer with autofocus focuses once per stack
group: the sweep is centred on the stack's reference slice (the middle slice
of a centre stack, the slice built at the user's focus), and the found focus
lands on that slice, with every other slice at its built distance from it.
The field case put the focus on the bottom slice.

One step at the given position is stacked through ``ScopeSession.apply_zstacking``
(3 slices, centre), then ``ProtocolRunner.start_autofocus_all_steps`` runs it.
The sweep's centre and its found focus are read from the autofocus log's
``AF START`` and ``AF DONE`` lines; the ladder is the slices' Z the run wrote
back into the protocol.

    python tests/capability/autofocus_p02_a_stack_sweeps_around_its_reference.py   # simulator
    python tests/capability/autofocus_p02_a_stack_sweeps_around_its_reference.py --hardware \\
        --z <focus um> [--x <mm> --y <mm>] [--layer BF] [--step 50]

On hardware the stage homes at bring-up, so the focus is passed as ``--z``
(the Z the GUI shows on a focused sample); X and Y default to where bring-up
leaves the stage, given as ``--x`` / ``--y`` in plate mm to put a sample
under it. The layer's LED is lit while the sweep runs.
"""

import argparse
import itertools
import logging
import re
import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report

RUN_TIMEOUT_S = 600.0
# The AF log prints the centre to 0.1 um and the best to 0.01 um.
CENTRE_TOLERANCE_UM = 0.06
LADDER_TOLERANCE_UM = 0.01
SIM_STACK_Z_UM = 5010.0

_START = re.compile(r'AF START .*center=(-?[\d.]+)')
_DONE = re.compile(r'AF DONE best=(-?[\d.]+)um')


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--layer', default='BF')
    parser.add_argument('--step', type=float, default=50.0, help='slice spacing, um')
    parser.add_argument(
        '--z', type=float, default=None, help='the focus, um (required on hardware)'
    )
    parser.add_argument('--x', type=float, default=None, help='plate X, mm')
    parser.add_argument('--y', type=float, default=None, help='plate Y, mm')
    args = parser.parse_args()
    if HARDWARE and args.z is None:
        print('REFUSED: --z <focus um> is required on hardware: bring-up homes Z.', flush=True)
        sys.exit(2)
    return args


class _AutofocusLines(logging.Handler):
    """The AF START centres and AF DONE bests, in the order the run logged them."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.centres = []
        self.bests = []

    def emit(self, record):
        text = record.getMessage()
        if m := _START.search(text):
            self.centres.append(float(m.group(1)))
        if m := _DONE.search(text):
            self.bests.append(float(m.group(1)))


def _ladder(protocol):
    """(slice index, Z) for each step, in slice order."""
    steps = protocol.steps()
    return sorted(zip(steps['Z-Slice'].astype(int), steps['Z'].astype(float), strict=True))


def main():
    args = _arguments()
    if HARDWARE:
        session_cm = hardware_session()
    else:
        session_cm = headless_session(probe_dir('af_stack_reference'), acquiring=(args.layer,))

    lines = _AutofocusLines()
    af_log = logging.getLogger('LVP.autofocus')
    af_log.addHandler(lines)
    try:
        with session_cm as (session, runner):
            from _lumascope.modules.coord_transformations import PlateLocation
            from _lumascope.modules.zstack_config import ZStackReference

            live = session.get_current_plate_position()
            z = args.z
            if z is None:
                # The simulated sample is in focus at Z 5000 um (its bring-up
                # log says so); the stack is built a little off it, so the
                # sweep has a focus to find that is not its own centre.
                z = SIM_STACK_Z_UM
            at = PlateLocation(
                x_mm=live.x_mm if args.x is None else args.x,
                y_mm=live.y_mm if args.y is None else args.y,
                z_um=z,
            )
            figure('position', f'x={at.x_mm} mm y={at.y_mm} mm z={at.z_um} um')
            figure('layer', args.layer)

            protocol = session.create_empty_protocol()
            session.add_step(protocol, channels=[args.layer], at=at)
            session.apply_zstacking(
                protocol,
                range_um=2 * args.step,
                step_size_um=args.step,
                z_reference=ZStackReference.CENTER,
            )
            before = _ladder(protocol)
            figure('slices before, um', [z for _, z in before])
            if not check('the stack has 3 slices', len(before) == 3, f'{len(before)}'):
                return
            reference_before = before[1][1]
            figure('reference slice Z before, um', reference_before)

            outcome = runner.start_autofocus_all_steps(protocol).result(timeout=RUN_TIMEOUT_S)
            figure('outcome', (outcome.status, outcome.reason))
            check('the focus was written into the protocol', outcome.focus_written is True)
            # A step whose sweep found no focus is not a failed run: the scan
            # leaves that step's Z as it was and names it here.
            check('every step focused', not outcome.focus_failures, str(outcome.focus_failures))

            after = [z for _, z in _ladder(protocol)]
            figure('slices after, um', after)
            figure('AF START centres, um', lines.centres)
            figure('AF DONE bests, um', lines.bests)
            if not check('one sweep for the stack', len(lines.bests) == 1 == len(lines.centres)):
                return
            centre, best = lines.centres[0], lines.bests[0]
            check(
                'the sweep was centred on the reference slice',
                abs(centre - reference_before) <= CENTRE_TOLERANCE_UM,
                f'centre {centre} vs reference {reference_before}',
            )
            # Plan section 8's carried item: the field sweep's centre against
            # the file's slice 0.
            figure('sweep centre - slice 0 Z before, um', round(centre - before[0][1], 2))

            figure('expected, um', [round(best - args.step, 2), best, round(best + args.step, 2)])
            check(
                'the found focus is on the reference slice',
                abs(after[1] - best) <= LADDER_TOLERANCE_UM,
                f'{after[1]} vs best {best}',
            )
            spacing = [round(b - a, 2) for a, b in itertools.pairwise(after)]
            figure('spacing after, um', spacing)
            check(
                'every slice is at its built distance from the reference',
                all(abs(s - args.step) <= LADDER_TOLERANCE_UM for s in spacing),
                f'{spacing} vs {args.step}',
            )
    finally:
        af_log.removeHandler(lines)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
