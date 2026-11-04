"""Five autofocuses on one field land within the final pass's step.

The final pass places the focus by a Gaussian on a floor fitted to that
pass's own points (``autofocus_runner``'s final fit): on one unmoved field,
five autofocuses started from one Z agree within the final pass's step,
and each run's final pass sweeps its whole window (7 points at the 40x's
1 um over its +/-3 um) and logs its fit. A fit that cannot run or does not
converge falls back to the raw maximum and says so in its FINAL FIT line,
which this probe counts as a failure of that run's fit.

Each run goes through the member the autofocus button reaches,
``ProtocolRunner.start_autofocus``, on the stored BF settings, and starts
from the same Z so the five are comparable (the member leaves the stage at
the focus it found). The autofocus log lines are collected from the
``LVP.autofocus`` logger while each run is in flight: the pass and fit
lines are the run's record of its passes, the product this reads.

    python tests/capability/autofocus_p03_five_on_one_field.py
    python tests/capability/autofocus_p03_five_on_one_field.py --hardware --z 7117
"""

import argparse
import logging
import re
import sys

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report

_RUNS = 5
_PASS = re.compile(r'PASS (\d+) complete: (\d+) pts, step=([\d.]+)')
_FIT = re.compile(r'FINAL FIT: (.*)')


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--z', type=float, default=None, help='the Z each run starts from, um')
    parser.add_argument('--layer', default='BF')
    return parser.parse_args()


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _one(runner, motion, layer, start_z, index):
    motion.move_z(start_z)
    lines = _Lines()
    logger = logging.getLogger('LVP.autofocus')
    logger.addHandler(lines)
    try:
        outcome = runner.start_autofocus(channel=layer).result(timeout=300)
    finally:
        logger.removeHandler(lines)
    focus = motion.get_current_position('Z')
    name = f'run{index}'
    check(f'{name}: completed', outcome.status == 'completed', f'{outcome.status} {outcome.reason}')
    passes = [tuple(float(g) for g in m.groups()) for m in map(_PASS.search, lines.lines) if m]
    fits = [m.group(1) for m in map(_FIT.search, lines.lines) if m]
    figure(f'{name}.focus_um', round(focus, 3))
    figure(f'{name}.passes', [(int(n), int(p), s) for n, p, s in passes])
    figure(f'{name}.final_fit', fits)
    check(
        f'{name}: the final pass swept 7 points',
        bool(passes) and passes[-1][1] == 7,
        str(passes[-1] if passes else 'no pass logged'),
    )
    check(
        f'{name}: the final fit ran and converged',
        len(fits) == 1 and 'too few' not in fits[0] and 'did not converge' not in fits[0],
        str(fits),
    )
    return focus, (passes[-1][2] if passes else None)


def main():
    args = _arguments()
    live = probe_dir('autofocus_five')
    session_cm = hardware_session() if HARDWARE else headless_session(live, acquiring=(args.layer,))
    with session_cm as (session, runner):
        motion = session.scope.motion
        if not HARDWARE:
            # Simulated, the camera shows a focus target focused inside the
            # sweep's window; on hardware the specimen is the scene.
            camera = session.scope._camera_driver
            camera.set_test_pattern(enabled=True, pattern='focus_target')
            camera.set_focal_z(3030.0)
        start_z = (
            args.z
            if args.z is not None
            else (session.get_settings_snapshot()[args.layer]['focus'] if HARDWARE else 3000.0)
        )
        figure('start_z_um', start_z)
        figure('objective', session.get_settings_snapshot()['objective_id'])
        results = [_one(runner, motion, args.layer, start_z, i + 1) for i in range(_RUNS)]
        foci = [f for f, _ in results]
        steps = {s for _, s in results if s is not None}
        spread = max(foci) - min(foci)
        figure('foci_um', [round(f, 3) for f in foci])
        figure('spread_um', round(spread, 3))
        figure('final_step_um', sorted(steps))
        check('every final pass used one step', len(steps) == 1, str(sorted(steps)))
        if len(steps) == 1:
            step = steps.pop()
            check(
                'the five foci lie within the final step',
                spread <= step,
                f'spread {spread:.3f} um against step {step:.2f} um',
            )
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
