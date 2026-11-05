# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus sweeps every target of every pass, its final pass seven fine steps.

Each pass's targets are counted from its start (``SweepPass.targets``),
never by adding the step to the stage's last target: that target carries
the motor's microstep rounding, and the sum drifted past the window's top
and dropped the pass's last point (10x pass 2, 40x's final pass, 60x pass
1, 100x passes 2-3). The final pass sweeps three fine steps either side of
the earlier passes' best, the seven points its fit needs, on every
objective. The time estimator counts the frames from the same passes.

Driven on the simulated scope through ``AutofocusRunner.run()``, one sweep
per distinct set of the catalogue's three autofocus numbers, plus an
objective whose coarse and fine steps are equal, a case the estimator's
own copy of the pass loop once counted as one pass where the runner makes
two. The frames are counted where they are scored, which is what a pass
sweeps whatever moves it.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil

import pytest

from _lumascope.modules import protocol_time_estimator
from _lumascope.modules.protocol_time_estimator import ProtocolTimeEstimator
from tests.af_drives import af_runner_on, park_z, sweep
from tests.ast_seams import REPO_ROOT
from tests.scope_fakes import bind_settings_like_a_session, build_scope

pytestmark = pytest.mark.slow

# A test objective whose AF_max equals its AF_min, added to a copy of
# the shipped catalogue: the runner sweeps the range, then a final pass.
EQUAL_STEPS_ID = 'Equal steps 20x'
EQUAL_STEPS = {'AF_range': 10.0, 'AF_max': 2.0, 'AF_min': 2.0}

# Points per pass, by (AF_range, AF_max, AF_min). Every pass after the
# first sweeps its window whole; the final one is 7 points.
EXPECTED_POINTS_PER_PASS = {
    (3000, 1000, 300): (7, 7, 7),
    (1000, 350, 50): (6, 7, 7),
    (150, 72, 20): (5, 7, 7),
    (75, 36, 8): (5, 7, 7),
    (40, 18, 2): (5, 7, 7),
    (20, 9, 1): (5, 7, 7),
    (15, 6, 0.5): (6, 7, 7, 7),
    (10, 4, 0.2): (6, 7, 7, 7),
    (10, 2, 2): (11, 7),
}

START_Z_UM = 7000.0


def _numbers(objective) -> tuple[float, float, float]:
    return (
        objective.autofocus_half_range_um,
        objective.autofocus_step_max_um,
        objective.autofocus_step_min_um,
    )


def _passes(scored_z: list[float]) -> list[list[float]]:
    """The scored Zs split into passes: each pass sweeps upward from its start."""
    passes = [[scored_z[0]]]
    for previous, z in itertools.pairwise(scored_z):
        if z > previous:
            passes[-1].append(z)
        else:
            passes.append([z])
    return passes


@pytest.fixture(scope='module')
def swept(tmp_path_factory):
    """(the scope's catalogue, {numbers: the Zs scored in each pass}), one sweep per set of numbers.

    The scope reads a copy of the shipped data with the equal-steps
    objective added, as bring-up reads an installation's. The score is a
    stand-in by design: a Gaussian in Z centred just off the start, wide
    enough that every pass's peak sits inside its window, so no pass
    extends and the points counted are the targets the pass asked for.
    """
    root = tmp_path_factory.mktemp('installation')
    shutil.copytree(REPO_ROOT / 'data', root / 'data')
    catalogue_file = root / 'data' / 'objectives.json'
    catalogue = json.loads(catalogue_file.read_text())
    catalogue[EQUAL_STEPS_ID] = {**catalogue['20x Oly'], **EQUAL_STEPS}
    catalogue_file.write_text(json.dumps(catalogue))

    scope = build_scope(simulate=True, sim_model='LS850', source_path=root)
    try:
        bind_settings_like_a_session(scope)
        for driver in (scope._led_driver, scope._motion_driver, scope._camera_driver):
            driver.set_timing_mode('fast')
        scope.imaging.enable_streaming()
        runner = af_runner_on(scope, pattern='white')
        objectives = scope.objective_helper.objectives
        by_numbers = {}
        for objective in objectives.values():
            by_numbers.setdefault(_numbers(objective), objective)
        scored = {}
        with pytest.MonkeyPatch.context() as mp:
            for numbers, objective in by_numbers.items():
                half_range, _step_max, step_min = numbers
                centre = START_Z_UM + 0.37 * step_min
                width = half_range / 4
                zs = []

                def score(image, centre=centre, width=width, zs=zs):
                    z = scope.motion.get_current_position('Z')
                    zs.append(z)
                    return 1e7 * math.exp(-(((z - centre) / width) ** 2) / 2)

                mp.setattr('_lumascope.modules.autofocus_functions.autofocus_score', score)
                park_z(scope, START_Z_UM)
                sweep(runner, scope, objective_id=objective.id)
                scored[numbers] = _passes(zs)
        yield objectives, scored
    finally:
        scope.imaging.disable_streaming()
        scope.disconnect()


def test_every_pass_sweeps_every_target_and_the_final_pass_seven(swept):
    _objectives, scored = swept
    offenders = {
        numbers: (tuple(len(p) for p in passes), EXPECTED_POINTS_PER_PASS.get(numbers))
        for numbers, passes in scored.items()
        if tuple(len(p) for p in passes) != EXPECTED_POINTS_PER_PASS.get(numbers)
    }
    assert not offenders, f'(AF_range, AF_max, AF_min): (swept, expected) {offenders}'


def test_the_final_pass_steps_at_the_fine_step(swept):
    _objectives, scored = swept
    offenders = {}
    for numbers, passes in scored.items():
        step_min = numbers[2]
        final = passes[-1]
        gaps = [b - a for a, b in itertools.pairwise(final)]
        if any(abs(gap - step_min) > 0.05 * step_min for gap in gaps):
            offenders[numbers] = gaps
    assert not offenders, offenders


def test_the_time_estimator_counts_the_frames_a_sweep_takes(swept):
    objectives, scored = swept
    estimator = ProtocolTimeEstimator(objectives_loader=None)
    exposure_s = 0.05
    offenders = {}
    for numbers, passes in scored.items():
        objective = next(o for o in objectives.values() if _numbers(o) == numbers)
        half_range, step_max, _ = numbers
        per_frame = (
            step_max / protocol_time_estimator.Z_SPEED_UM_PER_S
            + max(exposure_s, 0.075)
            + protocol_time_estimator.AF_STEP_OVERHEAD_S
        )
        moves = 2 * half_range / protocol_time_estimator.Z_SPEED_UM_PER_S
        counted = (estimator._estimate_autofocus(objective, exposure_s) - moves) / per_frame
        frames = sum(len(p) for p in passes)
        if round(counted, 6) != frames:
            offenders[numbers] = (counted, frames)
    assert not offenders, f'(AF_range, AF_max, AF_min): (estimated, swept) {offenders}'


def test_a_first_pass_that_is_no_whole_number_of_steps_stays_in_its_window(sim_scope, monkeypatch):
    # 2x's first window, +-1000 at 350, is 5.7 steps: its 6 targets end at
    # 750 above the start's window bottom. Started 1000 below the top of
    # travel, a 7th target would be asked past it.
    runner = af_runner_on(sim_scope, pattern='white')
    top = sim_scope.motion.get_axis_limits('Z').max_um
    start = top - 1000.0
    scored = []

    def score(image):
        z = sim_scope.motion.get_current_position('Z')
        scored.append(z)
        return 1e7 * math.exp(-(((z - start) / 250.0) ** 2) / 2)

    monkeypatch.setattr('_lumascope.modules.autofocus_functions.autofocus_score', score)
    park_z(sim_scope, start)

    focus = sweep(runner, sim_scope, objective_id='2x Oly')

    first = _passes(scored)[0]
    assert len(first) == 6, first
    assert max(first) <= top, first
    assert focus == pytest.approx(start, abs=1.0)
