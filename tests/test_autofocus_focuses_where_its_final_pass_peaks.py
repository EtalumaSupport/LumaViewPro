# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus focuses where its final pass's own points put it.

The contract (``AutofocusRunner.run``): the coarse passes only aim the next
pass, each centred on the highest usable score so far; the focus is where
the final pass's floor-plus-Gaussian fit puts it, no coarse point in it. A
sweep refuses only when it has no usable score (one finite and above zero)
-- none at a pass's end so far, or none in its final pass -- and never for
a curve's shape: a field with nothing to focus on focuses where its final
pass scored highest.

Bug history: the final pass asked whether the curve had a peak before it
asked where, and the whether came from a log-quadratic over every pass's
points. On a bench LS820 at 10x (2026-11-24) the coarse points bent that
fit 18 um from the final pass's peak, the shift check refused it, and an
autofocus that had found its focus failed "Focus curve is flat or invalid"
and put Z back where it started.
"""

import numpy as np
import pytest

from _lumascope.modules.autofocus_runner import AutofocusRunner
from _lumascope.modules.exceptions import AutofocusFailedError
from tests.af_drives import (
    FOCAL_Z_UM,
    PARKED_Z_UM,
    af_runner_on,
    park_z,
    settled_z,
    sweep,
    z_asked,
)

# The 10x objective's three passes from PARKED_Z_UM: 5 targets 36 um apart,
# 7 at 12 um, then the final pass's 7 at 8 um.
_COARSE_POINTS = 12
_FINAL_STEP_UM = 8.0


def _score_by_z(scope, monkeypatch, curve):
    """Score every frame by the Z it was made at, through ``curve(z)``."""
    monkeypatch.setattr(
        '_lumascope.modules.autofocus_functions.autofocus_score',
        lambda image: curve(scope.motion.get_current_position('Z')),
    )


def _score_by_frame(monkeypatch, scores):
    """Score the n-th frame of the sweep ``scores(n)``, n counted from 0."""
    taken = []

    def score(image):
        taken.append(None)
        return scores(len(taken) - 1)

    monkeypatch.setattr('_lumascope.modules.autofocus_functions.autofocus_score', score)


def _a_peak_at(z_um):
    """A curve on a floor with its summit at ``z_um``."""
    return lambda z: 1e5 + 1e6 * np.exp(-((z - z_um) ** 2) / (2 * 30.0**2))


# --- A focus the final pass found ----------------------------------------

# The bench LS820's first press, 2026-11-24 (4x lens, '10x Oly' set): every
# point its autofocus.log has, the passes' repeats at one Z averaged. Its
# final pass peaked at 3949 um; the next press with the right objective
# focused at 3949.46 um.
_LS820_CURVE = {
    3925.0: np.mean([14068299.0, 14071261.0, 14046849.0]),
    3933.0: 14243174.0,
    3937.0: 14270851.0,
    3941.0: 14325335.0,
    3949.0: np.mean([14309773.0, 14326699.0]),
    3957.0: 14303680.0,
    3961.0: np.mean([14243738.0, 14277983.0]),
    3965.0: 14167055.0,
    3973.0: np.mean([14014694.0, 14027824.0]),
    3985.0: 13769407.0,
    3997.0: np.mean([13436483.0, 13431846.0]),
    4033.0: 12484155.0,
    4069.0: 11917274.0,
}
_LS820_CENTRE_UM = 4000.0


def test_a_peak_on_a_high_floor_is_focused_where_the_final_pass_puts_it(sim_scope, monkeypatch):
    runner = af_runner_on(sim_scope)
    park_z(sim_scope, _LS820_CENTRE_UM)
    z, s = zip(*sorted(_LS820_CURVE.items()), strict=True)
    _score_by_z(sim_scope, monkeypatch, lambda at: float(np.interp(at, z, s)))

    focus = sweep(runner, sim_scope)

    assert focus == pytest.approx(3948.0, abs=2.0)
    assert settled_z(sim_scope) == pytest.approx(focus, abs=0.01)


def test_a_field_with_nothing_to_focus_on_focuses_and_succeeds(sim_scope, monkeypatch):
    """The stated cost: the LS720's sweep with no sample (2026-10-14), every
    score within about 1% of the others in no order, is focused where its
    final pass scored highest, inside the sweep's range."""
    runner = af_runner_on(sim_scope)
    _score_by_z(sim_scope, monkeypatch, lambda z: 1.75e7 + 2e5 * ((z * 7.31) % 1.0))

    focus = sweep(runner, sim_scope)

    assert PARKED_Z_UM - 75.0 <= focus <= PARKED_Z_UM + 75.0, focus
    assert settled_z(sim_scope) == pytest.approx(focus, abs=0.01)


# --- No usable score in the final pass -------------------------------------


@pytest.mark.parametrize(
    # parametrize-justified: a zero and a NaN leave the final pass by
    # different paths (a floor point the fit could take, a frame it drops).
    'final_score',
    [0.0, float('nan')],
)
def test_a_final_pass_with_no_usable_score_fails_whatever_the_coarse_passes_scored(
    sim_scope, monkeypatch, final_score
):
    runner = af_runner_on(sim_scope)
    peak = _a_peak_at(FOCAL_Z_UM)
    _score_by_frame(
        monkeypatch,
        lambda n: (
            peak(sim_scope.motion.get_current_position('Z')) if n < _COARSE_POINTS else final_score
        ),
    )
    asked = z_asked(sim_scope, monkeypatch)

    with pytest.raises(AutofocusFailedError) as failed:
        sweep(runner, sim_scope)

    assert failed.value.reason == 'no_focus_scores'
    assert asked[-1] == PARKED_Z_UM, f'the stage went somewhere after the failure: {asked}'


def test_nan_frames_around_a_mid_window_peak_neither_extend_the_pass_nor_move_the_focus(
    sim_scope, monkeypatch
):
    """Every other final-pass frame scores NaN, the first and last among
    them: three finite points are left, too few for the fit, and the
    highest of them is the focus (its peak in the middle of the window)."""
    runner = af_runner_on(sim_scope)
    peak = _a_peak_at(FOCAL_Z_UM)
    _score_by_frame(
        monkeypatch,
        lambda n: (
            float('nan')
            if n >= _COARSE_POINTS and (n - _COARSE_POINTS) % 2 == 0
            else peak(sim_scope.motion.get_current_position('Z'))
        ),
    )
    asked = z_asked(sim_scope, monkeypatch)

    focus = sweep(runner, sim_scope)

    final_window_top = max(asked[_COARSE_POINTS : _COARSE_POINTS + 7])
    assert max(asked[_COARSE_POINTS:]) <= final_window_top, (
        f'the final pass extended past its window: {asked[_COARSE_POINTS:]}'
    )
    assert focus == pytest.approx(FOCAL_Z_UM, abs=_FINAL_STEP_UM)


# --- What aims the next pass ------------------------------------------------


def test_the_next_window_centres_on_the_earliest_of_tied_highest_scores(sim_scope, monkeypatch):
    """The first pass's targets 2925, 2961, 2997, 3033, 3069 um: the second
    and fourth score the same, the highest, and the second pass centres on
    the earlier of them, its window 36 um either side."""
    runner = af_runner_on(sim_scope)
    first_pass = [1.0e5, 3.0e5, 2.0e5, 3.0e5, 1.0e5]
    peak = _a_peak_at(2961.0)
    _score_by_frame(
        monkeypatch,
        lambda n: first_pass[n] if n < 5 else peak(sim_scope.motion.get_current_position('Z')),
    )
    asked = z_asked(sim_scope, monkeypatch)

    sweep(runner, sim_scope)

    second_pass = asked[5:12]
    assert second_pass[0] == pytest.approx(2961.0 - 36.0, abs=0.01), second_pass
    assert second_pass[-1] == pytest.approx(2961.0 + 36.0, abs=0.01), second_pass


# --- A real sweep of a ruling in too little light ----------------------------

# The simulated Ronchi under the 40x, in focus where Z is parked: the
# window is centred on the focus, as V5's walk centred it (2026-11-11).
_RONCHI_OBJECTIVE = '40x w/collar'
_RONCHI_FOCAL_Z_UM = 2950.0


@pytest.fixture
def ronchi_scope():
    """A fast-tier simulated LS850 under the 40x with the bench's Ronchi ruling on its stage,
    homed, parked at its focus, streaming."""
    from _lumascope.drivers.simulated_specimen import Slide
    from tests.scope_fakes import bind_settings_like_a_session, build_scope, home_sim_scope

    scope = build_scope(simulate=True, sim_model='LS850', sim_slide=Slide.RONCHI)
    bind_settings_like_a_session(scope, objective_id=_RONCHI_OBJECTIVE)
    for driver in (scope._led_driver, scope._motion_driver, scope._camera_driver):
        driver.set_timing_mode('fast')
    scope.imaging.enable_streaming()
    home_sim_scope(scope)
    scope._camera_driver.set_focal_z(_RONCHI_FOCAL_Z_UM)
    park_z(scope, _RONCHI_FOCAL_Z_UM)
    yield scope
    scope.imaging.disable_streaming()
    scope.disconnect()


def test_a_sweep_with_dark_frames_and_a_lit_final_pass_focuses_on_that_pass(ronchi_scope):
    """The Ronchi under the 40x at brightfield's shipped light (5 mA, 2 ms,
    1 dB) is near black, as on the bench: frames below 1 DN that round to a
    fixed pattern, and to all black where the blur is heaviest, which score
    NaN. Its final pass has usable scores, so the sweep focuses on that
    pass's peak and the stage goes there."""
    runner = AutofocusRunner(scope=ronchi_scope)

    focus = sweep(
        runner,
        ronchi_scope,
        objective_id=_RONCHI_OBJECTIVE,
        led_illumination=5.0,
        camera_exposure=2.0,
        camera_gain=1.0,
    )

    assert focus == pytest.approx(_RONCHI_FOCAL_Z_UM, abs=25.0)
    assert settled_z(ronchi_scope) == pytest.approx(focus, abs=0.01)


def test_a_dark_frame_is_scored_as_it_is_with_one_capture_per_point(ronchi_scope, monkeypatch):
    """A capture waits out the light's change itself (``frame_validity``), so
    a dark frame is the sample's, not one taken before the light came up:
    the sweep scores it as it is and captures once per point, however dark
    every frame is."""
    captured = []
    scored = []
    capture_still = ronchi_scope.imaging.capture_still

    def counted_capture(**kwargs):
        captured.append(None)
        return capture_still(**kwargs)

    from _lumascope.modules import autofocus_functions

    real_score = autofocus_functions.autofocus_score

    def counted_score(image):
        scored.append(float(np.mean(image)))
        return real_score(image=image)

    monkeypatch.setattr(ronchi_scope.imaging, 'capture_still', counted_capture)
    monkeypatch.setattr(autofocus_functions, 'autofocus_score', counted_score)

    sweep(
        AutofocusRunner(scope=ronchi_scope),
        ronchi_scope,
        objective_id=_RONCHI_OBJECTIVE,
        led_illumination=5.0,
        camera_exposure=2.0,
        camera_gain=1.0,
    )

    assert any(mean < 1.0 for mean in scored)
    assert len(captured) == len(scored)


def test_the_same_ruling_lit_focuses_on_it(ronchi_scope):
    """Lit at the 40x Ronchi's start band (25 mA, 25 ms, 0 dB), the same sweep
    finds the ruling's focus."""
    runner = AutofocusRunner(scope=ronchi_scope)

    focus = sweep(
        runner,
        ronchi_scope,
        objective_id=_RONCHI_OBJECTIVE,
        led_illumination=25.0,
        camera_exposure=25.0,
        camera_gain=0.0,
    )

    assert focus == pytest.approx(_RONCHI_FOCAL_Z_UM, abs=1.0)
