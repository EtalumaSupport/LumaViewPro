# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus whose final pass has no peak it trusts fails; it does not report a made-up focus.

``_find_best`` judged the curve and threw the judgment away: a fit it did
not trust fell through to the raw maximum, and the sweep reported that as
its focus, written to the step and the settings. On an LS720 with no
sample under the objective (2026-10-14), every pass's scores sat within
1.5% of each other, the fit shifted past twice the step spacing, and the
sweep reported ``best=117.76um``. The final pass now raises
``AutofocusFailedError('flat_focus_curve')`` on an untrusted pick, and the
failure path takes the stage back to where the sweep started.
"""

import pandas as pd
import pytest

from _lumascope.modules.autofocus_runner import AutofocusRunner
from _lumascope.modules.exceptions import AutofocusFailedError
from tests.af_drives import PARKED_Z_UM, af_runner_on, park_z, settled_z, sweep, z_asked

# The LS720's sweep with no sample, 2026-10-14 (20x with collar, range 40,
# step 18): every point the three passes measured, as autofocus.log has them.
_EMPTY_FIELD = [
    (81.74, 17330224.0),
    (99.75, 17481731.0),
    (117.76, 17595051.0),
    (135.76, 17537229.0),
    (153.77, 17575754.0),
    (117.10, 17444492.0),
    (123.08, 17526081.0),
    (129.07, 17541552.0),
    (135.05, 17429117.0),
    (141.03, 17460092.0),
    (147.02, 17527097.0),
    (153.00, 17510521.0),
    (111.77, 17505951.0),
    (113.75, 17522587.0),
    (115.73, 17467029.0),
    (117.70, 17470139.0),
    (119.68, 17433423.0),
    (121.66, 17516330.0),
    (123.63, 17430442.0),
]


def test_the_empty_field_sweep_is_no_focus():
    pick = AutofocusRunner._find_best(pd.DataFrame(_EMPTY_FIELD, columns=['position', 'score']))

    assert pick == (117.76, 'every score within 5% of the peak')


def _an_empty_field(scope, monkeypatch):
    """Score every frame as the LS720's empty field scored: within about 1%
    of each other, in no order the fit can trust, whatever Z it was made at."""
    monkeypatch.setattr(
        '_lumascope.modules.autofocus_functions.autofocus_score',
        lambda image: 1.75e7 + 2e5 * ((scope.motion.get_current_position('Z') * 7.31) % 1.0),
    )


def test_a_final_pass_with_no_trusted_peak_fails_and_moves_to_no_focus(sim_scope, monkeypatch):
    # Which arm made it untrusted is _find_best's (tests/test_find_best.py).
    runner = af_runner_on(sim_scope)
    _an_empty_field(sim_scope, monkeypatch)
    asked = z_asked(sim_scope, monkeypatch)

    with pytest.raises(AutofocusFailedError) as failed:
        sweep(runner, sim_scope)

    assert failed.value.reason == 'flat_focus_curve'
    assert asked[-1] == PARKED_Z_UM, f'the stage went somewhere after the failure: {asked}'
    assert settled_z(sim_scope) == pytest.approx(PARKED_Z_UM, abs=0.01)


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


def test_a_sweep_whose_frames_show_nothing_fails_and_moves_to_no_focus(ronchi_scope, monkeypatch):
    """The Ronchi under the 40x at brightfield's shipped light (5 mA, 2 ms,
    1 dB) is near black, as on the bench: frames below 1 DN that round to a
    fixed pattern, and to all black where the blur is heaviest. A centre of
    nothing but black measured nothing; scored 0 and counted as a point, it
    read as a peak's tail and the sweep reported a focus about 20 um low.
    With no point there, the fixed pattern's curve has no peak."""
    runner = AutofocusRunner(scope=ronchi_scope)
    start_z = settled_z(ronchi_scope)
    asked = z_asked(ronchi_scope, monkeypatch)

    with pytest.raises(AutofocusFailedError) as failed:
        sweep(
            runner,
            ronchi_scope,
            objective_id=_RONCHI_OBJECTIVE,
            led_illumination=5.0,
            camera_exposure=2.0,
            camera_gain=1.0,
        )

    assert failed.value.reason == 'flat_focus_curve'
    assert asked[-1] == start_z, f'the stage went somewhere after the failure: {asked}'
    assert settled_z(ronchi_scope) == pytest.approx(start_z, abs=0.01)


def test_the_same_ruling_lit_focuses_on_it(ronchi_scope):
    """Lit at the 40x Ronchi's start band (25 mA, 25 ms, 0 dB), the same sweep
    finds the ruling's focus: a frame with something in it keeps its point."""
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
