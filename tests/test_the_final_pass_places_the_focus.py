# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where an autofocus puts the stage is the final pass's own fit.

A Gaussian on a floor fitted to the final pass's points alone
(``focus_of_the_final_pass``): points at one Z in different passes
disagree by a few percent of the peak, and the curve's tails sit on a
floor, so a model over every pass wanders between runs on one field where
this one holds. Whether the curve has a peak at all is still every point's
question (``tests/test_autofocus_no_peak_is_a_failure.py``).

The recorded passes are bench sweeps' final passes, (Z um, score) as the
autofocus log wrote them.
"""

from __future__ import annotations

import logging
import math

import pytest

from _lumascope.modules.autofocus_runner import focus_of_the_final_pass
from tests.af_drives import af_runner_on, park_z, sweep

# 2026-10-22, an LS850T at 40x on one field, swept three times; step 1 um.
FIELD_40X = [
    [
        (7106.99, 9618720.0),
        (7108.0, 10804193.0),
        (7109.0, 10868471.0),
        (7110.01, 10127890.0),
        (7111.01, 9439438.0),
        (7112.02, 8573256.0),
    ],
    [
        (7106.99, 9572829.0),
        (7108.0, 10805495.0),
        (7109.0, 10859147.0),
        (7110.01, 10130607.0),
        (7111.01, 9412972.0),
        (7112.02, 8619834.0),
    ],
    [
        (7106.15, 8935812.0),
        (7107.16, 9945062.0),
        (7108.16, 10994517.0),
        (7109.17, 10664375.0),
        (7110.17, 9909426.0),
        (7111.17, 9232955.0),
    ],
]

# 2026-09-16, an LS850T at 20x on one field, swept twelve times, its peak
# score falling from run to run; step 2 um. The fit over every pass spread
# its foci over 3.5 um.
FIELD_20X = [
    [(3988.06, 2316631.0), (3990.05, 2501801.0), (3992.05, 2623932.0), (3994.05, 2691735.0), (3996.05, 2687145.0), (3998.05, 2545237.0), (4000.04, 2384417.0)],
    [(3985.83, 2112458.0), (3987.82, 2272546.0), (3989.82, 2436350.0), (3991.81, 2635267.0), (3993.82, 2702525.0), (3995.8, 2678240.0), (3997.81, 2571561.0)],
    [(3986.04, 2114344.0), (3988.04, 2323301.0), (3990.02, 2473319.0), (3992.02, 2610478.0), (3994.04, 2696530.0), (3996.03, 2660375.0), (3998.03, 2536216.0)],
    [(3986.12, 2130180.0), (3988.09, 2311153.0), (3990.12, 2481636.0), (3992.1, 2644234.0), (3994.08, 2673410.0), (3996.09, 2665027.0), (3998.09, 2556413.0)],
    [(3985.81, 2105743.0), (3987.81, 2267805.0), (3989.8, 2448670.0), (3991.8, 2607859.0), (3993.79, 2674888.0), (3995.79, 2642457.0), (3997.8, 2561329.0)],
    [(3985.78, 2075230.0), (3987.77, 2275600.0), (3989.77, 2466692.0), (3991.76, 2587897.0), (3993.77, 2701320.0), (3995.76, 2658540.0), (3997.76, 2572202.0)],
    [(3985.87, 180749.0), (3987.87, 196962.0), (3989.86, 214603.0), (3991.86, 227099.0), (3993.84, 233482.0), (3995.86, 230652.0), (3997.86, 220854.0)],
    [(3986.1, 184062.0), (3988.08, 199113.0), (3990.07, 214777.0), (3992.07, 227018.0), (3994.08, 234149.0), (3996.06, 229101.0), (3998.07, 219133.0)],
    [(3985.76, 105587.0), (3987.72, 113996.0), (3989.75, 123534.0), (3991.74, 130527.0), (3993.73, 134643.0), (3995.73, 135935.0), (3997.73, 131343.0), (3999.72, 119833.0)],
    [(3985.77, 106288.0), (3987.77, 116050.0), (3989.77, 123995.0), (3991.77, 131535.0), (3993.74, 135998.0), (3995.76, 136296.0), (3997.75, 130691.0), (3999.75, 120232.0)],
    [(3986.37, 27277.0), (3988.35, 29330.0), (3990.35, 32181.0), (3992.35, 32891.0), (3994.35, 34348.0), (3996.35, 34398.0), (3998.34, 30957.0), (4000.35, 29086.0)],
    [(3985.66, 25895.0), (3987.64, 28413.0), (3989.63, 30192.0), (3991.64, 33351.0), (3993.64, 33668.0), (3995.62, 34032.0), (3997.63, 31872.0), (3999.63, 30873.0)],
]  # fmt: skip

# 2026-09-09, a bench sweep at 20x: the final pass falls from its first
# point, the summit below the window.
SUMMIT_BELOW = [
    (6296.9, 510844136.0),
    (6298.88, 506183634.0),
    (6300.88, 501797574.0),
    (6302.88, 497413977.0),
    (6304.88, 490234555.0),
    (6306.88, 482590545.0),
    (6308.88, 473390854.0),
]


def _focus(points, step_um):
    return focus_of_the_final_pass([z for z, _ in points], [s for _, s in points], step_um)


def test_one_field_swept_again_focuses_at_one_z():
    foci_40x = [_focus(p, 1.0) for p in FIELD_40X]
    foci_20x = [_focus(p, 2.0) for p in FIELD_20X]

    assert foci_40x == pytest.approx([7108.7] * 3, abs=0.2)
    assert max(foci_20x) - min(foci_20x) <= 0.3, foci_20x


def test_a_summit_at_the_pass_edge_is_said_in_the_log(caplog):
    with caplog.at_level(logging.INFO, logger='LVP.autofocus'):
        _focus(SUMMIT_BELOW, 2.0)
        edge = caplog.messages[-1]
        _focus(FIELD_20X[0], 2.0)
        interior = caplog.messages[-1]

    assert "the pass's first point" in edge, edge
    assert 'FINAL FIT' in interior and 'point' not in interior, interior


def test_a_curve_on_a_floor_focuses_at_its_summit(sim_scope, monkeypatch):
    # A peak on a bright floor (60% of the peak's height), off the step
    # grid: the fit over every pass landed 1.9 um below it at 4x.
    runner = af_runner_on(sim_scope, pattern='white')
    start = 7000.0
    summit = start + 0.37 * 20.0

    def score(image):
        z = sim_scope.motion.get_current_position('Z')
        return 1e7 * (0.6 + math.exp(-(((z - summit) / 50.0) ** 2) / 2))

    monkeypatch.setattr('_lumascope.modules.autofocus_functions.autofocus_score', score)
    park_z(sim_scope, start)

    focus = sweep(runner, sim_scope, objective_id='4x Oly')

    assert focus == pytest.approx(summit, abs=0.2)
