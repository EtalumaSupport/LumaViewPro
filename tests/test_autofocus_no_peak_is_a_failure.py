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

from modules.autofocus_runner import AutofocusRunner, FocusPick
from modules.exceptions import AutofocusFailedError
from tests.af_drives import AF_CENTER_Z, af_runner_and_scope, drive_af

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


def test_the_empty_field_sweep_has_no_peak_the_fit_trusts():
    pick = AutofocusRunner._find_best(pd.DataFrame(_EMPTY_FIELD, columns=['position', 'score']))

    assert pick == (117.76, 'fit shifted past twice the step spacing')


def _z_moves(scope):
    return [c.args[1] for c in scope.motion.start_move_absolute.call_args_list if c.args[0] == 'Z']


def test_a_final_pass_with_no_trusted_peak_fails_and_moves_to_no_focus():
    # Which arm made it untrusted is _find_best's (tests/test_find_best.py).
    runner, scope = af_runner_and_scope()
    runner._find_best = lambda df: FocusPick(AF_CENTER_Z + 7.0, 'fit not concave')

    with pytest.raises(AutofocusFailedError) as failed:
        drive_af(runner)

    assert failed.value.reason == 'flat_focus_curve'
    assert AF_CENTER_Z + 7.0 not in _z_moves(scope)


def test_a_final_pass_whose_peak_is_trusted_is_the_focus():
    runner, scope = af_runner_and_scope()
    runner._find_best = lambda df: FocusPick(AF_CENTER_Z + 7.0, None)

    focus = drive_af(runner)

    assert focus == AF_CENTER_Z + 7.0
    assert AF_CENTER_Z + 7.0 in _z_moves(scope)
