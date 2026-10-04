# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether a tile grid can be built is the protocol's answer, not the panel's.

Before, the Apply Tiling button refused an already-tiled protocol itself, so
a script or REST caller got no refusal: a second grid over a tiled protocol
silently changed nothing and reported "0 tiles skipped". The protocol also
answered success for builds it could not do -- a step whose objective is not
in the catalogue was tiled at NaN positions, and a step whose grid could not
be computed was dropped from the protocol. Each is now refused or raised
before any step changes, and a refusal is reported once.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pandas as pd
import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader
from tests.ast_seams import find_def
from tests.test_step_label_ssot import _build_protocol, _labeled_step
from tests.test_zstack_group_identity import _WIDE_XY


def _tile(proto, capabilities, tiling, *, frame_dimensions=None):
    return proto.apply_tiling(
        tiling=tiling,
        frame_dimensions=frame_dimensions or {'width': 1900, 'height': 1900},
        binning_size=1,
        axes_config=_WIDE_XY,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        stage_offset={'x': 0, 'y': 0},
        overlap_percent=0.0,
        capabilities=capabilities,
        objective_helper=ObjectiveLoader(),
    )


def _refused(proto, capabilities, tiling):
    before = proto.steps().copy()
    with (
        patch('modules.protocol.notifications.report_outcome') as report,
        pytest.raises(ProtocolRunRefusedError) as refusal,
    ):
        _tile(proto, capabilities, tiling)
    pd.testing.assert_frame_equal(proto.steps(), before)
    report.assert_called_once()
    assert report.call_args.args[0] is refusal.value
    return refusal.value


@pytest.mark.parametrize('second', ['2x2', '3x3', '1x1'])
def test_a_tiled_protocol_refuses_another_grid(scale_capabilities, second):
    proto = _build_protocol([_labeled_step()])
    _tile(proto, scale_capabilities, '2x2')
    assert proto.num_steps() == 4

    refusal = _refused(proto, scale_capabilities, second)
    assert refusal.reason == 'already_tiled'
    assert '2x2' in str(refusal)


def test_a_grid_the_installation_does_not_offer_is_refused(scale_capabilities):
    proto = _build_protocol([_labeled_step()])
    refusal = _refused(proto, scale_capabilities, '13x13')
    assert refusal.reason == 'tiling_unknown'


def test_a_step_with_an_unknown_objective_is_refused_not_tiled_at_nan(scale_capabilities):
    proto = _build_protocol([_labeled_step(), _labeled_step(label='B')])
    steps = proto.steps()
    steps.loc[1, 'Objective'] = 'no_such_objective'
    proto._set_steps(steps)

    refusal = _refused(proto, scale_capabilities, '2x2')
    assert refusal.reason == 'objective_unknown'
    assert 'no_such_objective' in str(refusal)


def test_a_step_whose_grid_cannot_be_computed_fails_the_build(scale_capabilities):
    proto = _build_protocol([_labeled_step(), _labeled_step(label='B')])
    before = proto.steps().copy()
    with pytest.raises(KeyError):
        _tile(proto, scale_capabilities, '2x2', frame_dimensions={'width': 1900})
    pd.testing.assert_frame_equal(proto.steps(), before)


def test_an_untiled_protocol_still_tiles(scale_capabilities):
    proto = _build_protocol([_labeled_step()])
    _tile(proto, scale_capabilities, '2x2')
    assert sorted(proto.steps()['Tile']) == ['A1', 'A2', 'B1', 'B2']


def test_the_panel_decides_nothing_about_a_grid():
    # The button hands the press to the one reporter; neither it nor the body
    # it runs works out the protocol's current tiling or writes its own popup
    # for a failure.
    for name in ('apply_tiling', '_apply_tiling'):
        node = find_def('ui/protocol_settings.py', name, class_name='ProtocolSettings')
        assert node is not None, name
        calls = {
            getattr(n.func, 'attr', None) or getattr(n.func, 'id', None)
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
        }
        assert 'determine_tiling_label_from_tiles' not in calls, name
        assert not any(isinstance(n, ast.Try) for n in ast.walk(node)), name
    button = find_def('ui/protocol_settings.py', 'apply_tiling', class_name='ProtocolSettings')
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'run_reported'
        for n in ast.walk(button)
    )
