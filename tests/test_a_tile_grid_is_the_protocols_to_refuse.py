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
import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from modules.exceptions import ProtocolRunRefusedError, RefusalCause
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader
from modules.protocol import Protocol
from tests.ast_seams import find_def
from tests.scope_fakes import build_scope
from tests.test_a_zstack_with_no_range_is_refused import _standalone_config
from tests.test_step_label_ssot import _build_protocol, _labeled_step
from tests.test_zstack_group_identity import _WIDE_XY


_REPO_ROOT = pathlib.Path(__file__).parent.parent


def _tile(proto, capabilities, tiling, *, frame_dimensions=None):
    return proto.apply_tiling(
        tiling=tiling,
        frame_dimensions=frame_dimensions or {'width': 1900, 'height': 1900},
        binning_size=1,
        axis_limits=_WIDE_XY,
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


@pytest.mark.parametrize('tiling', ['13x13', ''])
def test_a_new_protocol_in_a_grid_the_installation_does_not_offer_is_refused(tiling):
    # The build from a config refused nothing: it raised KeyError from the
    # grid lookup, which reached the GUI's New as "Operation failed" when
    # the spinner was blank, and a script or REST caller as a bare KeyError.
    config = _standalone_config({'range': 20.0, 'step_size': 5.0}, use_zstacking=False)
    config['tiling'] = tiling
    scope = build_scope(simulate=True, source_path=_REPO_ROOT)
    try:
        with (
            patch('modules.protocol.notifications.report_outcome') as report,
            pytest.raises(ProtocolRunRefusedError) as refusal,
        ):
            Protocol.from_config(
                input_config=config,
                tiling_configs_file_loc=_REPO_ROOT / 'data' / 'tiling.json',
                capabilities=scope.capabilities,
                objective_helper=scope.objective_helper,
                wellplate_loader=scope.wellplate_loader,
            )
    finally:
        scope.disconnect()
    assert refusal.value.reason == 'tiling_unknown'
    report.assert_called_once()


def test_a_step_with_an_unknown_objective_is_refused_not_tiled_at_nan(scale_capabilities):
    proto = _build_protocol([_labeled_step(), _labeled_step(label='B')])
    steps = proto.steps()
    steps.loc[1, 'Objective'] = 'no_such_objective'
    proto._set_steps(steps)

    refusal = _refused(proto, scale_capabilities, '2x2')
    assert refusal.reason == 'objective_not_in_catalogue'
    assert refusal.cause == RefusalCause.REQUEST
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


def _tiled_in_no_offered_grid():
    # One row of two tiles: a hand-edited file's shape, since every grid the
    # installation builds is square.
    proto = _build_protocol([_labeled_step(), _labeled_step(label='B')])
    steps = proto.steps()
    steps['Tile'] = ['A1', 'A2']
    proto._set_steps(steps)
    return proto


def test_a_protocol_tiled_in_no_offered_grid_refuses_another(scale_capabilities):
    # Before, only a tiling that named an offered grid was refused, so a grid
    # over these tiles changed nothing and returned as if it had built.
    refusal = _refused(_tiled_in_no_offered_grid(), scale_capabilities, '2x2')
    assert refusal.reason == 'already_tiled'


def test_the_protocol_names_the_grid_its_steps_carry(scale_capabilities):
    proto = _build_protocol([_labeled_step()])
    assert proto.tiling() == '1x1'
    _tile(proto, scale_capabilities, '2x2')
    assert proto.tiling() == '2x2'
    assert _tiled_in_no_offered_grid().tiling() is None


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
