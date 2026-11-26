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

from _lumascope.modules.coord_transformations import PlateLocation, StageOffset
from _lumascope.modules.exceptions import ProtocolRunRefusedError, RefusalCause
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.objectives_loader import ObjectiveLoader
from tests.ast_seams import find_def
from tests.protocol_fixtures import protocol_from_file
from tests.test_adding_a_step_is_an_api_capability import _only_acquiring
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture
from tests.test_protocol_overwrite_guard import _build_tsv, _step_row
from tests.test_step_label_ssot import _build_protocol, _labeled_step
from tests.test_zstack_group_identity import _WIDE_XY


def _tile(proto, capabilities, tiling, *, frame_dimensions=None):
    return proto.apply_tiling(
        tiling=tiling,
        frame_dimensions=frame_dimensions or {'width': 1900, 'height': 1900},
        binning_size=1,
        axis_limits=_WIDE_XY,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        stage_offset=StageOffset(x_um=0.0, y_um=0.0),
        overlap_percent=0.0,
        capabilities=capabilities,
        objective_helper=ObjectiveLoader(),
    )


def _refused(proto, capabilities, tiling):
    before = proto.steps().copy()
    with (
        patch('_lumascope.modules.protocol.notifications.report_outcome') as report,
        pytest.raises(ProtocolRunRefusedError) as refusal,
    ):
        _tile(proto, capabilities, tiling)
    pd.testing.assert_frame_equal(proto.steps(), before)
    # Raised, not reported: the boundary the caller sits behind reports it.
    report.assert_not_called()
    return refusal.value


@pytest.mark.parametrize('second', ['2x2', '3x3', '1x1'])
def test_a_protocol_whose_every_step_is_tiled_refuses_another_grid(scale_capabilities, second):
    proto = _build_protocol([_labeled_step()])
    _tile(proto, scale_capabilities, '2x2')
    assert proto.num_steps == 4

    refusal = _refused(proto, scale_capabilities, second)
    assert refusal.reason == 'already_tiled'
    assert '2x2' in str(refusal)


def test_a_grid_the_installation_does_not_offer_is_refused(scale_capabilities):
    proto = _build_protocol([_labeled_step()])
    refusal = _refused(proto, scale_capabilities, '13x13')
    assert refusal.reason == 'tiling_unknown'


@pytest.mark.parametrize('tiling', ['13x13', ''])
def test_a_new_protocol_in_a_grid_the_installation_does_not_offer_is_refused(session, tiling):
    # The build from a config refused nothing: it raised KeyError from the
    # grid lookup, which reached the GUI's New as "Operation failed" when
    # the spinner was blank, and a script or REST caller as a bare KeyError.
    session.set_channel_settings('BF', acquire='image')
    with (
        patch('_lumascope.modules.protocol.notifications.report_outcome') as report,
        pytest.raises(ProtocolRunRefusedError) as refusal,
    ):
        session.new_protocol(tiling=tiling)
    assert refusal.value.reason == 'tiling_unknown'
    report.assert_not_called()


def test_a_step_with_an_unknown_objective_is_refused_not_tiled_at_nan(scale_capabilities):
    proto = _build_protocol([_labeled_step(), _labeled_step(label='B')])
    steps = proto.steps()
    steps.loc[1, 'Objective'] = 'no_such_objective'
    proto._set_steps(steps)

    refusal = _refused(proto, scale_capabilities, '2x2')
    assert refusal.reason == 'objective_not_in_catalog'
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
    assert proto.tiling == '1x1'
    _tile(proto, scale_capabilities, '2x2')
    assert proto.tiling == '2x2'
    # Two steps tiled by one Apply are two tile groups in one grid.
    two_steps = _build_protocol([_labeled_step(), _labeled_step(label='B')])
    _tile(two_steps, scale_capabilities, '2x2')
    assert two_steps.tiling == '2x2'
    # A hand-edited file's tiles carry tile group -1 on both steps; they are
    # still tiles, in a layout no grid offers.
    assert _tiled_in_no_offered_grid().tiling is None


def _partly_tiled(session):
    """The reporter's protocol: a step tiled 2x2, then a step added untiled."""
    _only_acquiring(session, 'BF')
    proto = session.scope.protocols.create_protocol(
        empty_config=session.get_sequenced_capture_config()
    )
    session.add_step(proto, at=PlateLocation(x_mm=20.0, y_mm=15.0, z_um=5000.0))
    session.apply_tiling(proto, '2x2')
    session.add_step(proto, at=PlateLocation(x_mm=40.0, y_mm=30.0, z_um=5000.0))
    return proto


def _tiles(steps):
    return {(row['Tile'], row['X'], row['Y'], row['Tile Group ID']) for _, row in steps.iterrows()}


class TestApplyTilesTheStepsNotYetTiled:
    """A step added to a tiled protocol takes the grid Apply is given; the
    steps already tiled keep their tiles (#805's report: 2x2, Add, 3x3)."""

    def test_an_added_step_takes_its_own_grid(self, session):
        proto = _partly_tiled(session)
        before = proto.steps()
        tiled_before = _tiles(before[before['Tile'] != ''])

        session.apply_tiling(proto, '3x3')

        steps = proto.steps()
        assert proto.num_steps == 4 + 9
        assert tiled_before <= _tiles(steps)
        new = steps[~steps['Tile Group ID'].isin(before['Tile Group ID'])]
        assert sorted(new['Tile']) == ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
        assert new['Tile Group ID'].nunique() == 1
        assert (steps['Tile'] != '').all()

    def test_a_protocol_tiled_in_two_grids_refuses_another_and_names_it_mixed(self, session):
        proto = _partly_tiled(session)
        session.apply_tiling(proto, '3x3')
        before = proto.steps().copy()
        with (
            patch('_lumascope.modules.protocol.notifications.report_outcome') as report,
            pytest.raises(ProtocolRunRefusedError) as refusal,
        ):
            session.apply_tiling(proto, '2x2')
        assert refusal.value.reason == 'already_tiled'
        assert 'Mixed' in str(refusal.value)
        pd.testing.assert_frame_equal(proto.steps(), before)
        report.assert_not_called()

    def test_one_tile_over_a_partly_tiled_protocol_changes_nothing(self, session):
        proto = _partly_tiled(session)
        before = proto.steps().copy()
        session.apply_tiling(proto, '1x1')
        pd.testing.assert_frame_equal(proto.steps(), before)


class TestAProtocolOfMoreThanOneGridSaysMixed:
    """``Protocol.tiling`` names one grid only when every step carries it
    (Eric's ruling on #805, 2026-09-03: a partly tiled protocol is not
    reported as one grid)."""

    def test_a_grid_beside_an_untiled_step(self, session):
        assert _partly_tiled(session).tiling == 'Mixed'

    def test_two_grids(self, session):
        proto = _partly_tiled(session)
        session.apply_tiling(proto, '3x3')
        assert proto.tiling == 'Mixed'

    def test_a_grid_clipped_at_the_edge_of_the_travel(self, tmp_path):
        # Until 4.0.0-beta31 a tile outside the travel was skipped, so a
        # protocol made in one grid can hold an edge well's group with a
        # column missing; it is not that grid (Eric 2026-11-26).
        full = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
        clipped = [t for t in full if not t.endswith('3')]
        rows = ''.join(
            _step_row(f'{well}_BF_T{tile}', well, tile, -1, group, x0 + i * 0.5, 30.0, 5000.0)
            for well, tiles, group, x0 in (('A1', full, 0, 10.0), ('A12', clipped, 1, 100.0))
            for i, tile in enumerate(tiles)
        )
        path = tmp_path / 'clipped.tsv'
        path.write_text(_build_tsv(rows))
        assert protocol_from_file(file_path=path).tiling == 'Mixed'


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
