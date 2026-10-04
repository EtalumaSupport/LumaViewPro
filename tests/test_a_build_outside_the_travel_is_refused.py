# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A tile grid or z-stack with a position outside the stage's travel is refused.

Before, Apply Tiling and Apply Z-Stacking dropped the tiles and slices that
fell outside the travel, built the rest, and returned a count that only the
GUI read, turning it into a popup of its own. A script or REST caller got a
grid with holes, or a stack missing its ends, that stitched and projected as
if it were whole. The protocol now refuses the build before any step changes,
naming how many positions are outside and which steps they belong to, and
the GUI decides nothing: it hands the press to the one reporter.

A stack whose range or step size is not greater than zero is refused the same
way and for the same reason as ``Protocol.from_config`` refuses it.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pandas as pd
import pytest

from modules.coord_transformations import CoordinateTransformer
from modules.exceptions import ProtocolRunRefusedError
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader
from tests.ast_seams import find_def
from tests.test_step_label_ssot import _build_protocol, _labeled_step
from tests.test_zstack_group_identity import _WIDE_XY

_PLATE = '6 well microplate'
_STACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}  # 4950..5050


def _tile(proto, capabilities, axis_limits):
    proto.apply_tiling(
        tiling='2x2',
        frame_dimensions={'width': 1900, 'height': 1900},
        binning_size=1,
        axis_limits=axis_limits,
        labware=WellPlateLoader().get_plate(_PLATE),
        stage_offset={'x': 0, 'y': 0},
        overlap_percent=0.0,
        capabilities=capabilities,
        objective_helper=ObjectiveLoader(),
    )


def _refused(proto, build):
    before = proto.steps().copy()
    with (
        patch('modules.protocol.notifications.report_outcome') as report,
        pytest.raises(ProtocolRunRefusedError) as refusal,
    ):
        build()
    pd.testing.assert_frame_equal(proto.steps(), before)
    report.assert_called_once()
    assert report.call_args.args[0] is refusal.value
    return refusal.value


def test_a_grid_with_a_tile_outside_the_travel_is_refused(scale_capabilities):
    # The stage X of each tile, from a build with travel to spare; the X
    # travel then ends between the two columns, so two of the four are out.
    probe = _build_protocol([_labeled_step(x=60.0, y=40.0)])
    _tile(probe, scale_capabilities, _WIDE_XY)
    plate = WellPlateLoader().get_plate(_PLATE)
    stage_x = sorted(
        {
            CoordinateTransformer().plate_to_stage(
                labware=plate, stage_offset={'x': 0, 'y': 0}, px=x, py=y
            )[0]
            for x, y in zip(probe.steps()['X'], probe.steps()['Y'], strict=True)
        }
    )
    assert len(stage_x) == 2
    narrow = {
        'X': {'min': -1_000_000.0, 'max': sum(stage_x) / 2},
        'Y': _WIDE_XY['Y'],
    }

    proto = _build_protocol([_labeled_step(x=60.0, y=40.0)])
    refusal = _refused(proto, lambda: _tile(proto, scale_capabilities, narrow))

    assert refusal.reason == 'tiles_outside_travel'
    assert '2 of the 4 tiles' in str(refusal)
    assert proto.steps()['Name'][0] in str(refusal)


def test_a_stack_with_a_slice_outside_the_travel_is_refused():
    proto = _build_protocol([_labeled_step(z=5000.0)])
    refusal = _refused(
        proto,
        lambda: proto.apply_zstacking(
            zstack_params=_STACK, axis_limits={'Z': {'min': 4960.0, 'max': 10_000.0}}
        ),
    )

    assert refusal.reason == 'zslices_outside_travel'
    assert '1 of the 6 z-slices' in str(refusal)
    assert proto.steps()['Name'][0] in str(refusal)


@pytest.mark.parametrize(
    ('range_um', 'step_um'), [(0.0, 20.0), (100.0, 0.0), (-100.0, 20.0), (100.0, -20.0)]
)
def test_a_stack_with_no_extent_is_refused(range_um, step_um):
    proto = _build_protocol([_labeled_step(z=5000.0)])
    params = {'range': range_um, 'step_size': step_um, 'z_reference': 'center'}
    refusal = _refused(
        proto,
        lambda: proto.apply_zstacking(
            zstack_params=params, axis_limits={'Z': {'min': 0.0, 'max': 10_000.0}}
        ),
    )

    assert refusal.reason == 'zstack_not_configured'


def test_the_panel_decides_nothing_about_a_stack():
    # The button hands the press to the one reporter; neither it nor the body
    # it runs checks the values, catches what the protocol raises, or writes
    # its own popup.
    for name in ('apply_zstacking', '_apply_zstacking'):
        node = find_def('ui/protocol_settings.py', name, class_name='ProtocolSettings')
        assert node is not None, name
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert 'show_notification_popup' not in names, name
        assert not any(isinstance(n, (ast.Try, ast.If, ast.Compare)) for n in ast.walk(node)), name
    button = find_def('ui/protocol_settings.py', 'apply_zstacking', class_name='ProtocolSettings')
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'run_reported'
        for n in ast.walk(button)
    )


def test_the_panel_shows_no_count_of_its_own_for_a_grid():
    node = find_def('ui/protocol_settings.py', '_apply_tiling', class_name='ProtocolSettings')
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert 'show_notification_popup' not in names
