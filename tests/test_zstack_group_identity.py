# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack group identifies the slices at ONE XY position.

Bug history
-----------
Applying tiling to an already-z-stacked protocol copied the parent step's
``Z-Stack Group ID`` onto every tile, so four distinct XY positions
claimed to be one z-stack group. The LED-hold decision reads exactly that
column -- ``same_zstack_group`` holds illumination across a step boundary
unconditionally, because within a real stack the stage barely moves -- so
the forged id held the LED lit while the stage travelled from tile to
tile, exposing sample regions that were not being imaged. No user setting
avoided it: the Advanced Settings opt-in is the OTHER arm of that ``or``.

The two construction paths disagreed about what the column meant.
``Protocol.from_config`` allocates a fresh id inside its tile loop, so a
New-Protocol build was already correct; only the Apply-Z-Stacking then
Apply-Tiling button order produced the forged id.

These tests pin the INVARIANT -- one group, one XY -- rather than either
button order.
"""

from __future__ import annotations

from modules.labware_loader import WellPlateLoader
from tests.test_step_label_ssot import (
    _ZSTACK,
    _WIDE_Z,
    _build_protocol,
    _labeled_step,
)


_WIDE_XY = {
    'X': {'limits': {'min': -1_000_000.0, 'max': 1_000_000.0}},
    'Y': {'limits': {'min': -1_000_000.0, 'max': 1_000_000.0}},
}


def _tile_2x2(proto, capabilities):
    return proto.apply_tiling(
        tiling='2x2',
        frame_dimensions={'width': 1900, 'height': 1900},
        binning_size=1,
        curr_step_idx=0,
        axes_config=_WIDE_XY,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        stage_offset={'x': 0, 'y': 0},
        overlap_percent=0.0,
        capabilities=capabilities,
    )


def _groups_to_xy(steps) -> dict[int, set[tuple[float, float]]]:
    groups: dict[int, set[tuple[float, float]]] = {}
    for _, step in steps.iterrows():
        group = int(step['Z-Stack Group ID'])
        if group == -1:
            continue
        groups.setdefault(group, set()).add((step['X'], step['Y']))
    return groups


def test_tiling_a_zstack_gives_each_tile_its_own_group(scale_capabilities):
    """One group must never span more than one XY position."""
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0, z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=_WIDE_Z)
    assert len(proto.steps()) == 6

    status = _tile_2x2(proto, scale_capabilities)
    assert status['tiles_skipped'] == 0

    steps = proto.steps()
    assert len(steps) == 24, '6 slices x 4 tiles'

    groups = _groups_to_xy(steps)
    assert len(groups) == 4, f'one group per tile, got {sorted(groups)}'
    for group, positions in groups.items():
        assert len(positions) == 1, (
            f'z-stack group {group} spans {len(positions)} XY positions: '
            f'{sorted(positions)} -- the LED-hold decision would hold '
            f'illumination across the move between them'
        )


def test_tiling_a_zstack_keeps_every_slice_of_a_tile_together(scale_capabilities):
    """The grouping must still bind the slices AT one position."""
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0, z=5000.0)])
    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=_WIDE_Z)
    _tile_2x2(proto, scale_capabilities)

    steps = proto.steps()
    counts = steps.groupby('Z-Stack Group ID').size()
    assert set(counts) == {6}, f'each tile keeps its 6 slices, got {dict(counts)}'


def test_tiling_an_ungrouped_step_allocates_no_group(scale_capabilities):
    """A step that is not part of a stack stays ungrouped through tiling."""
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0)])
    _tile_2x2(proto, scale_capabilities)

    steps = proto.steps()
    assert len(steps) == 4
    assert set(steps['Z-Stack Group ID']) == {-1}
