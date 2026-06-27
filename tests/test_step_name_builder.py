# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the canonical step-name builder/parser (StepNameComponents).

Three guarantees:

1. CANONICAL OUTPUT -- each component record renders to an exact, fixed-order
   filename string (base, channel, tile, objective, turret, z, scan, post). The
   explicit (components -> name) table is the on-disk filename contract; a token
   reorder or format change breaks a case.

2. IDEMPOTENCE / replace-not-append -- rebuilding after changing one component
   yields exactly one token for that component. The previous append-if-absent
   builder left the stale token beside the new one when a built name was fed
   back as the seed; this is the structural fix for the channel-change and
   stitch-filename corruption.

3. ROUND-TRIP -- build_step_name(parse_step_name(s)) == s for any canonical s.
"""

import dataclasses

import pytest

from modules.common_utils import (
    StepNameComponents,
    build_step_name,
    parse_step_name,
)

_KNOWN_LAYERS = ['Blue', 'Green', 'Red', 'BF', 'PC', 'DF', 'Lumi']
_KNOWN_OBJECTIVES = ['10x', '4x', '20x']


# Canonical (components -> rendered name) cases spanning every dimension and the
# fixed token order: base, channel, tile (T<tile>), objective, turret (Turret<n>),
# z (Z<n>), scan (4+ digits), then the post-suffix chain. Each expected string is
# the contract a step-name filename must keep; a builder change that shifts a
# token order or format breaks one of these.
_CANONICAL_CASES = [
    (StepNameComponents(well='A2', channel='BF'), 'A2_BF'),
    (StepNameComponents(well='A1', channel='Green'), 'A1_Green'),
    (StepNameComponents(well='H12', channel='Red', tile='A1'), 'H12_Red_TA1'),
    (StepNameComponents(well='A1', channel='BF', objective='10x'), 'A1_BF_10x'),
    (StepNameComponents(well='A1', channel='BF', turret_position=2), 'A1_BF_Turret2'),
    (StepNameComponents(well='A1', channel='BF', z_index=5), 'A1_BF_Z5'),
    (StepNameComponents(well='A1', channel='BF', tile='B2', z_index=3), 'A1_BF_TB2_Z3'),
    (StepNameComponents(well='A1', channel='BF', scan_count=0), 'A1_BF_0000'),
    (StepNameComponents(well='A1', channel='BF', scan_count=12), 'A1_BF_0012'),
    (StepNameComponents(well='A1', channel='BF', scan_count=12345), 'A1_BF_12345'),
    (
        StepNameComponents(well='A1', channel='BF', tile='A1', z_index=2, scan_count=7),
        'A1_BF_TA1_Z2_0007',
    ),
    (StepNameComponents(custom_prefix='custom0001', channel='BF'), 'custom0001_BF'),
    (
        StepNameComponents(custom_prefix='custom0042', channel='Red', tile='C3'),
        'custom0042_Red_TC3',
    ),
    (StepNameComponents(well='A1', channel='BF', post=('stitched',)), 'A1_BF_stitched'),
    (
        StepNameComponents(well='A1', channel='Composite', post=('stitched',)),
        'A1_Composite_stitched',
    ),
    (StepNameComponents(well='A1', channel='BF', post=('video',)), 'A1_BF_video'),
    (StepNameComponents(well='A1', post=('hyperstack',)), 'A1_hyperstack'),
    (StepNameComponents(well='A1', channel='BF', post=('stack',)), 'A1_BF_stack'),
    (StepNameComponents(well='A1', channel='BF', post=('zproj_median',)), 'A1_BF_zproj_median'),
    # Chained post-outputs: a stitch then z-projected / video'd carries both
    # suffixes in order; the single-token post field could only hold one.
    (
        StepNameComponents(well='A1', channel='BF', post=('stitched', 'zproj_median')),
        'A1_BF_stitched_zproj_median',
    ),
    (
        StepNameComponents(well='A1', channel='BF', post=('stitched', 'video')),
        'A1_BF_stitched_video',
    ),
    (StepNameComponents(well='A1'), 'A1'),  # bare well, no channel yet
]

_FRESH_CASES = [c for c, _ in _CANONICAL_CASES]


class TestCanonicalOutput:
    @pytest.mark.parametrize(('c', 'expected'), _CANONICAL_CASES)
    def test_build_renders_canonical_name(self, c, expected):
        assert build_step_name(c) == expected, (
            f'builder rendered {build_step_name(c)!r}, expected {expected!r} for {c}'
        )


class TestRoundTrip:
    @pytest.mark.parametrize('c', _FRESH_CASES)
    def test_build_parse_build_is_identity(self, c):
        name = build_step_name(c)
        reparsed = parse_step_name(
            name, known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES
        )
        assert build_step_name(reparsed) == name, (
            f'round-trip changed {name!r} -> {build_step_name(reparsed)!r}'
        )

    def test_parse_classifies_a_full_name(self):
        c = parse_step_name(
            'A1_Green_TB2_10x_Z3_0005',
            known_layers=_KNOWN_LAYERS,
            known_objectives=_KNOWN_OBJECTIVES,
        )
        assert c.well == 'A1'
        assert c.channel == 'Green'
        assert c.tile == 'B2'
        assert c.objective == '10x'
        assert c.z_index == 3
        assert c.scan_count == 5

    def test_parse_recovers_chained_post_suffixes(self):
        # A z-projection of a stitched output carries both suffixes; parse must
        # recover the ordered tuple, not just the last token.
        c = parse_step_name('A1_BF_0003_stitched_zproj_median', known_layers=_KNOWN_LAYERS)
        assert c.post == ('stitched', 'zproj_median')
        assert c.scan_count == 3

    def test_parse_recovers_custom_prefix(self):
        c = parse_step_name('custom0001_BF', known_layers=_KNOWN_LAYERS)
        assert c.custom_prefix == 'custom0001'
        assert c.well == ''
        assert c.channel == 'BF'

    def test_parse_strips_parent_folder(self):
        c = parse_step_name('Green/A1_Green_TB2', known_layers=_KNOWN_LAYERS)
        assert c.well == 'A1'
        assert c.channel == 'Green'
        assert c.tile == 'B2'


class TestIdempotentChannelChange:
    """The channel-change corruption: feeding a built name back as the seed and
    appending the new channel left the old one in place (A2_BF -> A2_BF_Green).
    Rebuilding from components must yield exactly one channel token."""

    def test_well_step_channel_change(self):
        c = parse_step_name('A2_BF', known_layers=_KNOWN_LAYERS)
        c = dataclasses.replace(c, channel='Green')
        assert build_step_name(c) == 'A2_Green'

    def test_custom_step_channel_change(self):
        # The custom step took the same corruption via the Well != '' gate.
        c = parse_step_name('custom0001_BF', known_layers=_KNOWN_LAYERS)
        c = dataclasses.replace(c, channel='Red')
        assert build_step_name(c) == 'custom0001_Red'

    def test_channel_change_preserves_other_tokens(self):
        c = parse_step_name('A1_BF_TB2_Z3', known_layers=_KNOWN_LAYERS)
        c = dataclasses.replace(c, channel='Green')
        assert build_step_name(c) == 'A1_Green_TB2_Z3'


class TestTileOmittedNotStripped:
    """The stitch-filename corruption: the tile token was removed by a strip
    helper keyed on an external Tile column that was empty post-record, so the
    token survived. Building with tile=None omits the token by construction --
    no external column consulted."""

    def test_stitch_omits_tile_regardless_of_origin(self):
        c = parse_step_name('A1_BF_TA1', known_layers=_KNOWN_LAYERS)
        c = dataclasses.replace(c, tile=None, post=('stitched',))
        assert build_step_name(c) == 'A1_BF_stitched'

    def test_composite_collapses_channel_and_tile(self):
        c = parse_step_name('A1_Green_TA1', known_layers=_KNOWN_LAYERS)
        c = dataclasses.replace(c, channel='Composite', tile=None, post=('stitched',))
        assert build_step_name(c) == 'A1_Composite_stitched'


class TestTurretToken:
    """The turret token shares its leading 'T' with the tile token, so a loose
    tile match ('any segment starting with T') swallowed 'Turret<n>': the turret
    position was lost and surfaced as a bogus tile ('urret<n>'). Parsing must
    keep the two distinct in every combination, and the round-trip must hold."""

    def test_turret_only_round_trips(self):
        name = build_step_name(StepNameComponents(well='A1', channel='BF', turret_position=3))
        assert name == 'A1_BF_Turret3'
        c = parse_step_name(name, known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES)
        assert c.turret_position == 3
        assert c.tile is None
        assert build_step_name(c) == name

    def test_turret_with_objective_round_trips(self):
        # Objective and turret are distinct tokens; the loose tile match also
        # reordered them by stealing the turret segment first.
        name = build_step_name(
            StepNameComponents(
                well='A1', channel='BF', objective='10x', turret_position=3, z_index=0
            )
        )
        assert name == 'A1_BF_10x_Turret3_Z0'
        c = parse_step_name(name, known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES)
        assert c.objective == '10x'
        assert c.turret_position == 3
        assert c.tile is None
        assert build_step_name(c) == name

    def test_turret_with_tile_keeps_both(self):
        name = build_step_name(
            StepNameComponents(well='A1', channel='BF', tile='B2', turret_position=4, z_index=1)
        )
        assert name == 'A1_BF_TB2_Turret4_Z1'
        c = parse_step_name(name, known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES)
        assert c.tile == 'B2'
        assert c.turret_position == 4
        assert build_step_name(c) == name

    def test_full_name_with_all_tokens_round_trips(self):
        name = build_step_name(
            StepNameComponents(
                well='A1',
                channel='Green',
                tile='C3',
                objective='20x',
                turret_position=2,
                z_index=4,
                scan_count=7,
            )
        )
        c = parse_step_name(name, known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES)
        assert build_step_name(c) == name
        assert c.tile == 'C3'
        assert c.turret_position == 2
        assert c.objective == '20x'


class TestLargeMosaicTile:
    """A mosaic past 26 rows labels rows with one-OR-MORE letters ('AA', 'AB',
    ... bijective base-26), not chr() punctuation. The multi-letter tile token
    must round-trip through parse/build, and widening the row label must still
    leave 'Turret<n>' -- whose leading 'T' it shares -- classified as a turret."""

    @pytest.mark.parametrize('tile', ['AA1', 'AB3', 'BZ12', 'ZZ9'])
    def test_multiletter_tile_round_trips(self, tile):
        name = build_step_name(StepNameComponents(well='A1', channel='BF', tile=tile))
        assert name == f'A1_BF_T{tile}'
        c = parse_step_name(name, known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES)
        assert c.tile == tile
        assert build_step_name(c) == name

    def test_turret_not_swallowed_by_multiletter_tile(self):
        c = parse_step_name(
            'A1_BF_Turret7', known_layers=_KNOWN_LAYERS, known_objectives=_KNOWN_OBJECTIVES
        )
        assert c.turret_position == 7
        assert c.tile is None


class TestBuildEdgeCases:
    def test_empty_components_render_empty(self):
        assert build_step_name(StepNameComponents()) == ''

    def test_custom_prefix_wins_over_well(self):
        c = StepNameComponents(well='A1', custom_prefix='custom0001', channel='BF')
        assert build_step_name(c) == 'custom0001_BF'

    def test_scan_count_zero_padded(self):
        assert build_step_name(StepNameComponents(well='A1', scan_count=7)) == 'A1_0007'

    def test_large_scan_count_round_trips(self):
        # scan_count renders as 4-OR-MORE digits (zero-pad is a minimum width),
        # so a long timelapse (>= 10000 scans) must still parse back.
        name = build_step_name(StepNameComponents(well='A1', channel='BF', scan_count=12345))
        assert name == 'A1_BF_12345'
        assert parse_step_name(name, known_layers=_KNOWN_LAYERS).scan_count == 12345
