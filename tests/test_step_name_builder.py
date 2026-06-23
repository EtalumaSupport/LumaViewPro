# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the canonical step-name builder/parser (StepNameComponents).

Three guarantees:

1. EQUIVALENCE -- for a fresh build (a clean well or custom_prefix base, the
   way every legitimate build site calls it), build_step_name reproduces the
   legacy generate_default_step_name byte-for-byte. This is the safety net for
   migrating the 12 call sites: swapping the builder cannot change any filename
   written to disk for a correctly-named step.

2. IDEMPOTENCE / replace-not-append -- rebuilding after changing one component
   yields exactly one token for that component, where the legacy append-if-
   absent builder left the stale token beside the new one when a built name was
   fed back as the seed. This is the structural fix for the channel-change and
   stitch-filename corruption.

3. ROUND-TRIP -- build_step_name(parse_step_name(s)) == s for any canonical s.
"""

import dataclasses

import pytest

from modules.common_utils import (
    StepNameComponents,
    build_step_name,
    generate_default_step_name,
    parse_step_name,
)

_KNOWN_LAYERS = ['Blue', 'Green', 'Red', 'BF', 'PC', 'DF', 'Lumi']
_KNOWN_OBJECTIVES = ['10x', '4x', '20x']


def _to_legacy_kwargs(c: StepNameComponents) -> dict:
    """Map components to the legacy generate_default_step_name keyword set."""
    kwargs = {
        'well_label': c.well,
        'custom_name_prefix': c.custom_prefix or None,
        'color': c.channel,
        'tile_label': c.tile,
        'objective_short_name': c.objective,
        'turret_position': c.turret_position,
        'z_height_idx': c.z_index,
        'scan_count': c.scan_count,
    }
    for p in c.post:
        if p == 'stitched':
            kwargs['stitched'] = True
        elif p == 'video':
            kwargs['video'] = True
        elif p == 'stack':
            kwargs['stack'] = True
        elif p == 'hyperstack':
            kwargs['hyperstack'] = True
        elif p.startswith('zproj_'):
            kwargs['zprojection'] = p[len('zproj_') :]
    return kwargs


# Fresh-build component cases mirroring the real call sites. Each must render
# identically through both builders (no token is pre-baked into the base, so
# the legacy substring guards never fire).
_FRESH_CASES = [
    StepNameComponents(well='A2', channel='BF'),
    StepNameComponents(well='A1', channel='Green'),
    StepNameComponents(well='H12', channel='Red', tile='A1'),
    StepNameComponents(well='A1', channel='BF', objective='10x'),
    StepNameComponents(well='A1', channel='BF', turret_position=2),
    StepNameComponents(well='A1', channel='BF', z_index=5),
    StepNameComponents(well='A1', channel='BF', tile='B2', z_index=3),
    StepNameComponents(well='A1', channel='BF', scan_count=0),
    StepNameComponents(well='A1', channel='BF', scan_count=12),
    StepNameComponents(well='A1', channel='BF', scan_count=12345),  # 5+ digits
    StepNameComponents(well='A1', channel='BF', tile='A1', z_index=2, scan_count=7),
    StepNameComponents(custom_prefix='custom0001', channel='BF'),
    StepNameComponents(custom_prefix='custom0042', channel='Red', tile='C3'),
    StepNameComponents(well='A1', channel='BF', post=('stitched',)),
    StepNameComponents(well='A1', channel='Composite', post=('stitched',)),
    StepNameComponents(well='A1', channel='BF', post=('video',)),
    StepNameComponents(well='A1', post=('hyperstack',)),
    StepNameComponents(well='A1', channel='BF', post=('stack',)),
    StepNameComponents(well='A1', channel='BF', post=('zproj_median',)),
    # Chained post-outputs: a stitch then z-projected / video'd carries both
    # suffixes in order. The legacy builder appends stitched then zproj/video,
    # so equivalence holds only for the canonical (stitched-first) ordering.
    StepNameComponents(well='A1', channel='BF', post=('stitched', 'zproj_median')),
    StepNameComponents(well='A1', channel='BF', post=('stitched', 'video')),
    StepNameComponents(well='A1'),  # bare well, no channel yet
]


class TestEquivalenceToLegacy:
    @pytest.mark.parametrize('c', _FRESH_CASES)
    def test_build_matches_generate_default_step_name(self, c):
        legacy = generate_default_step_name(**_to_legacy_kwargs(c))
        assert build_step_name(c) == legacy, (
            f'new builder diverged from legacy for {c}: {build_step_name(c)!r} != {legacy!r}'
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
