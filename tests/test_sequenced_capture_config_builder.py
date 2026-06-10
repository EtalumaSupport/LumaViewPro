# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Tests for the canonical sequenced-capture config builder.

The input_config dict for Protocol.from_config used to be assembled three ways
(UI / settings / zstack); the settings lane omitted tiling_overlap_percent,
silently forcing 0% overlap. build_sequenced_capture_config is now the single
assembler, so the key set cannot drift between lanes and the overlap key is
always present.
"""

import pytest

from modules.config_helpers import build_sequenced_capture_config

# Mirrors what Protocol.from_config consumes (see protocol.py).
_CANONICAL_KEYS = {
    'labware_id', 'objective_id', 'zstack_params', 'use_zstacking', 'tiling',
    'tiling_overlap_percent', 'layer_configs', 'period', 'duration',
    'frame_dimensions', 'binning_size', 'stim_config',
}


def _values(**overrides):
    base = {
        'labware_id': 'plate_a',
        'objective_id': 'obj_4x',
        'zstack_params': {},
        'use_zstacking': False,
        'tiling': '1x1',
        'layer_configs': {},
        'period': None,
        'duration': None,
        'frame_dimensions': {'width': 800, 'height': 600},
        'binning_size': 1,
        'stim_config': {},
    }
    base.update(overrides)
    return base


def test_output_contains_full_canonical_key_set():
    config = build_sequenced_capture_config(_values(tiling_overlap_percent=10.0))
    assert _CANONICAL_KEYS.issubset(config.keys())


def test_tiling_overlap_defaults_to_zero_when_omitted():
    # The bug class: a source lane that forgets the key must still yield a
    # config that runs at the explicit default, not silently re-derive 0%.
    config = build_sequenced_capture_config(_values())
    assert 'tiling_overlap_percent' in config
    assert config['tiling_overlap_percent'] == 0.0


def test_tiling_overlap_preserved_when_supplied():
    config = build_sequenced_capture_config(_values(tiling_overlap_percent=25.0))
    assert config['tiling_overlap_percent'] == 25.0


def test_positions_included_only_when_supplied():
    with_positions = build_sequenced_capture_config(
        _values(positions=[{'name': 'ZStack'}])
    )
    assert with_positions['positions'] == [{'name': 'ZStack'}]

    without_positions = build_sequenced_capture_config(_values())
    assert 'positions' not in without_positions


def test_missing_required_key_raises():
    incomplete = _values()
    del incomplete['labware_id']
    with pytest.raises(KeyError):
        build_sequenced_capture_config(incomplete)
