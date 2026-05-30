# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#535 regression: per-well z values must carry over when clicking New.

Bug
---
Protocol.from_config built labware-derived positions with z=None for
every well, then fell back to layer_config['focus']. Clicking New
blew away whatever per-well z the user had tuned. dimin asked for
the focus values to survive.

Fix
---
Protocol.from_config now reads input_config['previous_well_z'] (a
dict of well_label -> z). For labware-derived positions, the per-well
z is the carry-over value if present, else None (falls through to
the layer focus as before).

ProtocolSettings.new_protocol builds the previous_well_z map from
self._protocol.steps() before calling create_protocol, taking the
first row's Z per well as the per-well base focus.

Test approach
-------------
1. Functional test on Protocol.from_config: build a config with
   previous_well_z={'A1': 5.0, 'B2': 7.5} on a 96-well plate; assert
   the resulting steps have Z=5.0 for A1, Z=7.5 for B2, and
   layer_config['focus'] for the rest.
2. AST structural lock on ProtocolSettings.new_protocol so the
   carry-over extraction runs before create_protocol.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'
TILING_CONFIGS = REPO / 'data' / 'tiling.json'


def _build_input_config(previous_well_z=None):
    """Minimal valid input_config for Protocol.from_config."""
    cfg = {
        'labware_id': '96 well microplate',
        'objective_id': '4x Oly',
        'zstack_params': {'range': 0, 'step_size': 0, 'z_reference': 'center'},
        'use_zstacking': False,
        'tiling': '1x1',
        'layer_configs': {
            'BF': {
                'acquire': 'image',
                'autofocus': False,
                'false_color': False,
                'illumination_ma': 50.0,
                'gain_db': 1.0,
                'auto_gain': False,
                'exposure_ms': 10.0,
                'sum': 1,
                'focus': 100.0,
                'video_config': {'duration': 30},
                'stim_config': None,
            },
        },
        'period': None,
        'duration': None,
        'frame_dimensions': {'width': 2048, 'height': 2048},
        'binning_size': 1,
        'stim_config': {},
    }
    if previous_well_z is not None:
        cfg['previous_well_z'] = previous_well_z
    return cfg


def test_protocol_from_config_applies_previous_well_z():
    """Wells named in previous_well_z get those values; others fall back."""
    from modules.protocol import Protocol

    cfg = _build_input_config(previous_well_z={'A1': 5.0, 'B2': 7.5})
    protocol = Protocol.from_config(input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS)
    df = protocol.steps()

    a1_z = df.loc[df['Well'] == 'A1', 'Z'].iloc[0]
    b2_z = df.loc[df['Well'] == 'B2', 'Z'].iloc[0]
    c3_z = df.loc[df['Well'] == 'C3', 'Z'].iloc[0]

    assert a1_z == pytest.approx(5.0, abs=1e-6), f'A1 z={a1_z} expected 5.0 (carry-over)'
    assert b2_z == pytest.approx(7.5, abs=1e-6), f'B2 z={b2_z} expected 7.5 (carry-over)'
    assert c3_z == pytest.approx(100.0, abs=1e-6), (
        f'C3 z={c3_z} expected 100.0 (layer focus fallback for un-carried wells)'
    )


def test_protocol_from_config_no_previous_well_z_falls_through():
    """No previous_well_z key -> every well uses layer_config focus."""
    from modules.protocol import Protocol

    cfg = _build_input_config()  # no previous_well_z
    protocol = Protocol.from_config(input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS)
    df = protocol.steps()

    # Every well's z is the layer focus (100.0).
    assert (df['Z'] == 100.0).all(), (
        'Without previous_well_z, every step Z should be the layer focus '
        '(100.0); got unique values: {}'.format(df['Z'].unique())
    )


def test_protocol_from_config_empty_previous_well_z_falls_through():
    """Empty dict treated the same as missing key."""
    from modules.protocol import Protocol

    cfg = _build_input_config(previous_well_z={})
    protocol = Protocol.from_config(input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS)
    df = protocol.steps()
    assert (df['Z'] == 100.0).all()


def _new_protocol_method() -> ast.FunctionDef:
    source = PROTOCOL_SETTINGS_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ProtocolSettings':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'new_protocol':
                    return child
    raise AssertionError('ProtocolSettings.new_protocol not found')


def test_new_protocol_extracts_previous_well_z_before_create():
    """The UI handler must extract per-well z from self._protocol and
    pass via config['previous_well_z'] before calling create_protocol."""
    method = _new_protocol_method()
    src = ast.unparse(method)
    assert "config['previous_well_z']" in src or 'config["previous_well_z"]' in src, (
        'new_protocol must populate config["previous_well_z"] from '
        'self._protocol.steps() before create_protocol. (#535)'
    )
    assert ('groupby' in src and "'Well'" in src) or '"Well"' in src, (
        'new_protocol must groupby Well to build the per-well z map. (#535)'
    )

    # Ordering: assignment to config must run BEFORE the create_protocol call.
    assign_idx = -1
    create_idx = -1
    for i, stmt in enumerate(method.body):
        unparsed = ast.unparse(stmt)
        if assign_idx == -1 and 'previous_well_z' in unparsed and 'config' in unparsed:
            assign_idx = i
        if create_idx == -1 and 'create_protocol' in unparsed:
            create_idx = i
    assert assign_idx >= 0, 'previous_well_z assignment not found in new_protocol body'
    assert create_idx >= 0, 'create_protocol call not found in new_protocol body'
    assert assign_idx < create_idx, (
        f'previous_well_z must be set at statement {assign_idx} BEFORE '
        f'create_protocol at statement {create_idx}. (#535)'
    )
