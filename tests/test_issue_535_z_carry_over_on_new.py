# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""New Protocol resets per-channel focus; the per-well z carry-over is dormant.

Behavior
--------
New Protocol resets every step to its channel's saved focus baseline
(layer_config['focus']). A per-(well, channel) Z carry-over from the prior
in-memory protocol was added on an external request, but it harvested
autofocus-refined Z along with user-tuned Z and overrode a freshly-saved
focus, so it was removed as the default. Per-well focus is re-established on
demand via "Autofocus All Steps".

The Protocol.from_config consumer still honors an explicitly-passed
previous_well_z map, kept dormant so the carry-over can return as an opt-in
setting without re-plumbing.

Test approach
-------------
1. Functional tests on Protocol.from_config lock the dormant opt-in
   machinery: an explicit previous_well_z map is applied; an empty/missing
   map falls back to layer_config['focus'].
2. AST structural lock on ProtocolSettings.new_protocol: it must NOT build a
   previous_well_z carry-over (New resets to the per-channel baseline).
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


def test_protocol_from_config_applies_previous_well_z(scale_capabilities):
    """(well, channel) entries get their carried-over Z; others fall back.

    The carry-over is keyed by (well, channel) so each channel keeps its own
    tuned Z (a per-well key pasted one channel's Z onto every channel -- #681).
    This config has a single BF layer, so the key is (well, 'BF')."""
    from modules.protocol import Protocol

    cfg = _build_input_config(previous_well_z={('A1', 'BF'): 5.0, ('B2', 'BF'): 7.5})
    protocol = Protocol.from_config(
        input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS, capabilities=scale_capabilities
    )
    df = protocol.steps()

    a1_z = df.loc[df['Well'] == 'A1', 'Z'].iloc[0]
    b2_z = df.loc[df['Well'] == 'B2', 'Z'].iloc[0]
    c3_z = df.loc[df['Well'] == 'C3', 'Z'].iloc[0]

    assert a1_z == pytest.approx(5.0, abs=1e-6), f'A1 BF z={a1_z} expected 5.0 (carry-over)'
    assert b2_z == pytest.approx(7.5, abs=1e-6), f'B2 BF z={b2_z} expected 7.5 (carry-over)'
    assert c3_z == pytest.approx(100.0, abs=1e-6), (
        f'C3 z={c3_z} expected 100.0 (layer focus fallback for un-carried wells)'
    )


def test_protocol_from_config_no_previous_well_z_falls_through(scale_capabilities):
    """No previous_well_z key -> every well uses layer_config focus."""
    from modules.protocol import Protocol

    cfg = _build_input_config()  # no previous_well_z
    protocol = Protocol.from_config(
        input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS, capabilities=scale_capabilities
    )
    df = protocol.steps()

    # Every well's z is the layer focus (100.0).
    assert (df['Z'] == 100.0).all(), (
        'Without previous_well_z, every step Z should be the layer focus '
        '(100.0); got unique values: {}'.format(df['Z'].unique())
    )


def test_protocol_from_config_empty_previous_well_z_falls_through(scale_capabilities):
    """Empty dict treated the same as missing key."""
    from modules.protocol import Protocol

    cfg = _build_input_config(previous_well_z={})
    protocol = Protocol.from_config(
        input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS, capabilities=scale_capabilities
    )
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


def test_new_protocol_does_not_carry_over_z_by_default():
    """New Protocol must NOT build a previous_well_z carry-over: each step
    resets to its channel's saved focus baseline. The carry-over was removed
    as the default because it overrode a freshly-saved focus with
    autofocus-refined Z; the Protocol.from_config map is kept dormant for a
    future opt-in setting."""
    method = _new_protocol_method()
    src = ast.unparse(method)
    assert 'previous_well_z' not in src, (
        'new_protocol must not build a previous_well_z carry-over; New resets '
        'each step to the per-channel focus baseline.'
    )
