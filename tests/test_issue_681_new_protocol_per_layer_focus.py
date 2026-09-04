# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""New Protocol resets to per-channel focus; the carry-over consumer is dormant.

New Protocol resets every step to its channel's saved focus baseline. The
per-well Z carry-over (which the producer used to populate on New) was removed
as the default because it overrode a freshly-saved focus with autofocus-refined
Z. Protocol.from_config still honors an explicitly-passed previous_well_z map,
kept dormant for a future opt-in -- and when given such a map it must key by
(well, channel) so each channel keeps its own Z rather than inheriting a
sibling channel's. These functional tests lock that dormant-machinery contract;
the producer test confirms New no longer builds the carry-over.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'
TILING_CONFIGS = REPO / 'data' / 'tiling.json'


def _layer(focus: float) -> dict:
    return {
        'acquire': 'image',
        'autofocus': False,
        'false_color': False,
        'illumination_ma': 50.0,
        'gain_db': 1.0,
        'auto_gain': False,
        'exposure_ms': 10.0,
        'sum': 1,
        'focus': focus,
        'video_config': {'duration': 30},
        'stim_config': None,
    }


def _build_input_config(previous_well_z=None):
    """input_config with three channels at distinct focus (BF/Green/Red)."""
    cfg = {
        'labware_id': '96 well microplate',
        'objective_id': '4x Oly',
        'zstack_params': {'range': 0, 'step_size': 0, 'z_reference': 'center'},
        'use_zstacking': False,
        'tiling': '1x1',
        'layer_configs': {
            'BF': _layer(100.0),
            'Green': _layer(200.0),
            'Red': _layer(300.0),
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


def _well_channel_z(df, well, color):
    sel = df[(df['Well'] == well) & (df['Color'] == color)]
    return sel['Z'].iloc[0]


def test_new_protocol_keeps_each_channel_tuned_z(scale_capabilities):
    """Per-(well, channel) carry-over: each channel keeps its own tuned Z."""
    from modules.protocol import Protocol

    prev = {('A1', 'BF'): 7000.0, ('A1', 'Green'): 8000.0, ('A1', 'Red'): 9000.0}
    cfg = _build_input_config(previous_well_z=prev)
    df = Protocol.from_config(
        input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS, capabilities=scale_capabilities
    ).steps()

    assert _well_channel_z(df, 'A1', 'BF') == pytest.approx(7000.0, abs=1e-6)
    assert _well_channel_z(df, 'A1', 'Green') == pytest.approx(8000.0, abs=1e-6)
    assert _well_channel_z(df, 'A1', 'Red') == pytest.approx(9000.0, abs=1e-6)


def test_tuned_channel_does_not_clobber_siblings(scale_capabilities):
    """The core #681 guarantee: tuning one channel must not paste its Z onto
    the other channels -- untuned channels keep their own focus default."""
    from modules.protocol import Protocol

    # Only BF tuned in A1; Green/Red were never focused there.
    cfg = _build_input_config(previous_well_z={('A1', 'BF'): 7000.0})
    df = Protocol.from_config(
        input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS, capabilities=scale_capabilities
    ).steps()

    assert _well_channel_z(df, 'A1', 'BF') == pytest.approx(7000.0, abs=1e-6)
    # Pre-fix these were 7000 (clobbered); now each keeps its focus default.
    assert _well_channel_z(df, 'A1', 'Green') == pytest.approx(200.0, abs=1e-6)
    assert _well_channel_z(df, 'A1', 'Red') == pytest.approx(300.0, abs=1e-6)


def test_no_carry_over_uses_each_channels_focus(scale_capabilities):
    """No carry-over -> every channel uses its own focus default."""
    from modules.protocol import Protocol

    cfg = _build_input_config()  # no previous_well_z
    df = Protocol.from_config(
        input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS, capabilities=scale_capabilities
    ).steps()

    assert (df[df['Color'] == 'BF']['Z'] == 100.0).all()
    assert (df[df['Color'] == 'Green']['Z'] == 200.0).all()
    assert (df[df['Color'] == 'Red']['Z'] == 300.0).all()


def _new_protocol_src() -> str:
    """Unparsed code of ProtocolSettings.new_protocol (comments stripped, so a
    dormant-machinery mention in a comment does not trip the producer lock)."""
    tree = ast.parse(PROTOCOL_SETTINGS_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ProtocolSettings':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'new_protocol':
                    return ast.unparse(child)
    raise AssertionError('ProtocolSettings.new_protocol not found')


def test_producer_does_not_build_carry_over():
    """Structural lock on the producer: New Protocol must NOT build a
    previous_well_z carry-over -- each step resets to its channel's saved focus
    baseline. (The dormant from_config consumer is exercised by the functional
    tests above for the future opt-in.)"""
    src = _new_protocol_src()
    assert 'previous_well_z' not in src, (
        'new_protocol must not build a previous_well_z carry-over; New resets '
        'each step to the per-channel focus baseline.'
    )
