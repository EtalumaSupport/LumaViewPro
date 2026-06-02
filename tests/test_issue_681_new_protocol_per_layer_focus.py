# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: New Protocol keeps each channel's own focus.

A New Protocol carries tuned Z forward from the prior in-memory protocol so
focus work is not lost. That carry-over used to be keyed by WELL only -- the
first row's Z (one channel) was pasted onto every channel in the well, so
three channels focused at 7000 / 8000 / 9000 all collapsed to 7000 after New.

The carry-over is now keyed by (well, channel): each channel keeps its own
tuned Z, and a channel the user never tuned falls back to its own focus
default rather than inheriting a sibling channel's Z.

Companion to the per-well-survival behavior (a tuned Z surviving New) and the
Add/insert-step per-channel behavior, which the matching test files cover.
"""

from __future__ import annotations

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


def test_new_protocol_keeps_each_channel_tuned_z():
    """Per-(well, channel) carry-over: each channel keeps its own tuned Z."""
    from modules.protocol import Protocol

    prev = {('A1', 'BF'): 7000.0, ('A1', 'Green'): 8000.0, ('A1', 'Red'): 9000.0}
    cfg = _build_input_config(previous_well_z=prev)
    df = Protocol.from_config(input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS).steps()

    assert _well_channel_z(df, 'A1', 'BF') == pytest.approx(7000.0, abs=1e-6)
    assert _well_channel_z(df, 'A1', 'Green') == pytest.approx(8000.0, abs=1e-6)
    assert _well_channel_z(df, 'A1', 'Red') == pytest.approx(9000.0, abs=1e-6)


def test_tuned_channel_does_not_clobber_siblings():
    """The core #681 guarantee: tuning one channel must not paste its Z onto
    the other channels -- untuned channels keep their own focus default."""
    from modules.protocol import Protocol

    # Only BF tuned in A1; Green/Red were never focused there.
    cfg = _build_input_config(previous_well_z={('A1', 'BF'): 7000.0})
    df = Protocol.from_config(input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS).steps()

    assert _well_channel_z(df, 'A1', 'BF') == pytest.approx(7000.0, abs=1e-6)
    # Pre-fix these were 7000 (clobbered); now each keeps its focus default.
    assert _well_channel_z(df, 'A1', 'Green') == pytest.approx(200.0, abs=1e-6)
    assert _well_channel_z(df, 'A1', 'Red') == pytest.approx(300.0, abs=1e-6)


def test_no_carry_over_uses_each_channels_focus():
    """No carry-over -> every channel uses its own focus default."""
    from modules.protocol import Protocol

    cfg = _build_input_config()  # no previous_well_z
    df = Protocol.from_config(input_config=cfg, tiling_configs_file_loc=TILING_CONFIGS).steps()

    assert (df[df['Color'] == 'BF']['Z'] == 100.0).all()
    assert (df[df['Color'] == 'Green']['Z'] == 200.0).all()
    assert (df[df['Color'] == 'Red']['Z'] == 300.0).all()


def test_producer_keys_carry_over_by_well_and_channel():
    """Structural lock on the producer: the New-Protocol handler must build
    previous_well_z keyed by (Well, Color), not by Well alone (a per-well key
    is exactly what pasted one channel's Z onto every channel)."""
    src = PROTOCOL_SETTINGS_SRC.read_text()
    assert "groupby(['Well', 'Color']" in src or 'groupby(["Well", "Color"]' in src, (
        'new_protocol must group the carry-over by (Well, Color) so each '
        'channel keeps its own tuned Z.'
    )
    assert "groupby('Well', sort=False)" not in src, (
        'the per-well-only carry-over (the #681 clobber source) must be gone.'
    )
