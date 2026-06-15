"""Regression: a multi-channel Add Step takes each channel's own saved
focus, not the single live stage Z.

On Blank labware the Add-Step flow builds one protocol step per enabled
channel by calling Protocol.insert_step in a loop, passing the SAME
plate_position (the current physical stage Z) to every channel. The bug
was that insert_step used plate_position['z'] for the step Z, so three
channels saved at focus 7000 / 8000 / 9000 all collapsed to the single
current stage Z.

The fix mirrors the labware build path (from_config): the step Z comes
from layer_config['focus'] per channel, falling back to plate_position
['z'] only when a layer has no saved focus.
"""

from __future__ import annotations

import datetime
import pathlib

import pytest

from modules.protocol import Protocol

TILING_PATH = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'

PLATE_X, PLATE_Y = 50.0, 40.0
LIVE_STAGE_Z = 5000.0  # the single current stage Z passed for every channel


def _layer_config(focus, *, include_focus=True):
    cfg = {
        'autofocus': False,
        'false_color': False,
        'illumination_ma': 100.0,
        'gain_db': 10.0,
        'auto_gain': False,
        'exposure_ms': 5.0,
        'sum': 1,
        'acquire': 'image',
        'video_config': {'duration': 5, 'fps': 30},
    }
    if include_focus:
        cfg['focus'] = focus
    return cfg


def _empty_protocol():
    """A protocol with no steps but valid header config."""
    import pandas as pd

    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': pd.DataFrame(columns=list(Protocol.CURRENT_COLUMNS)),
        'period': datetime.timedelta(minutes=20.0),
        'duration': datetime.timedelta(hours=48.0),
        'labware_id': 'Blank',
        'capture_root': '',
        'tiling': '1x1',
        'custom_step_count': 0,
    }
    return Protocol(tiling_configs_file_loc=TILING_PATH, config=config)


def _insert(protocol, layer, layer_config, after_step):
    return protocol.insert_step(
        step_name=f'{layer}_step',
        layer=layer,
        layer_config=layer_config,
        plate_position={'x': PLATE_X, 'y': PLATE_Y, 'z': LIVE_STAGE_Z},
        objective_id='4x Oly',
        stim_configs={},
        before_step=None if protocol.num_steps() else 0,
        after_step=(protocol.num_steps() - 1) if protocol.num_steps() else None,
    )


def test_each_channel_step_uses_its_own_saved_focus():
    protocol = _empty_protocol()

    focuses = {'BF': 7000.0, 'Green': 8000.0, 'Red': 9000.0}
    for layer, focus in focuses.items():
        _insert(protocol, layer, _layer_config(focus), after_step=None)

    assert protocol.num_steps() == 3
    got = {
        protocol.step(idx=i)['Color']: protocol.step(idx=i)['Z']
        for i in range(protocol.num_steps())
    }
    for layer, focus in focuses.items():
        assert got[layer] == pytest.approx(focus), (
            f'{layer} step Z={got[layer]} but its saved focus is {focus}; '
            f'a multi-channel Add Step must use per-layer focus, not the '
            f'live stage Z ({LIVE_STAGE_Z})'
        )
    # The bug signature was every channel collapsing to the live stage Z.
    assert not all(z == pytest.approx(LIVE_STAGE_Z) for z in got.values())


def test_falls_back_to_stage_z_when_focus_unset():
    protocol = _empty_protocol()
    _insert(protocol, 'BF', _layer_config(None, include_focus=False), after_step=None)

    assert protocol.num_steps() == 1
    assert protocol.step(idx=0)['Z'] == pytest.approx(LIVE_STAGE_Z), (
        'when a layer has no saved focus, the step Z falls back to the live stage Z'
    )
