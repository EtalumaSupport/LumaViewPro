# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A channel whose focus was never saved is imaged at the current Z.

The shipped template gave every layer a focus of 4950.0, which the default
merge copied into each user's live settings, so a channel nobody focused
looked saved and every step built for it went to that height: Add, New and
a composite capture alike (bench, LS850T, 2026-10-02: BF saved at 5998.6,
Blue never saved, Blue's steps at 4950). A saved focus still wins (#681).
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest

from modules import settings_init
from modules.common_utils import get_layers
from modules.exceptions import ConfigError, FocusNotSavedError
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader


REPO = pathlib.Path(__file__).resolve().parent.parent
TILING_CONFIGS = REPO / 'data' / 'tiling.json'


def _layer(focus):
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


def _input_config(**extra):
    cfg = {
        'labware_id': '6 well microplate',
        'objective_id': '4x Oly',
        'zstack_params': {'range': 0, 'step_size': 0, 'z_reference': 'center'},
        'use_zstacking': False,
        'tiling': '1x1',
        'layer_configs': {'BF': _layer(5998.6), 'Blue': _layer(None)},
        'period': None,
        'duration': None,
        'frame_dimensions': {'width': 2048, 'height': 2048},
        'binning_size': 1,
        'stim_config': {},
    }
    cfg.update(extra)
    return cfg


def _from_config(cfg, capabilities):
    from modules.protocol import Protocol

    return Protocol.from_config(
        input_config=cfg,
        tiling_configs_file_loc=TILING_CONFIGS,
        capabilities=capabilities,
        objective_helper=ObjectiveLoader(),
        wellplate_loader=WellPlateLoader(),
    ).steps()


def test_the_template_saves_no_focus_for_any_layer():
    template = json.loads((REPO / 'data' / 'settings.json').read_text())

    assert {layer: template[layer]['focus'] for layer in get_layers()} == dict.fromkeys(
        get_layers()
    )


def test_a_stored_shipped_focus_is_read_as_never_saved():
    stored = {layer: {'focus': 4950.0} for layer in get_layers()}
    stored['BF']['focus'] = 5998.6357435197815

    forgotten = settings_init.forget_shipped_focus(stored)

    assert stored['BF']['focus'] == 5998.6357435197815
    assert forgotten == [layer for layer in get_layers() if layer != 'BF']
    assert all(stored[layer]['focus'] is None for layer in forgotten)


def test_the_load_migrations_read_a_stored_shipped_focus_as_never_saved(caplog):
    import logging

    stored = {'Blue': {'focus': 4950.0}, 'BF': {'focus': 5998.6}}

    with caplog.at_level(logging.INFO):
        settings_init._apply_load_migrations(logging.getLogger('test'), stored)

    assert stored['Blue']['focus'] is None
    assert stored['BF']['focus'] == 5998.6
    assert 'No focus was ever saved for Blue' in caplog.text


def test_new_images_an_unsaved_channel_at_the_current_z_and_a_saved_one_at_its_focus(
    scale_capabilities,
):
    steps = _from_config(_input_config(current_z=6100.0), scale_capabilities)

    assert set(steps[steps['Color'] == 'Blue']['Z']) == {6100.0}
    assert set(steps[steps['Color'] == 'BF']['Z']) == {5998.6}


def test_a_build_with_an_unsaved_channel_and_no_current_z_is_refused(scale_capabilities):
    with pytest.raises(ConfigError, match='No focus is saved for Blue'):
        _from_config(_input_config(), scale_capabilities)


def test_add_images_an_unsaved_channel_at_the_current_z():
    from modules.protocol import Protocol

    protocol = Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={'steps': pd.DataFrame(), 'custom_step_count': 0},
    )
    for layer, focus in (('BF', 5998.6), ('Blue', None)):
        protocol.insert_step(
            step_name=None,
            layer=layer,
            layer_config=_layer(focus),
            plate_position={'x': 1.0, 'y': 2.0, 'z': 6100.0},
            objective_id='4x Oly',
            stim_configs={},
            after_step=protocol.num_steps() - 1,
        )

    assert list(protocol.steps()['Z']) == [5998.6, 6100.0]


def test_the_saved_focus_of_a_channel_never_saved_is_refused():
    from modules.scope_session import ScopeSession

    session = SimpleNamespace(settings={'Blue': {'focus': None}, 'BF': {'focus': 5998.6}})

    assert ScopeSession.saved_focus(session, 'BF') == 5998.6
    with pytest.raises(FocusNotSavedError, match='No focus is saved for Blue'):
        ScopeSession.saved_focus(session, 'Blue')
