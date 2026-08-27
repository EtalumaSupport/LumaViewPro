# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The per-layer key rename carries real user values across, on every read path.

`settings[layer]['ill_ma'/'exp_ms']` became `illumination_ma`/`exposure_ms`
when the storage dict became the L2 API surface. The failure these tests
exist to catch is silent: the settings.json default-merge only ADDS missing
keys, so an install carrying ill_ma = 25.0 would get the shipped
illumination_ma = 5.0 merged in beside it and come up on the default while
the real value sat unread one key away. Nothing raises; the user just finds
their illumination changed.
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from modules import config_helpers, settings_init
from tests.test_settings_schema_parity import MIGRATED_PER_LAYER, RUNTIME_ONLY_PER_LAYER

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO_ROOT, 'data', 'settings.json')


@pytest.fixture(autouse=True)
def _restore_settings_globals():
    """Put settings_init's module globals back after every test here.

    ``load_lvp_settings`` writes ``settings_init.settings`` and
    ``rejected_current_json`` and leaves them set for the rest of the
    process. Other tests read that first global: a simulated Lumascope
    takes its model from ``settings.get('microscope', 'LS850T')``, so
    leaking a loaded config downgrades the simulated scope from the
    turret-bearing default to whatever the template ships, and turret
    tests fail depending on collection order rather than on their own
    subject.
    """
    saved_settings = settings_init.settings
    saved_rejected = settings_init.rejected_current_json
    try:
        yield
    finally:
        settings_init.settings = saved_settings
        settings_init.rejected_current_json = saved_rejected


_COMMON = {
    'gain_db': 1.0,
    'acquire': None,
    'autofocus': False,
    'false_color': False,
    'focus': 4950.0,
    'sum': 1,
    'auto_gain': False,
    'file_root': 'BF_',
    'video_config': {'duration': 5, 'fps': 30},
}

OLD_SHAPE = {**_COMMON, 'ill_ma': 25.0, 'exp_ms': 14.56}
NEW_SHAPE = {**_COMMON, 'illumination_ma': 7.0, 'exposure_ms': 8.0}
# What a downgrade leaves behind: the new names hold the values the previous
# upgrade wrote, the old names hold what the user set on the downgraded build.
BOTH_SHAPES = {
    **_COMMON,
    'illumination_ma': 25.0,
    'exposure_ms': 14.56,
    'ill_ma': 111.0,
    'exp_ms': 33.0,
}


def _appdata_with(tmp_dir, bf_layer):
    """Lay out a data/ dir holding the shipped template and a current.json."""
    data_dir = os.path.join(tmp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(TEMPLATE, os.path.join(data_dir, 'settings.json'))
    with open(TEMPLATE) as template_file:
        current = json.load(template_file)
    current['BF'] = bf_layer
    with open(os.path.join(data_dir, 'current.json'), 'w') as current_file:
        json.dump(current, current_file)
    return tmp_dir


@pytest.mark.parametrize(
    'bf_layer, expected_illumination, expected_exposure, why',
    [
        (OLD_SHAPE, 25.0, 14.56, 'old spellings carry their values, not template defaults'),
        (NEW_SHAPE, 7.0, 8.0, 'an already-migrated file is untouched'),
        (
            BOTH_SHAPES,
            111.0,
            33.0,
            'both names present: the OLD one wins, because a build carrying the '
            'migration never writes it -- finding it proves an older build wrote '
            'this file more recently, so its value is the fresher one',
        ),
    ],
    ids=['old-shape', 'already-migrated', 'both-names-after-rollback'],
)
def test_gui_load_carries_layer_values(bf_layer, expected_illumination, expected_exposure, why):
    with tempfile.TemporaryDirectory() as tmp_dir:
        _appdata_with(tmp_dir, bf_layer)
        settings_init.load_lvp_settings(MagicMock(), tmp_dir)
        loaded = settings_init.settings['BF']

    assert loaded['illumination_ma'] == expected_illumination, why
    assert loaded['exposure_ms'] == expected_exposure, why
    leftover = sorted(MIGRATED_PER_LAYER & set(loaded))
    assert not leftover, f'old spellings survived the load: {leftover}'


def test_headless_session_read_path_carries_layer_values():
    """The L2 entry point reads the file itself and never calls load_settings.

    ``ScopeSession.create_headless`` resolves and reads current.json directly
    when no settings have been loaded into the process -- the normal state in
    a fresh REST or CLI process. A migration living in the GUI's load path
    would never run here, and ``get_layer_configs`` would raise KeyError on
    every pre-upgrade config.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _appdata_with(tmp_dir, OLD_SHAPE)
        settings_path = settings_init._resolve_settings_path(tmp_dir)
        settings = settings_init.read_settings_json(settings_path)

    assert os.path.basename(settings_path) == 'current.json'
    config = config_helpers.get_layer_configs(settings, ['BF'])['BF']
    assert config['illumination_ma'] == 25.0
    assert config['exposure_ms'] == 14.56


def test_old_shape_current_json_passes_per_layer_parity():
    """A stale current.json must not read as schema drift.

    The parity test is skipped when no current.json exists, so on CI it
    asserts nothing about this. Constructing the old-shape file here makes
    the exemption a real guard rather than a property of one developer's
    machine.
    """
    with open(TEMPLATE) as template_file:
        template = json.load(template_file)
    current = json.loads(json.dumps(template))
    current['BF'] = OLD_SHAPE

    drift = []
    for key in set(template) & set(current):
        template_value, current_value = template[key], current[key]
        if not (isinstance(template_value, dict) and isinstance(current_value, dict)):
            continue
        allowed = RUNTIME_ONLY_PER_LAYER.get(key, set()) | MIGRATED_PER_LAYER
        current_only = set(current_value) - set(template_value) - allowed
        if current_only:
            drift.append(f'{key}: {sorted(current_only)}')

    assert not drift, f'old-shape current.json read as drift: {drift}'
