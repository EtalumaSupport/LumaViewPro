# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The installation's own files, copied into a test's data folder.

A scope reads every file the installation ships from the folder it is
started on, before anything starts, and refuses to come up without them.
A test that builds a session on a folder of its own copies them here, so
a new installation file is added in one place rather than at every site.
"""

import json
import pathlib
import shutil

SHIPPED_DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'

INSTALLATION_FILES = (
    'labware.json',
    'objectives.json',
    'scopes.json',
    'motorconfig_defaults.json',
    'settings.json',
    'tiling.json',
)


def copy_installation_files(data: pathlib.Path) -> None:
    """Copy the shipped installation files into ``data``, an existing data folder."""
    for name in INSTALLATION_FILES:
        shutil.copy(SHIPPED_DATA / name, pathlib.Path(data) / name)


def _fluorescence_record(key: str, color: str, step_order: int) -> dict:
    return {
        'key': key,
        'modality': 'fluorescence',
        'background': 'dark',
        'manual_max_exposure_ms': 1000,
        'display_color': color,
        'step_order': step_order,
        'sim_light_per_ma_s': 0.0126,
    }


def _row(key: str, led: int, nm: int | None) -> dict:
    return {'key_name': key, 'display_name': key, 'led_channel': led, 'excitation_nm': nm}


# Two fluorescence channels the release does not ship, on the two LED
# addresses no shipped model drives, in colours no shipped channel has.
ADDED_CHANNELS = (
    _fluorescence_record('Ex445', '#00ffff', 7),
    _fluorescence_record('Ex640', '#ff00ff', 8),
)

# Three units on that release, by how many fluorescence channels each
# carries: none, the two added ones, and the stock three with them.
_BF = _row('BF', 3, None)
FLUORESCENCE_UNITS = {
    'LS850-F0': [_BF, _row('PC', 4, None), _row('DF', 5, None)],
    'LS850-F2': [_BF, _row('Ex445', 6, 445), _row('Ex640', 7, 640)],
    'LS850-F5': [
        _BF,
        _row('Blue', 0, 405),
        _row('Green', 1, 488),
        _row('Red', 2, 589),
        _row('Ex445', 6, 445),
        _row('Ex640', 7, 640),
    ],
}


# The release once its fluorescence keys are named for their excitation: each
# old colour key's record under its stock filterset's key, every field
# unchanged, and the non-stock filterset's two other channels after them.
EXCITATION_KEYS = {'Blue': 'Ex405', 'Green': 'Ex488', 'Red': 'Ex589'}
OLD_CHANNEL_NAMES = {
    'Blue': {'led_address': 0, 'stock_key': 'Ex405'},
    'Green': {'led_address': 1, 'stock_key': 'Ex488'},
    'Red': {'led_address': 2, 'stock_key': 'Ex589'},
}
NON_STOCK_CHANNELS = (
    _fluorescence_record('Ex550', '#00ff00', 7),
    _fluorescence_record('Ex640', '#ff0000', 8),
)


def non_stock_block(transmitted: list[dict]) -> dict:
    """A board's Channels block for the 488 / 550 / 640 filterset on LEDs 0 / 1 / 2,
    shown blue, green and red, beside the unit's ``transmitted`` model rows."""
    channels = {
        row['key_name']: {k: v for k, v in row.items() if k != 'key_name'} for row in transmitted
    }
    for key, led, nm, color in (
        ('Ex488', 0, 488, '#0000ff'),
        ('Ex550', 1, 550, '#00ff00'),
        ('Ex640', 2, 640, '#ff0000'),
    ):
        channels[key] = {
            'display_name': None,
            'led_channel': led,
            'excitation_nm': nm,
            'display_color': color,
        }
    return {'Channels': channels, 'Filterset': '488/550/640'}


def write_release_in_excitation_keys(data: pathlib.Path) -> pathlib.Path:
    """Write ``data/scopes.json``: the shipped release with ``EXCITATION_KEYS``
    for its colour keys, ``NON_STOCK_CHANNELS`` added, and the old names in its
    ``OldChannelNames``.

    The caller installs it as the process's release (``use_release``).
    """
    scopes = json.loads((SHIPPED_DATA / 'scopes.json').read_text(encoding='utf-8'))
    scopes['LayerOrder'] = [
        {**record, 'key': EXCITATION_KEYS.get(record['key'], record['key'])}
        for record in scopes['LayerOrder']
    ] + list(NON_STOCK_CHANNELS)
    for entry in scopes['Models'].values():
        entry['Layers'] = [
            {
                **row,
                'key_name': EXCITATION_KEYS.get(row['key_name'], row['key_name']),
                'display_name': EXCITATION_KEYS.get(row['display_name'], row['display_name']),
            }
            for row in entry['Layers']
        ]
    scopes['OldChannelNames'] = OLD_CHANNEL_NAMES
    path = pathlib.Path(data) / 'scopes.json'
    path.write_text(json.dumps(scopes), encoding='utf-8')
    return path


def use_release(monkeypatch, scopes: pathlib.Path) -> None:
    """Make ``scopes`` the process's release: its layer catalogue and its old names.

    Both are read once per process from the installation's own folder, so a
    test replaces the two caches for its own duration.
    """
    from modules import layer_record
    from modules.path_utils import read_installation_file

    contents = read_installation_file(scopes)
    catalogue = layer_record.load_layer_catalogue(contents, scopes)
    monkeypatch.setattr(layer_record, '_CATALOGUE_CACHE', catalogue)
    monkeypatch.setattr(
        layer_record,
        '_OLD_NAMES_CACHE',
        layer_record.load_old_channel_names(contents, scopes, catalogue),
    )


def write_release_with_added_channels(data: pathlib.Path) -> pathlib.Path:
    """Write ``data/scopes.json``: the shipped release plus ``ADDED_CHANNELS``
    and the ``FLUORESCENCE_UNITS`` models, each an LS850 with its own rows.

    Adding a channel is a data change: one catalogue record and a unit's
    row. The caller replaces the process-wide catalogue with this file's
    (``layer_record._CATALOGUE_CACHE``), since the release catalogue is
    read from the installation's own folder, not the scope's.
    """
    scopes = json.loads((SHIPPED_DATA / 'scopes.json').read_text(encoding='utf-8'))
    scopes['LayerOrder'] = [*scopes['LayerOrder'], *ADDED_CHANNELS]
    for model, rows in FLUORESCENCE_UNITS.items():
        scopes['Models'][model] = {**scopes['Models']['LS850'], 'Layers': rows}
    path = pathlib.Path(data) / 'scopes.json'
    path.write_text(json.dumps(scopes), encoding='utf-8')
    return path
