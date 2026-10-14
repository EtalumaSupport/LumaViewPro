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
