# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The installation's own files, copied into a test's data folder.

A scope reads every file the installation ships from the folder it is
started on, before anything starts, and refuses to come up without them.
A test that builds a session on a folder of its own copies them here, so
a new installation file is added in one place rather than at every site.
"""

import functools
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


@functools.cache
def shipped_grids():
    """The shipped tiling.json's grids, read as a scope reads them at bring-up.

    For a protocol a test builds without a scope; a scope's own protocols are
    built on the copy it read (``ProtocolsAPI.tiling_grids``). Read once, since
    the grids never change after they are read.
    """
    from modules.tiling_config import TilingGrids

    return TilingGrids(SHIPPED_DATA / 'tiling.json')


def _fluorescence_record(key: str, color: str, step_order: int) -> dict:
    return {
        'key': key,
        'modality': 'fluorescence',
        'background': 'dark',
        'manual_max_exposure_ms': 1000,
        'manual_max_illumination_ma': None,
        'display_color': color,
        'step_order': step_order,
        'sim_light_per_ma_s': 0.0126,
    }


def _row(key: str, led: int, nm: int | None) -> dict:
    """A model's row for ``key``: a fluorescence row states its excitation and
    no name, any other its name."""
    if nm is not None:
        return {'key_name': key, 'led_channel': led, 'excitation_nm': nm}
    return {'key_name': key, 'display_name': key, 'led_channel': led, 'excitation_nm': None}


# Two fluorescence channels the release does not ship, on the two LED
# addresses no shipped model drives, in colours no shipped channel has.
ADDED_CHANNELS = (
    _fluorescence_record('Ex445', '#00ffff', 9),
    _fluorescence_record('Ex730', '#ff00ff', 10),
)

# Three units on that release, by how many fluorescence channels each
# carries: none, the two added ones, and the stock three with them.
_BF = _row('BF', 3, None)
FLUORESCENCE_UNITS = {
    'LS850-F0': [_BF, _row('PC', 4, None), _row('DF', 5, None)],
    'LS850-F2': [_BF, _row('Ex445', 6, 445), _row('Ex730', 7, 730)],
    'LS850-F5': [
        _BF,
        _row('Ex405', 0, 405),
        _row('Ex488', 1, 488),
        _row('Ex589', 2, 589),
        _row('Ex445', 6, 445),
        _row('Ex730', 7, 730),
    ],
}


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


def write_installation(root: pathlib.Path, *, board_block: dict | None = None) -> pathlib.Path:
    """An installation at ``root``: the shipped files, and ``board_block``, a
    per-unit block its motor board carries (as ``non_stock_block`` gives), in
    the motor config every simulated board of the installation starts from.
    """
    data = pathlib.Path(root) / 'data'
    data.mkdir(parents=True)
    copy_installation_files(data)
    if board_block is not None:
        defaults = json.loads((data / 'motorconfig_defaults.json').read_text(encoding='utf-8'))
        (data / 'motorconfig_defaults.json').write_text(
            json.dumps({**defaults, **board_block}), encoding='utf-8'
        )
    return pathlib.Path(root)


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
