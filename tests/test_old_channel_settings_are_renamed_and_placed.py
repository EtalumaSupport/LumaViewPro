# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A user's old channel settings keep meaning the LED they were made for.

A ``current.json`` written before the fluorescence keys were named for
their excitation holds ``Blue`` / ``Green`` / ``Red`` blocks. The settings
are prepared before any scope exists, so preparation renames each to its
stock key (``Ex405`` / ``Ex488`` / ``Ex589``) before anything reads one;
the session's bring-up then places them by the unit, first: on a unit
whose LED 0 is 488 nm, ``Blue``'s settings are its ``Ex488``'s.

Each test runs on the shipped release, whose keys are named for their
excitation.
"""

from __future__ import annotations

import json
import logging
import pathlib

import pytest

from modules import settings_init
from modules.path_utils import read_installation_file
from modules.scope_session import ScopeSession
from tests.installation_fixtures import (
    SHIPPED_DATA,
    non_stock_block,
    write_installation,
)
from tests.settings_fixtures import complete_settings

logger = logging.getLogger('test_old_channel_settings_are_renamed_and_placed')

# The fluorescence keys and the old names that stored them, in LED order.
_STOCK = ('Ex405', 'Ex488', 'Ex589')
_OLD = ('Blue', 'Green', 'Red')
# An order an earlier release derived, in its own keys.
_OLD_STEP_ORDER = ['BF', 'PC', 'DF', 'Green', 'Blue', 'Red', 'Lumi']


def _write_old_current_json(root: pathlib.Path, *, microscope: str) -> None:
    """A ``current.json`` as an earlier release saved it: each fluorescence
    block under its old name, with gains of its own, ``Blue`` acquiring with
    the retired shipped focus, and the step order in the old keys."""
    settings = complete_settings(live_folder=str(root / 'live'), microscope=microscope)
    for gain, (stock, old) in enumerate(zip(_STOCK, _OLD, strict=True), start=3):
        block = settings.pop(stock)
        settings[old] = {**block, 'gain_db': float(gain)}
    settings['Blue'] |= {'focus': 4950.0, 'acquire': 'image'}
    settings['step_channel_order'] = _OLD_STEP_ORDER
    for key in ('Ex550', 'Ex640'):
        settings.pop(key)
    (root / 'data' / 'current.json').write_text(json.dumps(settings))


def _gains(settings: dict, keys) -> list[float]:
    return [settings[key]['gain_db'] for key in keys]


def _transmitted_rows() -> list[dict]:
    """The shipped LS850's transmitted rows, which a unit's own block carries too."""
    rows = read_installation_file(SHIPPED_DATA / 'scopes.json')['Models']['LS850']['Layers']
    return [row for row in rows if row['key_name'] in ('BF', 'PC', 'DF')]


def test_preparation_renames_old_blocks_to_their_stock_keys(tmp_path, monkeypatch):
    """Prepared, the old blocks are the stock keys' and every reader saw them
    there: ``Blue``'s retired 4950 focus is forgotten, no old key is left,
    and the old step order is re-derived without being reported."""
    root = write_installation(tmp_path)
    _write_old_current_json(root, microscope='LS850')
    settings_init.take_stored_replacements()

    prepared, rejected = settings_init.prepare_settings(
        logger, str(root), fall_back_to_template=False
    )

    assert rejected is None
    assert _gains(prepared, _STOCK) == [3.0, 4.0, 5.0]
    assert prepared['Ex405']['focus'] is None
    assert not set(_OLD) & set(prepared)
    assert prepared['step_channel_order'][3:6] == ['Ex488', 'Ex405', 'Ex589']
    assert settings_init.take_stored_replacements() == []


def test_an_old_block_beside_its_stock_keys_block_is_dropped_and_told(tmp_path, monkeypatch):
    """A file holding both ``Blue`` and ``Ex405`` (written across a rollback,
    or by hand) keeps ``Ex405``; ``Blue``'s block is replaced by nothing, the
    release having no such setting, and the load says so."""
    root = write_installation(tmp_path)
    _write_old_current_json(root, microscope='LS850')
    current = json.loads((root / 'data' / 'current.json').read_text())
    current['Ex405'] = {**current['Blue'], 'gain_db': 9.0}
    (root / 'data' / 'current.json').write_text(json.dumps(current))
    settings_init.take_stored_replacements()

    prepared, _rejected = settings_init.prepare_settings(
        logger, str(root), fall_back_to_template=False
    )

    assert prepared['Ex405']['gain_db'] == 9.0
    assert 'Blue' not in prepared
    (notice,) = settings_init.take_stored_replacements()
    assert [(path, used) for path, _saved, used in notice.replacements] == [('Blue', None)]


def _session(root: pathlib.Path, *, microscope: str):
    """The GUI's shape: the file prepared, then handed to ``create``."""
    _write_old_current_json(root, microscope=microscope)
    prepared, _rejected = settings_init.prepare_settings(
        logger, str(root), fall_back_to_template=False
    )
    return ScopeSession.create(prepared, source_path=str(root), simulate=True)


# parametrize-justified: each unit is its own installation and bring-up.
@pytest.mark.parametrize(
    ('microscope', 'block', 'expected'),
    [
        pytest.param(
            'LS850', False, {'Ex405': 3.0, 'Ex488': 4.0, 'Ex589': 5.0, 'Ex550': 10.0}, id='stock'
        ),
        pytest.param(
            'LS850',
            True,
            {'Ex488': 3.0, 'Ex550': 4.0, 'Ex640': 5.0, 'Ex405': 10.0, 'Ex589': 10.0},
            id='non-stock',
        ),
        pytest.param('LS560', False, {'Ex488': 4.0, 'Ex405': 3.0}, id='LS560'),
    ],
)
def test_bring_up_places_old_settings_on_the_units_own_leds(
    tmp_path, monkeypatch, microscope, block, expected
):
    """On a stock unit the renamed blocks stay; on a unit whose LEDs 0 / 1 /
    2 are 488 / 550 / 640, ``Blue``'s settings are its ``Ex488``'s and the
    stock keys it no longer claims are the template's; on an LS560, which
    has LED 1 alone, ``Green``'s are its ``Ex488``'s. The bring-up's own
    checks judge each where it lands: ``Blue`` acquiring stays acquiring on
    a unit with its LED, and stops on the LS560, whose ``Ex405`` it lacks."""
    root = tmp_path / 'installation'
    board_block = non_stock_block(_transmitted_rows()) if block else None
    write_installation(root, board_block=board_block)
    session = _session(root, microscope=microscope)
    try:
        held = {key: session.get_setting(f'{key}.gain_db') for key in expected}
        blue_lands = 'Ex488' if block else 'Ex405'
        acquiring = session.channel_acquire(blue_lands)
        leftovers = set(_OLD) & set(session.settings)
        record = settings_init.OLD_CHANNEL_BLOCKS in session.settings
    finally:
        session.shutdown()

    assert held == expected
    assert acquiring == ('image' if microscope == 'LS850' else None)
    assert (leftovers, record) == (set(), False)
