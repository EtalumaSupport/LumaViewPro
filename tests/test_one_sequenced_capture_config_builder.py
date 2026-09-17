# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: one builder assembles a sequenced capture config.

There used to be two -- a settings-reading one in config_helpers and a
widget-reading one in config_ui_getters -- and they could not be kept in
sync, because two of the GUI's inputs had no settings home at all. The
settings builder read them as settings['protocol'] keys that nothing
writes and no shipped template carries, so it answered '1x1' and False
for every caller forever while the GUI at the same instant answered
whatever the user had chosen. It also read labware raw, skipping the
fallback and the user-facing warning the GUI lane went through, and
ignored the wellplate loader it declared.

The GUI builder is now an adapter: it supplies the two authoring choices
that live only in running widgets and delegates the rest.
"""

import json
import pathlib
from unittest.mock import MagicMock

import pytest

from modules.config_helpers import get_sequenced_capture_config_from_settings
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader

REPO = pathlib.Path(__file__).resolve().parent.parent


def _settings(**overrides):
    settings = json.loads((REPO / 'data' / 'settings.json').read_text())
    settings['frame'] = {'width': 1900, 'height': 1900}
    settings.update(overrides)
    return settings


def _build(settings, **kwargs):
    return get_sequenced_capture_config_from_settings(
        settings,
        objective_helper=ObjectiveLoader(),
        wellplate_loader=WellPlateLoader(),
        **kwargs,
    )


@pytest.mark.parametrize(
    'tiling,use_zstacking',
    [('1x1', False), ('2x2', True), ('3x3', False)],
)
def test_the_authoring_choices_reach_the_config(tiling, use_zstacking):
    """They are arguments because they have no settings home.

    Read from settings['protocol'] these were always '1x1' and False --
    the keys do not exist in the shipped template and nothing writes
    them, so the headless lane could not express a tiled or z-stacked
    run at all.
    """
    config = _build(_settings(), tiling=tiling, use_zstacking=use_zstacking)

    assert config['tiling'] == tiling
    assert config['use_zstacking'] is use_zstacking


def test_the_keys_the_builder_used_to_read_still_do_not_exist():
    """The premise above, pinned: if someone adds these keys, the
    argument-passing becomes a second source of truth and this test says so."""
    protocol = json.loads((REPO / 'data' / 'settings.json').read_text())['protocol']

    assert 'tiling' not in protocol
    assert 'use_zstacking' not in protocol


def test_unloadable_labware_falls_back_instead_of_reaching_the_config_bare():
    """The settings lane resolves labware the way the GUI lane does.

    It used to read protocol['labware'] raw, so a missing or unknown
    plate reached the protocol as '' -- skipping the shipped-default
    fallback and the warning the user gets on the GUI lane.
    """
    config = _build(_settings(protocol={'labware': 'a plate that does not exist'}))

    assert config['labware_id'] != ''
    assert config['labware_id'] != 'a plate that does not exist'


def test_the_gui_adapter_and_the_builder_agree(monkeypatch):
    """The anti-duplication pin: same inputs, same config.

    Field-for-field, not spot-checked -- drift between the two lanes is
    the defect this consolidation exists to remove, and it hid for as
    long as it did because nothing compared them.
    """
    settings = _settings()

    ctx = MagicMock()
    ctx.settings = settings
    ctx.objective_helper = ObjectiveLoader()
    ctx.wellplate_loader = WellPlateLoader()
    protocol_settings = MagicMock()
    protocol_settings.ids = {
        'tiling_size_spinner': MagicMock(text='2x2'),
        'acquire_zstack_id': MagicMock(active=True),
    }
    ctx.motion_settings.ids = {'protocol_settings_id': protocol_settings}

    import modules.app_context as app_context

    monkeypatch.setattr(app_context, 'ctx', ctx)

    from modules.config_ui_getters import get_sequenced_capture_config_from_ui

    from_ui = get_sequenced_capture_config_from_ui()
    from_settings = _build(settings, tiling='2x2', use_zstacking=True)

    assert from_ui == from_settings
