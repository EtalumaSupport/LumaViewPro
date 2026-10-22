# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every catalogue key has a settings block, from the template's read.

The shipped template names a block only for a key with values of its own
and holds one shared block per modality in ``channel_defaults``;
``settings_init.load_template``, the template's one read, gives every other
catalogue key a copy of its modality's block and consumes the defaults. A
key added to the catalogue therefore has settings with no edit to the
template, and nothing a caller reads holds ``channel_defaults``. The order
a new protocol's steps take is each record's ``step_order``, said once, in
the catalogue.
"""

import dataclasses
import json
import logging
import pathlib
import shutil

import pytest

import modules.settings_init as settings_init
from modules import layer_record
from modules.exceptions import InstallationFileError
from modules.scope_session import ScopeSession
from tests.installation_fixtures import SHIPPED_DATA, copy_installation_files
from tests.settings_fixtures import complete_settings

logger = logging.getLogger('test_the_template_gives_every_catalogue_key_a_block')

SHIPPED_TEMPLATE = SHIPPED_DATA / 'settings.json'


def _shipped_fluorescence_block() -> dict:
    return json.loads(SHIPPED_TEMPLATE.read_text())['channel_defaults']['fluorescence']


def _installation(tmp_path: pathlib.Path) -> pathlib.Path:
    data = tmp_path / 'data'
    data.mkdir()
    copy_installation_files(data)
    return data


def _write_template(data: pathlib.Path, edit) -> None:
    template = json.loads(SHIPPED_TEMPLATE.read_text())
    edit(template)
    (data / 'settings.json').write_text(json.dumps(template))


def test_every_preparation_path_gives_every_key_a_block_and_holds_no_defaults(tmp_path):
    # A fresh install, a current.json that will not parse, and a late
    # rejection each come up on the template, and each holds a block for
    # every catalogue key and no channel_defaults.
    data = _installation(tmp_path)
    keys = layer_record.release_catalogue()
    fluorescence = _shipped_fluorescence_block()

    fresh, _ = settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=False)

    (data / 'current.json').write_text('{ not json')
    corrupt, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=True
    )
    assert rejected is not None

    shutil.copy(SHIPPED_TEMPLATE, data / 'current.json')
    settings_init.load_lvp_settings(logger, str(tmp_path))
    try:
        settings_init.fall_back_to_template(logger, str(tmp_path), 'a late rejection')
        late = settings_init.settings
    finally:
        settings_init.settings = None
        settings_init.rejected_current_json = None

    for name, prepared in (('fresh', fresh), ('corrupt', corrupt), ('late', late)):
        assert 'channel_defaults' not in prepared, name
        assert [k for k in keys if k not in prepared] == [], name
        for key in ('Blue', 'Green', 'Red'):
            assert prepared[key] == fluorescence, (name, key)


# parametrize-justified: each model is its own session bring-up.
@pytest.mark.parametrize('model', ['LS560', 'LS850-0', 'Lumi'])
def test_a_session_on_each_model_holds_every_block_and_publishes_no_defaults(tmp_path, model):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope=model),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        keys = layer_record.release_catalogue()
        assert [k for k in keys if k not in session.settings] == []
        assert 'channel_defaults' not in session.settings
        assert 'channel_defaults' not in session.scope.settings_template
        assert [k for k in keys if k not in session.scope.settings_template] == []
    finally:
        session.shutdown()
        session.scope.disconnect()


def test_a_key_added_to_the_catalogue_gets_its_modalitys_block(tmp_path, monkeypatch):
    shipped = layer_record.release_layers()
    blue = next(layer for layer in shipped if layer.key == 'Blue')
    nir = dataclasses.replace(blue, key='NIR', display_color='#ff00ff', step_order=99)
    monkeypatch.setattr(layer_record, '_CATALOGUE_CACHE', (*shipped, nir))

    template = settings_init.load_template(logger, str(SHIPPED_TEMPLATE))

    assert template['NIR'] == _shipped_fluorescence_block()
    assert template['NIR'] is not template['Blue']


def test_a_template_without_channel_defaults_is_refused_naming_it(tmp_path):
    data = _installation(tmp_path)
    _write_template(data, lambda t: t.pop('channel_defaults'))

    with pytest.raises(InstallationFileError) as refused:
        settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=True)

    assert refused.value.file_path == data / 'settings.json'
    assert 'channel_defaults' in str(refused.value)


def test_a_key_with_no_block_and_no_default_for_its_modality_is_refused_naming_it(tmp_path):
    data = _installation(tmp_path)
    _write_template(data, lambda t: t['channel_defaults'].pop('fluorescence'))

    with pytest.raises(InstallationFileError) as refused:
        settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=True)

    assert refused.value.file_path == data / 'settings.json'
    assert 'Blue (fluorescence)' in str(refused.value)


def test_a_fresh_install_adds_a_multi_channel_step_green_before_blue(tmp_path):
    # The shipped step order is the catalogue's step_order, which puts Green
    # before Blue for the composite's focus, not the display order.
    settings = complete_settings(live_folder=str(tmp_path), microscope='LS620')
    for key in layer_record.release_catalogue():
        settings[key]['acquire'] = None
    session = ScopeSession.create(settings, simulate=True, warn_pre_release=False)
    try:
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )
        for key in ('Blue', 'BF', 'Green'):
            session.set_layer_acquire(key, 'image')

        session.add_step(protocol)

        assert [protocol.step_row(step_index=i)['Color'] for i in range(3)] == [
            'BF',
            'Green',
            'Blue',
        ]
    finally:
        session.shutdown()
        session.scope.disconnect()


def test_a_key_added_to_the_catalogue_takes_its_step_order(monkeypatch):
    shipped = layer_record.release_layers()
    blue = next(layer for layer in shipped if layer.key == 'Blue')
    nir = dataclasses.replace(blue, key='NIR', display_color='#ff00ff', step_order=-1)
    monkeypatch.setattr(layer_record, '_CATALOGUE_CACHE', (*shipped, nir))

    template = settings_init.load_template(logger, str(SHIPPED_TEMPLATE))

    assert template['step_channel_order'] == [
        'NIR',
        'BF',
        'PC',
        'DF',
        'Green',
        'Blue',
        'Red',
        'Lumi',
    ]


def test_a_template_that_states_a_step_order_is_refused_naming_it(tmp_path):
    data = _installation(tmp_path)
    _write_template(data, lambda t: t.update(step_channel_order=['BF', 'Blue']))

    with pytest.raises(InstallationFileError) as refused:
        settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=True)

    assert refused.value.file_path == data / 'settings.json'
    assert 'step_channel_order' in str(refused.value)


def test_a_stored_red_illumination_is_kept(tmp_path):
    # The shared default is 250 mA where Red once shipped 350; nothing
    # migrates a stored value, so a user's 350 stays theirs.
    data = _installation(tmp_path)
    stored = settings_init.load_template(logger, str(SHIPPED_TEMPLATE))
    stored['Red']['illumination_ma'] = 350.0
    (data / 'current.json').write_text(json.dumps(stored))

    prepared, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=False
    )

    assert rejected is None
    assert prepared['Red']['illumination_ma'] == 350.0
    assert prepared['Blue']['illumination_ma'] == 250.0
