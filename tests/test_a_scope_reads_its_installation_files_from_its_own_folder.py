# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope reads scopes.json and the motor defaults from the folder it was started on.

The labware and objective catalogues were already read from the scope's
folder, but scopes.json and the motor defaults were read from the
installation's default folder. So a scope started on another folder built
from two folders: a folder whose scopes.json had no models, or no motor
defaults at all, still came up. A simulated scope first read scopes.json
after its worker lanes had started, so a bad file there also left both
lanes running, and it raised ConfigError, which the GUI reads as bad user
settings.

Now the scope reads all four files once, from its folder, before anything
starts, and a bad one is an InstallationFileError naming the file.
"""

import json
import pathlib
import shutil
import threading

import pytest

import modules.lumascope_api._lumascope as lumascope_module
from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from modules import layer_record
from modules.exceptions import ConfigError, InstallationFileError
from modules.lumascope_api import Lumascope
from tests.scope_fakes import build_scope

REPO_DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'


def _folder(tmp_path, mutate=None) -> pathlib.Path:
    shutil.copytree(REPO_DATA, tmp_path / 'data')
    if mutate is not None:
        mutate(tmp_path / 'data')
    return tmp_path


def _scopes(data: pathlib.Path) -> dict:
    return json.loads((data / 'scopes.json').read_text(encoding='utf-8'))


def _write_scopes(data: pathlib.Path, contents: dict) -> None:
    (data / 'scopes.json').write_text(json.dumps(contents), encoding='utf-8')


def _only_ls620(data):
    contents = _scopes(data)
    contents['Models'] = {'LS620': contents['Models']['LS620']}
    _write_scopes(data, contents)


def _no_models(data):
    contents = _scopes(data)
    del contents['Models']
    _write_scopes(data, contents)


def _no_motor_defaults(data):
    (data / 'motorconfig_defaults.json').unlink()


def _stub_real_boards(monkeypatch):
    # No real port is opened in the suite: both boards are answered null.
    monkeypatch.setattr(
        lumascope_module.motor_registry, 'create', lambda name='auto', **kw: NullMotionBoard()
    )
    monkeypatch.setattr(
        lumascope_module.led_registry, 'create', lambda name='auto', **kw: NullLEDBoard()
    )


def test_a_simulated_scope_takes_its_models_from_its_folder(tmp_path):
    root = _folder(tmp_path, _only_ls620)

    scope = build_scope(
        simulate=True, sim_model='LS620', source_path=str(root), warn_pre_release=False
    )

    assert sorted(scope.scope_models) == ['LS620']


def test_a_real_scope_takes_its_models_from_its_folder(tmp_path, monkeypatch):
    _stub_real_boards(monkeypatch)
    root = _folder(tmp_path, _only_ls620)

    scope = build_scope(
        simulate=False,
        camera_type='sim',
        configured_model='LS620',
        source_path=str(root),
        warn_pre_release=False,
        register_atexit=False,
    )

    assert sorted(scope.scope_models) == ['LS620']


def test_the_models_cannot_be_changed_through_the_scope(tmp_path):
    scope = build_scope(simulate=True, sim_model='LS850T', warn_pre_release=False)

    with pytest.raises(TypeError):
        scope.scope_models['LS999'] = {}


def test_a_model_the_folder_does_not_list_is_refused_before_a_lane_starts(tmp_path):
    root = _folder(tmp_path, _only_ls620)
    before = set(threading.enumerate())

    # The setting names a model the folder lacks: the settings' fault, not the file's.
    with pytest.raises(ConfigError, match="lists no model 'LS850T'"):
        Lumascope(simulate=True, sim_model='LS850T', source_path=str(root), warn_pre_release=False)

    assert set(threading.enumerate()) <= before


@pytest.mark.parametrize(
    ('mutate', 'name', 'says'),
    [
        (_no_models, 'scopes.json', 'no usable Models section'),
        (_no_motor_defaults, 'motorconfig_defaults.json', 'is missing'),
    ],
    ids=['scopes-json-without-models', 'motor-defaults-missing'],
)
class TestABadFileInTheScopesFolder:
    def test_stops_a_simulated_scope_before_a_lane_starts(self, tmp_path, mutate, name, says):
        root = _folder(tmp_path, mutate)
        before = set(threading.enumerate())

        with pytest.raises(InstallationFileError, match=says) as refused:
            Lumascope(
                simulate=True, sim_model='LS850T', source_path=str(root), warn_pre_release=False
            )

        assert refused.value.file_path == root / 'data' / name
        # Compared by identity: two lanes of a leaked scope carry the same
        # names as the lanes of any other scope alive in this process.
        assert set(threading.enumerate()) <= before

    def test_stops_a_real_scope(self, tmp_path, monkeypatch, mutate, name, says):
        _stub_real_boards(monkeypatch)
        root = _folder(tmp_path, mutate)

        with pytest.raises(InstallationFileError, match=says) as refused:
            Lumascope(
                simulate=False,
                camera_type='sim',
                configured_model='LS850T',
                source_path=str(root),
                warn_pre_release=False,
                register_atexit=False,
            )

        assert refused.value.file_path == root / 'data' / name

    def test_stops_the_diagnostic_scope(self, tmp_path, mutate, name, says):
        root = _folder(tmp_path, mutate)
        before = set(threading.enumerate())

        with pytest.raises(InstallationFileError, match=says) as refused:
            Lumascope.create_diagnostic(source_path=str(root))

        assert refused.value.file_path == root / 'data' / name
        assert set(threading.enumerate()) <= before


def test_the_identity_resolves_against_the_release_vocabulary(tmp_path):
    # The folder's own layer order is reversed; the ids still come from the
    # release's, the one vocabulary every layer list in the process uses.
    def reversed_order(data):
        contents = _scopes(data)
        contents['LayerOrder'] = list(reversed(contents['LayerOrder']))
        _write_scopes(data, contents)

    root = _folder(tmp_path, reversed_order)
    scope = build_scope(
        simulate=True, sim_model='LS850T', source_path=str(root), warn_pre_release=False
    )

    vocabulary = layer_record.release_catalogue()
    assert scope.layer_identity.layers
    for record in scope.layer_identity.layers:
        assert record.id == vocabulary.index(record.key_name)


@pytest.mark.parametrize(
    'contents',
    [None, '{not json', json.dumps({'Models': {}}), json.dumps({'Models': {}, 'LayerOrder': []})],
    ids=['missing', 'not-json', 'no-layer-order', 'empty-layer-order'],
)
def test_a_bad_installation_scopes_json_stops_the_settings_check(tmp_path, monkeypatch, contents):
    path = tmp_path / 'scopes.json'
    if contents is not None:
        path.write_text(contents, encoding='utf-8')
    monkeypatch.setattr(layer_record, 'resolve_data_file', lambda *parts, **kw: path)
    monkeypatch.setattr(layer_record, '_CATALOGUE_CACHE', None)

    with pytest.raises(InstallationFileError) as refused:
        layer_record.release_catalogue()

    assert refused.value.file_path == path
    # Nothing is cached, so the settings check never runs on an empty vocabulary.
    assert layer_record._CATALOGUE_CACHE is None
