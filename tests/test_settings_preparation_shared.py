"""Every host prepares settings the same way.

Reading the settings file is one step of preparation, not all of it. The
GUI used to run the rest -- shape check, folds, repairs, default merge --
and a headless session ran only the read, so an L2 caller got a dict that
parsed and was silently missing whatever newer releases had added. These
pin the shared pipeline and the headless caller's use of it.
"""

import copy
import json
import logging
import os
import shutil

import pytest

import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.settings_init as settings_init
from _lumascope.modules.exceptions import InstallationFileError
from _lumascope.modules.channel_record import release_old_names
from _lumascope.modules.zstack_config import ZStackReference
from tests.installation_fixtures import copy_installation_files
from tests.locale_fixtures import a_latin1_locale
from tests.stock_channels import LED1, STOCK_FLUORESCENCE

logger = logging.getLogger('test_settings_preparation')

# The name an older release stored each stock channel under.
_OLD_NAME = {old.stock_key: old.name for old in release_old_names()}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_TEMPLATE = os.path.join(REPO_ROOT, 'data', 'settings.json')


def _data_dir(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    shutil.copy(SHIPPED_TEMPLATE, data / 'settings.json')
    # The factory builds the session's helpers from this root and refuses
    # to configure the scope without them.
    copy_installation_files(data)
    return data


def _legacy_current(data_dir):
    """A current.json written by an older release, in that release's spellings."""
    current = settings_init.load_template(logger, SHIPPED_TEMPLATE)
    # Its fluorescence blocks under the old names, and none for a channel the
    # catalogue added since.
    for old in release_old_names():
        current[old.name] = current.pop(old.stock_key)
    for key in set(common_utils.get_fluorescence_channels()) - set(STOCK_FLUORESCENCE):
        current.pop(key)

    current['image_output_format']['sequenced'] = 'ImageJ Hyperstack'
    current['disable_protocol_accordions'] = True
    current.pop('video', None)
    current['manual_video'] = {'max_fps': 10}
    # absent entirely, the way a file older than the video feature has it
    current['BF'].pop('video_config', None)
    current['BF']['acquire'] = 'stack'
    current[_OLD_NAME[LED1]]['video_config'] = {'duration': 12, 'fps': 30}
    # a plate the catalogue renamed after this file was written
    current['protocol']['labware'] = 'Center Dish'
    # A key the running version ships and this older file predates.
    current.pop('stimulation_enabled', None)

    with open(data_dir / 'current.json', 'w', encoding='utf-8') as f:
        json.dump(current, f)
    return current


def test_legacy_file_is_fully_prepared(tmp_path):
    data_dir = _data_dir(tmp_path)
    _legacy_current(data_dir)

    prepared, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=False
    )

    assert rejected is None
    # the retired spinner label names a reader, not the format it writes
    assert prepared['image_output_format']['sequenced'] == 'OME-TIFF Hyperstack'
    # the section rename, which only the GUI boot used to apply
    assert 'manual_video' not in prepared
    assert prepared['video']['max_fps'] == 10
    # a preference for a toggle nothing reads any more
    assert 'disable_protocol_accordions' not in prepared
    # present-but-unusable values, which the default merge cannot see
    assert prepared['BF']['video_config'] == {'duration': 5}
    assert prepared['BF']['acquire'] is None
    # a layer's own recording rate, which every step it made recorded at
    # below the one global limit, dropped without losing the duration beside
    # it -- the merge cannot do this, nothing is missing
    assert prepared[LED1]['video_config'] == {'duration': 12}
    # the old channel names, each block read as its LED's key
    assert not set(_OLD_NAME.values()) & set(prepared)
    # a retired plate name, folded to its catalogue key so the one store
    # every reader trusts never carries a spelling the catalogue lacks
    assert prepared['protocol']['labware'] == 'Center Plate'
    # and the keys the running version added since the file was written
    assert 'stimulation_enabled' in prepared


@pytest.mark.parametrize(
    ('label', 'reference'),
    [
        ('Current Position at Top', 'top'),
        ('Current Position at Center', 'center'),
        ('Current Position at Bottom', 'bottom'),
    ],
)
def test_a_stored_zstack_label_loads_as_its_reference_with_nothing_replaced(
    tmp_path, label, reference
):
    """An earlier release stored the z-stack reference as the spinner's label;
    the load reads it as the ZStackReference value it names, so the user's
    choice is kept, not replaced by the shipped one and reported."""
    data_dir = _data_dir(tmp_path)
    current = settings_init.load_template(logger, SHIPPED_TEMPLATE)
    current['zstack']['position'] = label
    (data_dir / 'current.json').write_text(json.dumps(current))
    settings_init.take_stored_replacements()

    prepared, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=False
    )

    assert rejected is None
    assert prepared['zstack']['position'] == reference
    assert type(prepared['zstack']['position']) is str
    assert settings_init.take_stored_replacements() == []


def test_a_handed_zstack_label_is_read_as_its_reference(tmp_path):
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path), zstack={'position': 'Current Position at Bottom'}
        ),
        simulate=True,
    )
    try:
        assert session.setting('zstack')['position'] == ZStackReference.BOTTOM
    finally:
        session.shutdown()


@pytest.mark.slow
def test_a_headless_session_gets_the_same_preparation(tmp_path, monkeypatch):
    from _lumascope.modules.scope_session import ScopeSession

    data_dir = _data_dir(tmp_path)
    _legacy_current(data_dir)

    session = ScopeSession.create(
        ScopeSession.load_user_settings(str(tmp_path)), source_path=str(tmp_path), simulate=True
    )

    assert session.settings['image_output_format']['sequenced'] == 'OME-TIFF Hyperstack'
    assert 'manual_video' not in session.settings
    assert 'disable_protocol_accordions' not in session.settings
    assert session.settings[LED1]['video_config'] == {'duration': 12}
    assert 'stimulation_enabled' in session.settings


def test_an_unusable_current_json_surfaces_to_a_headless_caller(tmp_path):
    """There is nobody to ask, and the template is not the user's config."""
    data_dir = _data_dir(tmp_path)
    (data_dir / 'current.json').write_text('{ not json at all')

    with pytest.raises(settings_init.SettingsFileError):
        settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=False)


def test_a_settings_file_is_read_as_utf8_whatever_the_locale(tmp_path):
    """A hand edit saved as UTF-8 reads back as written on a machine whose
    locale encoding is not UTF-8, as on Windows."""
    path = tmp_path / 'current.json'
    path.write_bytes('{"live_folder": "C:/Users/Jos\u00e9/\u00b5m"}'.encode())

    with a_latin1_locale():
        read = settings_init.read_settings_json(str(path))

    assert read['live_folder'] == 'C:/Users/Jos\u00e9/\u00b5m'


def test_a_settings_file_not_in_utf8_is_refused_whatever_the_locale(tmp_path):
    """A hand edit saved as cp1252 is refused as an unreadable file is, even
    where the locale would decode it: never decoded a second way."""
    path = tmp_path / 'current.json'
    path.write_bytes('{"live_folder": "C:/Users/Jos\u00e9"}'.encode('cp1252'))

    with (
        a_latin1_locale(),
        pytest.raises(settings_init.SettingsFileError, match='could not be read'),
    ):
        settings_init.read_settings_json(str(path))


def test_a_template_lacking_a_needed_key_is_the_installations_refusal_naming_it(tmp_path):
    """A short template is the installation's fault, never the user's file set aside."""
    data_dir = _data_dir(tmp_path)
    template = json.loads((data_dir / 'settings.json').read_text())
    del template['frame']
    (data_dir / 'settings.json').write_text(json.dumps(template))
    shutil.copy(SHIPPED_TEMPLATE, data_dir / 'current.json')

    with pytest.raises(InstallationFileError) as refused:
        settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=True)

    assert refused.value.file_path == data_dir / 'settings.json'
    assert "['frame']" in str(refused.value)


def test_the_gui_comes_up_on_the_template_and_reports_what_it_set_aside(tmp_path):
    data_dir = _data_dir(tmp_path)
    (data_dir / 'current.json').write_text('{ not json at all')

    prepared, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=True
    )

    assert prepared['image_output_format']['sequenced'] is not None
    assert rejected is not None
    rejected_path, _reason = rejected
    assert rejected_path.endswith('current.json')
    # the user's only copy of their configuration is left exactly as it was
    assert (data_dir / 'current.json').read_text() == '{ not json at all'


def _shipped_with(**changes) -> str:
    """The shipped template as a current.json, with ``changes`` laid over its top level."""
    with open(SHIPPED_TEMPLATE, encoding='utf-8') as f:
        current = json.load(f)
    current.update(changes)
    return json.dumps(current)


# Each current.json the load cannot use, by what is wrong with it.
_CORRUPT_CURRENT = {
    'missing the keys the app needs': json.dumps({'microscope': 'LS850'}),
    'manual_video not a mapping': _shipped_with(manual_video='x'),
    'a turret slot that is not a number': _shipped_with(turret_objectives={'1.0': None}),
    'image_mode not a string beside a retired key': _shipped_with(
        image_mode=[], use_full_pixel_depth=True
    ),
    'an integer past the reader': '{"jpg_quality": 1' + '0' * 5000 + '}',
    'nesting past the reader': '{"a": ' + '[' * 100_000 + ']' * 100_000 + '}',
}


def test_a_corrupt_current_json_is_a_settings_file_error_on_every_host(tmp_path):
    """A headless caller is refused naming the file; the GUI comes up on the
    template with the file set aside and untouched."""
    wrong = []
    for n, (what, text) in enumerate(_CORRUPT_CURRENT.items()):
        root = tmp_path / str(n)
        root.mkdir()
        current = _data_dir(root) / 'current.json'
        current.write_text(text)
        try:
            settings_init.prepare_settings(logger, str(root), fall_back_to_template=False)
        except settings_init.SettingsFileError as e:
            if e.reason != 'settings_file_unreadable' or str(current) not in str(e):
                wrong.append((what, 'headless', str(e)[:200]))
        except Exception as e:
            wrong.append((what, 'headless', type(e).__name__))
        else:
            wrong.append((what, 'headless', 'loaded'))
        try:
            _prepared, rejected = settings_init.prepare_settings(
                logger, str(root), fall_back_to_template=True
            )
        except Exception as e:
            wrong.append((what, 'gui', type(e).__name__))
        else:
            if rejected is None or rejected[0] != str(current) or current.read_text() != text:
                wrong.append((what, 'gui', rejected))
    assert wrong == []


def test_normalization_reports_whether_it_changed_anything():
    with open(SHIPPED_TEMPLATE, encoding='utf-8') as f:
        shipped = json.load(f)

    assert settings_init.normalize_loaded_settings(copy.deepcopy(shipped)) is False

    legacy = copy.deepcopy(shipped)
    legacy['image_output_format']['sequenced'] = 'ImageJ Hyperstack'
    assert settings_init.normalize_loaded_settings(legacy) is True

    renamed_plate = copy.deepcopy(shipped)
    renamed_plate['protocol']['labware'] = '384 well Corning Spheroid Microplate'
    assert settings_init.normalize_loaded_settings(renamed_plate) is True
    assert renamed_plate['protocol']['labware'] == '384 well microplate'


def _stored_at(settings: dict, path: str, value: object) -> None:
    """Store ``value`` at the dotted ``path``, making each block it names."""
    *parents, leaf = path.split('.')
    for segment in parents:
        settings = settings.setdefault(segment, {})
    settings[leaf] = value


def _concrete(path: str) -> str:
    return path.replace('*', 'BF', 1) if path.startswith('*.') else path


def _holds(settings: dict, path: str) -> bool:
    *parents, leaf = path.split('.')
    for segment in parents:
        settings = settings.get(segment)
        if not isinstance(settings, dict):
            return False
    return leaf in settings


def test_a_retired_setting_stored_alone_is_dropped_and_reported_whatever_it_holds():
    # Null is the hard case: a stored null was once dropped without the
    # repair being reported.
    with open(SHIPPED_TEMPLATE, encoding='utf-8') as f:
        shipped = json.load(f)
    wrong = []
    for path in map(_concrete, settings_init._RETIRED_PATHS):
        stored = copy.deepcopy(shipped)
        _stored_at(stored, path, None)
        reported = settings_init.normalize_loaded_settings(stored)
        if not reported or _holds(stored, path):
            wrong.append((path, reported))
    assert wrong == []


def test_a_dead_rest_setting_handed_in_is_dropped_and_never_saved(tmp_path):
    """A dict a script hands in is prepared as the file is; the session saves what it holds."""
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    settings = complete_settings(live_folder=str(tmp_path))
    shipped = set(settings['rest_api'])
    settings['rest_api']['api_key'] = 's3cr3t'
    saved = tmp_path / 'current.json'
    session = ScopeSession.create(settings, simulate=True, settings_file=str(saved))
    try:
        assert 'api_key' not in session.settings_snapshot()['rest_api']
    finally:
        session.shutdown()

    assert b's3cr3t' not in saved.read_bytes()
    assert set(json.loads(saved.read_text())['rest_api']) == shipped


def test_a_stored_file_carrying_every_retired_setting_comes_up_without_them(tmp_path):
    data_dir = _data_dir(tmp_path)
    stored = json.loads((data_dir / 'settings.json').read_text())
    retired = [_concrete(path) for path in settings_init._RETIRED_PATHS]
    for path in retired:
        _stored_at(stored, path, 'a value an older release wrote')
    (data_dir / 'current.json').write_text(json.dumps(stored))

    prepared, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=False
    )

    assert rejected is None
    assert [path for path in retired if _holds(prepared, path)] == []


def test_a_plate_the_catalogue_never_had_is_left_as_written():
    # Preparation folds spellings; it does not judge. Whether the plate
    # exists is answered where the plate is selected, against the catalogue.
    with open(SHIPPED_TEMPLATE, encoding='utf-8') as f:
        shipped = json.load(f)
    unknown = copy.deepcopy(shipped)
    unknown['protocol']['labware'] = 'Acme 1536 Ultra Plate'

    assert settings_init.normalize_loaded_settings(unknown) is False
    assert unknown['protocol']['labware'] == 'Acme 1536 Ultra Plate'
