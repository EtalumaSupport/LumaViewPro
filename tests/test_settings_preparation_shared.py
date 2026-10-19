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

import modules.settings_init as settings_init
from modules.exceptions import InstallationFileError
from tests.installation_fixtures import copy_installation_files

logger = logging.getLogger('test_settings_preparation')

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

    current['image_output_format']['sequenced'] = 'ImageJ Hyperstack'
    current['disable_protocol_accordions'] = True
    current.pop('video', None)
    current['manual_video'] = {'max_fps': 10}
    # absent entirely, the way a file older than the video feature has it
    current['BF'].pop('video_config', None)
    current['BF']['acquire'] = 'stack'
    current['Green']['video_config'] = {'duration': 12, 'fps': 0}
    # a plate the catalogue renamed after this file was written
    current['protocol']['labware'] = 'Center Dish'
    # A key the running version ships and this older file predates.
    current.pop('stimulation_enabled', None)

    with open(data_dir / 'current.json', 'w') as f:
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
    assert prepared['BF']['video_config'] == {'duration': 5, 'fps': 30}
    assert prepared['BF']['acquire'] is None
    # a rate the recorder would divide by, repaired without losing the
    # duration beside it -- the merge cannot do this, nothing is missing
    assert prepared['Green']['video_config'] == {'duration': 12, 'fps': 30}
    # a retired plate name, folded to its catalogue key so the one store
    # every reader trusts never carries a spelling the catalogue lacks
    assert prepared['protocol']['labware'] == 'Center Plate'
    # and the keys the running version added since the file was written
    assert 'stimulation_enabled' in prepared


def test_a_headless_session_gets_the_same_preparation(tmp_path, monkeypatch):
    from modules.scope_session import ScopeSession

    data_dir = _data_dir(tmp_path)
    _legacy_current(data_dir)
    # load_user_settings reads the file only when no settings are already loaded
    monkeypatch.setattr(settings_init, 'settings', None)

    session = ScopeSession.create(
        ScopeSession.load_user_settings(str(tmp_path)), source_path=str(tmp_path), simulate=True
    )

    assert session.settings['image_output_format']['sequenced'] == 'OME-TIFF Hyperstack'
    assert 'manual_video' not in session.settings
    assert 'disable_protocol_accordions' not in session.settings
    assert session.settings['Green']['video_config'] == {'duration': 12, 'fps': 30}
    assert 'stimulation_enabled' in session.settings


def test_an_unusable_current_json_surfaces_to_a_headless_caller(tmp_path):
    """There is nobody to ask, and the template is not the user's config."""
    data_dir = _data_dir(tmp_path)
    (data_dir / 'current.json').write_text('{ not json at all')

    with pytest.raises(settings_init.SettingsFileError):
        settings_init.prepare_settings(logger, str(tmp_path), fall_back_to_template=False)


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


def test_normalization_reports_whether_it_changed_anything():
    with open(SHIPPED_TEMPLATE) as f:
        shipped = json.load(f)

    assert settings_init.normalize_loaded_settings(copy.deepcopy(shipped)) is False

    legacy = copy.deepcopy(shipped)
    legacy['image_output_format']['sequenced'] = 'ImageJ Hyperstack'
    assert settings_init.normalize_loaded_settings(legacy) is True

    renamed_plate = copy.deepcopy(shipped)
    renamed_plate['protocol']['labware'] = '384 well Corning Spheroid Microplate'
    assert settings_init.normalize_loaded_settings(renamed_plate) is True
    assert renamed_plate['protocol']['labware'] == '384 well microplate'


def test_a_stored_bf_focus_for_fluorescence_is_dropped():
    # The setting was removed; a file that still carries it loads without it.
    with open(SHIPPED_TEMPLATE) as f:
        stored = json.load(f)
    stored['protocol']['bf_af_for_fluorescence'] = True

    assert settings_init.normalize_loaded_settings(stored) is True
    assert 'bf_af_for_fluorescence' not in stored['protocol']


def test_the_settings_nothing_reads_are_dropped_from_a_stored_file():
    # Eight shipped keys had no reader; a file that still carries them loads
    # without them, and the autofocus and stage blocks they filled go whole.
    with open(SHIPPED_TEMPLATE) as f:
        stored = json.load(f)
    stored['camera_type'] = 'pylon'
    stored['profiling']['stim_profiling'] = False
    stored['autofocus'] = {
        'one_af_per_well': True,
        'reduce_af_range': False,
        'af_range_reduction_pct': -999,
        'suppress_background_during_protocol': False,
    }
    stored['stage'] = {'plate_bottom_z_estimate': None, 'focus_safety_margin_um': 500}

    assert settings_init.normalize_loaded_settings(stored) is True
    assert 'camera_type' not in stored
    assert 'stim_profiling' not in stored['profiling']
    assert 'autofocus' not in stored
    assert 'stage' not in stored


def test_a_stored_layer_file_root_is_dropped_at_load(tmp_path):
    # Every layer once shipped a filename prefix nothing read; a current.json
    # that still carries one comes up without it.
    data_dir = _data_dir(tmp_path)
    stored = json.loads((data_dir / 'settings.json').read_text())
    stored['BF']['file_root'] = 'BF_'
    (data_dir / 'current.json').write_text(json.dumps(stored))

    prepared, rejected = settings_init.prepare_settings(
        logger, str(tmp_path), fall_back_to_template=False
    )

    assert rejected is None
    assert 'file_root' not in prepared['BF']


def test_a_plate_the_catalogue_never_had_is_left_as_written():
    # Preparation folds spellings; it does not judge. Whether the plate
    # exists is answered where the plate is selected, against the catalogue.
    with open(SHIPPED_TEMPLATE) as f:
        shipped = json.load(f)
    unknown = copy.deepcopy(shipped)
    unknown['protocol']['labware'] = 'Acme 1536 Ultra Plate'

    assert settings_init.normalize_loaded_settings(unknown) is False
    assert unknown['protocol']['labware'] == 'Acme 1536 Ultra Plate'
