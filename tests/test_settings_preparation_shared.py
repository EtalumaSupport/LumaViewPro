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

logger = logging.getLogger('test_settings_preparation')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_TEMPLATE = os.path.join(REPO_ROOT, 'data', 'settings.json')


def _data_dir(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    shutil.copy(SHIPPED_TEMPLATE, data / 'settings.json')
    return data


def _legacy_current(data_dir):
    """A current.json written by an older release, in that release's spellings."""
    with open(SHIPPED_TEMPLATE) as f:
        current = json.load(f)

    current['image_output_format']['sequenced'] = 'ImageJ Hyperstack'
    current['disable_protocol_accordions'] = True
    current.pop('video', None)
    current['manual_video'] = {'max_fps': 10}
    # absent entirely, the way a file older than the video feature has it
    current['BF'].pop('video_config', None)
    current['BF']['acquire'] = 'stack'
    current['Green']['video_config'] = {'duration': 12, 'fps': 0}
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
    # and the keys the running version added since the file was written
    assert 'stimulation_enabled' in prepared


def test_a_headless_session_gets_the_same_preparation(tmp_path, monkeypatch):
    from modules.scope_session import ScopeSession

    data_dir = _data_dir(tmp_path)
    _legacy_current(data_dir)
    # create_headless reads the file only when no settings are already loaded
    monkeypatch.setattr(settings_init, 'settings', None)

    session = ScopeSession.create_headless(source_path=str(tmp_path))

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
