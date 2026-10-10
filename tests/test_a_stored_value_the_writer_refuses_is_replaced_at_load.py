# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored value no write could store is replaced at load, that key alone, and told once.

A current.json written before a range was held, or edited by hand, loaded
such a value unchanged; the GUI's settings panel then wrote it back through
the writer, which refused it -- a refusal nobody caused, at start-up. The
load holds the writer's own kind and range rules, so what the app runs on is
what a write could have stored.
"""

import json
import logging
import math
import pathlib
import shutil

import pytest

from modules import settings_init

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = REPO / 'data' / 'settings.json'


def _prepare(tmp_path, edit):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    current = json.loads(TEMPLATE.read_text())
    edit(current)
    (data_dir / 'current.json').write_text(json.dumps(current))
    settings, _rejected = settings_init.prepare_settings(
        logging.getLogger(__name__), str(tmp_path), fall_back_to_template=False
    )
    return settings, settings_init.take_stored_replacements()


@pytest.mark.parametrize(
    'section, key, saved',
    [
        ('video', 'max_fps', 500),
        ('image_output_format', 'live', 'OME-TIFF Hyperstack'),
        ('image_output_format', 'sequenced', 'PNG'),
        ('motion', 'acceleration_max_pct', 500),
        ('motion', 'acceleration_max_pct', '50'),
        ('protocol', 'period', 0.001),
        # A leaf with no range is held to its kind too, an infinity not a number.
        ('zstack', 'step_size', math.inf),
        ('scale_bar', 'enabled', 'yes'),
    ],
)
def test_that_key_alone_takes_the_shipped_value_and_is_told(tmp_path, section, key, saved):
    shipped = json.loads(TEMPLATE.read_text())

    def edit(current):
        current[section][key] = saved
        current['video']['max_duration_seconds'] = 120

    settings, told = _prepare(tmp_path, edit)

    assert settings[section][key] == shipped[section][key]
    assert settings['video']['max_duration_seconds'] == 120, 'a value in range was replaced'
    (notice,) = told
    assert notice.replacements == [(f'{section}.{key}', saved, shipped[section][key])]
    assert notice.reason == 'stored_setting_replaced'


def test_a_stored_nan_is_replaced_and_told(tmp_path):
    shipped = json.loads(TEMPLATE.read_text())

    def edit(current):
        current['zstack']['range'] = math.nan

    settings, told = _prepare(tmp_path, edit)

    assert settings['zstack']['range'] == shipped['zstack']['range']
    (notice,) = told
    ((path, saved, replaced_by),) = notice.replacements
    assert (path, replaced_by) == ('zstack.range', shipped['zstack']['range'])
    assert math.isnan(saved)


def test_a_stored_ceiling_no_write_could_store_is_replaced_and_told(tmp_path):
    """A hand-edited auto-gain exposure ceiling of NaN or of the wrong kind
    reached a run's camera arm unchanged; each class is replaced alone and
    told, and a class the file leaves out takes the shipped value."""
    shipped = json.loads(TEMPLATE.read_text())['ag_ae_max_exposure_ms']

    def edit(current):
        current['ag_ae_max_exposure_ms'] = {'fluorescence': math.nan, 'luminescence': True}

    settings, told = _prepare(tmp_path, edit)

    assert settings['ag_ae_max_exposure_ms'] == shipped
    (notice,) = told
    replaced = {path: (saved, by) for path, saved, by in notice.replacements}
    assert set(replaced) == {
        'ag_ae_max_exposure_ms.fluorescence',
        'ag_ae_max_exposure_ms.luminescence',
    }
    assert math.isnan(replaced['ag_ae_max_exposure_ms.fluorescence'][0])
    assert replaced['ag_ae_max_exposure_ms.luminescence'] == (True, shipped['luminescence'])


def test_a_stored_ceiling_in_range_is_kept(tmp_path):
    def edit(current):
        current['ag_ae_max_exposure_ms'] = {'fluorescence': 150.0}

    settings, told = _prepare(tmp_path, edit)

    assert settings['ag_ae_max_exposure_ms']['fluorescence'] == 150.0
    assert told == []


def test_every_value_one_load_replaced_is_in_one_notice(tmp_path):
    """The centre shows one notice of a kind at a time: a notice per value
    showed the first and hid the rest."""

    def edit(current):
        current['video']['max_fps'] = 500
        current['motion']['acceleration_max_pct'] = 500

    _settings, told = _prepare(tmp_path, edit)

    (notice,) = told
    assert sorted(path for path, _, _ in notice.replacements) == [
        'motion.acceleration_max_pct',
        'video.max_fps',
    ]
    assert 'video.max_fps 500' in str(notice)
    assert 'motion.acceleration_max_pct 500' in str(notice)


def test_values_in_range_are_kept_and_nothing_is_told(tmp_path):
    def edit(current):
        current['video']['max_fps'] = 25
        current['motion']['acceleration_max_pct'] = 37
        current['image_output_format']['live'] = 'JPG'

    settings, told = _prepare(tmp_path, edit)

    assert settings['video']['max_fps'] == 25
    assert settings['motion']['acceleration_max_pct'] == 37
    assert settings['image_output_format']['live'] == 'JPG'
    assert told == []
