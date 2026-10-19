# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A fluorescence channel added as data is a whole channel, on every unit.

Adding a channel is one catalogue record and a unit's row: its settings
block, its colour and its place in a composite follow, with no code naming
it. Three simulated units on a release carrying two added channels (445 nm
on LED address 6, 640 nm on 7), brought up the way every host is
(``prepare_settings``, then ``ScopeSession.create``), carry none, two and
five fluorescence channels; each has a block and a colour for every
channel, and a composite takes each of its channels in that channel's
colour. Each channel's drawer is the GUI's, pinned in
``test_the_drawers_show_the_units_records.py``.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest
import tifffile as tf

import modules.settings_init as settings_init
from modules import image_utils, layer_record
from modules.path_utils import read_installation_file
from tests.installation_fixtures import (
    ADDED_CHANNELS,
    copy_installation_files,
    write_release_with_added_channels,
)
from tests.test_composite_run_e2e import open_composite_session, single_run_dir

logger = logging.getLogger('test_a_channel_added_by_data_is_set_coloured_and_composited')

_ADDED = tuple(record['key'] for record in ADDED_CHANNELS)


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """An installation root whose release carries the added channels."""
    data = tmp_path / 'data'
    data.mkdir()
    copy_installation_files(data)
    scopes = write_release_with_added_channels(data)
    # The release catalogue is one process-wide read; this replaces it
    # with the written file's, the one prepare_settings must then read.
    monkeypatch.setattr(
        layer_record,
        '_CATALOGUE_CACHE',
        layer_record.load_layer_catalogue(read_installation_file(scopes), scopes),
    )
    return tmp_path


def _prepared(installation: pathlib.Path, model: str) -> dict:
    settings, rejected = settings_init.prepare_settings(
        logger, str(installation), fall_back_to_template=False
    )
    assert rejected is None
    settings['microscope'] = model
    settings['simulator_tier'] = 'fast'
    settings['live_folder'] = str(installation / 'live')
    settings['stage_offset'] = {'x': 0.0, 'y': 0.0}
    return settings


def _fluorescence(session) -> list[str]:
    return [r.key_name for r in session.scope.layer_identity.layers if r.modality == 'fluorescence']


# parametrize-justified: each unit is its own session bring-up.
@pytest.mark.parametrize(
    ('model', 'expected'),
    [
        ('LS850-F0', []),
        ('LS850-F2', ['Ex445', 'Ex640']),
        ('LS850-F5', ['Blue', 'Green', 'Red', 'Ex445', 'Ex640']),
    ],
)
def test_each_unit_has_a_block_and_a_colour_for_every_channel(installation, model, expected):
    settings = _prepared(installation, model)
    with open_composite_session(settings, source_path=installation) as (session, _):
        assert _fluorescence(session) == expected
        keys = layer_record.release_catalogue()
        assert set(_ADDED) <= set(keys)
        # The block is the template's fluorescence default: the catalogue
        # prepare_settings read is the one this test wrote.
        fluorescence_block = session.scope.settings_template['Blue']
        for key in keys:
            assert key in session.settings, key
        for key in _ADDED:
            assert session.settings[key] == fluorescence_block, key
        colors = session.channel_colors()
        for record in session.scope.layer_identity.layers:
            assert colors[record.key_name] == layer_record.parse_display_color(
                record.display_color
            ), record.key_name
        assert colors['Ex445'] == (0.0, 1.0, 1.0)
        assert colors['Ex640'] == (1.0, 0.0, 1.0)


def _gray(path: pathlib.Path) -> np.ndarray:
    image, bits, _ = image_utils.load_pixels(path, collapse_legacy_false_color=False)
    return image_utils.convert_to_8bit(image_utils.rgb_image_to_gray(image, baked=True), bits)


def _sum(frames: dict[str, np.ndarray], colors: dict) -> np.ndarray:
    """The additive composite, computed here: each frame times its colour, summed, clipped."""
    total = sum(
        frame[..., None].astype(np.float64) * np.asarray(colors[key])
        for key, frame in frames.items()
    )
    return np.clip(np.rint(total), 0, 255)


# parametrize-justified: each unit is its own session and composite run.
@pytest.mark.parametrize('model', ['LS850-F2', 'LS850-F5'])
def test_a_composite_takes_every_channel_in_its_colour(installation, model):
    settings = _prepared(installation, model)
    with open_composite_session(settings, source_path=installation) as (session, runner):
        channels = _fluorescence(session)
        for key in layer_record.release_catalogue():
            session.set_layer_acquire(key, 'image' if key in channels else None)

        runs = installation / 'runs'
        runs.mkdir()
        outcome = runner.run_composite(sequence_name='added', parent_dir=str(runs))
        colors = session.channel_colors()

    assert outcome.status == 'completed', outcome
    run_dir = single_run_dir(runs)
    frames = {}
    for key in channels:
        (path,) = run_dir.glob(f'*{key}*.tiff')
        frames[key] = _gray(path)
    merged = tf.imread(str(outcome.artifact_path)).astype(np.float64)

    assert np.abs(merged - _sum(frames, colors)).max() <= 1
    for key in channels:
        without = _sum({k: f for k, f in frames.items() if k != key}, colors)
        assert np.abs(merged - without).max() > 1, f'the composite does not change without {key}'
