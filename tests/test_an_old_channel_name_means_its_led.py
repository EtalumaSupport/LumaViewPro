# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An old channel name means the LED it was made with.

Earlier releases named the fluorescence channels for their colour --
``Blue``, ``Green``, ``Red`` -- and every unit that stored one drove it on
LED 0, 1 or 2. The release states that history once, its
``OldChannelNames`` table: each old name's LED address and its stock key,
the key of the stock filterset's LED there. Where the unit is known, an
old name is the unit's own channel on that address (``ChannelIdentity.
unit_key``); where none is, a stored old name reads as its stock key for
its colour, modality and settings and keeps its own name
(``stored_key``), so an old run folder builds in its old names.

Each test runs against the shipped release, whose keys are named for
their excitation and whose catalogue carries the non-stock filterset's
channels.
"""

from __future__ import annotations

import datetime
import ast
import json
import pathlib
import shutil

import numpy as np
import pytest
import tifffile

from modules import image_utils
from modules.common_utils import recover_step_label
from modules.exceptions import InstallationFileError, PostProcessingOperation
from modules.layer_record import (
    GREY,
    load_layer_catalogue,
    load_old_channel_names,
    read_scope_models,
    resolve_layer_identity,
    stored_key,
)
from modules.post_processing_api import Stitch, ZProject
from modules.path_utils import read_installation_file
from tests.case_walk import failing_cases
from tests.installation_fixtures import non_stock_block, write_installation
from tests.test_a_derived_image_says_how_it_was_made import write_run_folder


@pytest.fixture
def run_folder(tmp_path):
    """A BF and ``Blue`` run folder, as a release before the keys changed saved it."""
    return write_run_folder(tmp_path, 'Blue')


@pytest.fixture
def installation(tmp_path):
    """An installation folder of the shipped files."""
    return write_installation(tmp_path / 'installation')


def _release(installation: pathlib.Path) -> dict:
    return read_installation_file(installation / 'data' / 'scopes.json')


def _identity(installation, model, board_block=None):
    contents = _release(installation)
    catalogue = load_layer_catalogue(contents, 'scopes.json')
    old_names = load_old_channel_names(contents, 'scopes.json', catalogue)
    return resolve_layer_identity(
        board_block=board_block,
        board_config_read_ok=True,
        motor_model=model,
        configured_model=None,
        models=read_scope_models(contents, 'scopes.json', catalogue, old_names),
        catalogue=catalogue,
        report=lambda notice: None,
    )


def test_a_stored_old_name_reads_as_its_stock_key(installation):
    """With no unit, ``Blue`` reads as ``Ex405``; a release key and a name
    nothing knows read as themselves."""
    assert [stored_key(name) for name in ('Blue', 'Green', 'Red')] == ['Ex405', 'Ex488', 'Ex589']
    assert [stored_key(name) for name in ('Ex405', 'BF', 'Bleu')] == ['Ex405', 'BF', 'Bleu']


def test_a_table_that_cannot_be_true_is_refused(installation):
    """The table is refused, naming the file, when it is missing, names an
    old name that is a key of this release, a stock key the release lacks or
    that is no fluorescence key, or one LED address twice."""
    contents = _release(installation)
    catalogue = load_layer_catalogue(contents, 'scopes.json')
    ex405 = {'led_address': 0, 'stock_key': 'Ex405'}
    cases = [
        (None, 'has no usable OldChannelNames'),
        ({'Ex488': ex405}, 'also a key of this release'),
        ({'Blue': {'led_address': 0, 'stock_key': 'Ex999'}}, 'not a fluorescence key'),
        ({'Blue': {'led_address': 3, 'stock_key': 'BF'}}, 'not a fluorescence key'),
        ({'Blue': {'led_address': -1, 'stock_key': 'Ex405'}}, 'not an LED address'),
        ({'Blue': {'stock_key': 'Ex405'}}, 'an led_address and a stock_key'),
        ({'Blue': ex405, 'Violet': {'led_address': 0, 'stock_key': 'Ex488'}}, 'address 0'),
    ]

    def refused(table, words):
        data = {k: v for k, v in contents.items() if k != 'OldChannelNames'}
        if table is not None:
            data['OldChannelNames'] = table
        with pytest.raises(InstallationFileError) as caught:
            load_old_channel_names(data, 'scopes.json', catalogue)
        assert words in str(caught.value) and 'scopes.json' in str(caught.value)

    assert failing_cases(cases, refused) == {}


def test_a_model_row_off_its_old_names_address_is_refused(installation):
    """A model row keyed by an old name's stock key on another LED address
    would make that unit's old data light another LED: the model load
    refuses it."""
    contents = _release(installation)
    contents['Models']['LS850']['Layers'] = [
        {**row, 'led_channel': {0: 1, 1: 0}.get(row['led_channel'], row['led_channel'])}
        for row in contents['Models']['LS850']['Layers']
    ]
    catalogue = load_layer_catalogue(contents, 'scopes.json')
    old_names = load_old_channel_names(contents, 'scopes.json', catalogue)

    with pytest.raises(InstallationFileError, match=r"'LS850'.*Ex405.*LED address 1.*Blue"):
        read_scope_models(contents, 'scopes.json', catalogue, old_names)


def test_each_unit_names_an_old_name_for_its_own_led(installation):
    """A stock unit's ``Blue`` is ``Ex405``; a unit whose LED 0 is 488 has it
    as ``Ex488``; a unit with no LED 0 has it as the stock ``Ex405``, a key
    it lacks. A release key, and a name nothing knows, stay as they are."""
    transmitted = [
        row
        for row in _release(installation)['Models']['LS850']['Layers']
        if row['key_name'] in ('BF', 'PC', 'DF')
    ]
    units = {
        'stock': _identity(installation, 'LS850'),
        'non-stock': _identity(installation, 'LS850', non_stock_block(transmitted)),
        'LS560': _identity(installation, 'LS560'),
    }
    names = ('Blue', 'Green', 'Red', 'Ex589', 'BF', 'Bleu')

    assert {unit: [i.unit_key(n) for n in names] for unit, i in units.items()} == {
        'stock': ['Ex405', 'Ex488', 'Ex589', 'Ex589', 'BF', 'Bleu'],
        'non-stock': ['Ex488', 'Ex550', 'Ex640', 'Ex589', 'BF', 'Bleu'],
        'LS560': ['Ex405', 'Ex488', 'Ex589', 'Ex589', 'BF', 'Bleu'],
    }
    assert units['LS560'].find('Ex405') is None


def test_an_old_step_name_is_a_machine_name_before_and_after_its_rename(installation):
    """A pre-Label step named ``A1_Blue`` -- or ``A1_BF_Blue``, the stale
    token of an old channel change -- is machine-named whether its Color is
    still ``Blue`` or already the unit's ``Ex405``, so it re-renders rather
    than being kept as a user's label."""
    cases = [
        ('A1_Blue', 'Blue'),
        ('A1_Blue', 'Ex405'),
        ('A1_BF_Blue', 'Blue'),
        ('A1_BF_Blue', 'Ex405'),
    ]
    recovered = {
        case: recover_step_label(
            {'Name': case[0], 'Well': 'A1', 'Color': case[1], 'Tile': '', 'Z-Slice': ''}
        )
        for case in cases
    }

    assert recovered == dict.fromkeys(cases, ('', True))


@pytest.fixture
def old_session(run_folder, installation, tmp_path):
    """A simulated session on the excitation-key release, beside ``run_folder``:
    a BF and ``Blue`` run, written before the keys changed."""
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')),
        source_path=str(installation),
        simulate=True,
    )
    yield session
    session.shutdown()


def _texts(folder: pathlib.Path) -> dict[str, str]:
    return {name: (folder / name).read_text() for name in ('run.tsv', 'protocol_record.tsv')}


def test_an_old_run_folder_builds_in_its_old_names(old_session, run_folder):
    """Projected, stitched and composited on the new release, an old run's
    ``Blue`` images build: the outputs are named ``Blue``, state no
    excitation the inputs did not, and nothing in the folder is renamed."""
    before = _texts(run_folder)

    result = old_session.post_processing.build(
        run_folder, zproject=ZProject(), stitch=Stitch(), composite=True
    )

    made = {step.operation: step.result.artifact_paths for step in result.built}
    stitched = sorted(pathlib.Path(p).name for p in made[PostProcessingOperation.STITCH])
    assert stitched == [
        'custom0000_BF_0000_zproj_max_stitched.tiff',
        'custom0001_Blue_0000_zproj_max_stitched.tiff',
    ]
    assert len(made[PostProcessingOperation.COMPOSITE]) == 1
    blue = next(pathlib.Path(p) for p in made[PostProcessingOperation.STITCH] if '_Blue_' in str(p))
    with tifffile.TiffFile(blue) as tif:
        described = json.loads(tif.pages[0].description)
    assert described['Channel']['Name'] == ['Blue']
    assert 'ExcitationWavelength' not in described['Channel']
    assert _texts(run_folder) == before


def test_an_old_names_image_is_written_as_fluorescence(installation, tmp_path):
    """A 16-bit image of an old run's ``Blue``, written on the new release,
    states the fluorescence modality its stock key has."""
    path = tmp_path / 'A1_Blue_0000.tiff'
    image_utils.write_tiff(
        data=np.full((8, 8), 1000, dtype=np.uint16),
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'Blue',
            'objective': {'id': '20x Oly'},
            'exposure_time_ms': 2.0,
            'gain_db': 1.0,
            'illumination_ma': 5.0,
            'z_pos_um': 5000.0,
            'plate_pos_mm': {'x': 12.0, 'y': 12.0},
            'datetime': datetime.datetime(2026, 10, 25, 12, 0, 0),
            'significant_bits': 12,
        },
        ome=False,
        color=GREY,
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )

    with tifffile.TiffFile(path) as tif:
        channel = ast.literal_eval(tif.imagej_metadata['channel'])
    assert (channel['Name'], channel['Modality']) == (['Blue'], ['MIF'])


def test_an_old_folder_per_channel_is_read_from_its_old_names_folders(old_session, run_folder):
    """A run that saved a folder per channel keeps its ``Blue/`` images
    there; on the new release they are still the run's own images."""
    record = run_folder / 'protocol_record.tsv'
    lines = record.read_text().splitlines(keepends=True)
    moved = []
    for i, line in enumerate(lines):
        name = line.split('\t', 1)[0]
        if name.endswith('.tiff') and ('_BF_' in name or '_Blue_' in name):
            channel = 'BF' if '_BF_' in name else 'Blue'
            (run_folder / channel).mkdir(exist_ok=True)
            shutil.move(run_folder / name, run_folder / channel / name)
            lines[i] = f'{channel}/{line}'
            moved.append(name)
    record.write_text(''.join(lines))

    result = old_session.post_processing.zproject(run_folder, method='Max')

    assert len(moved) == 24
    assert result.new_count == 8


def _bake_blue(run_folder: pathlib.Path) -> dict[str, np.ndarray]:
    """Re-save the run's ``Blue`` images as an older release saved them in
    false colour, here a user's magenta: its red and blue planes each the
    mono. Each file's mono pixels, by name."""
    mono = {}
    for path in sorted(run_folder.glob('custom0001_Blue_*.tiff')):
        mono[path.name] = image_utils.load_pixels(path, collapse_legacy_false_color=False)[0]
        zeros = np.zeros_like(mono[path.name])
        tifffile.imwrite(
            path, np.stack([mono[path.name], zeros, mono[path.name]], axis=-1), photometric='rgb'
        )
    return mono


def test_an_old_runs_false_colour_images_stack_as_baked(old_session, run_folder):
    """An old run's ``Blue`` images saved in a two-plane false colour are
    stacked as a layer's baked plane, each whole, as ``Ex405``'s would be --
    not read by luminance, as a composite's colour image is."""
    mono = _bake_blue(run_folder)

    old_session.post_processing.run_hyperstack(
        run_folder,
        save_encoding='right_aligned',
        channel_colors=old_session.channel_colors(),
        written=lambda: None,
    ).result(timeout=60)

    with tifffile.TiffFile(run_folder / 'Hyperstack' / 'custom0000_TA1_hyperstack.ome.tiff') as tif:
        assert tif.series[0].axes == 'ZCYX'
        stack = tif.series[0].asarray()
    for z_slice in range(3):
        assert np.array_equal(stack[z_slice, 1], mono[f'custom0001_Blue_TA1_Z{z_slice}_0000.tiff'])


def test_an_old_names_image_is_enhanced_in_its_stock_keys_colour(old_session, run_folder):
    """Quick Enhance of an old run's ``Blue`` image shows it in ``Ex405``'s
    blue, not the grey of a channel it does not know."""
    source = sorted(run_folder.glob('custom0001_Blue_*.tiff'))[0]

    result = old_session.post_processing.enhance(source)

    (made,) = result.created
    shown = image_utils.load_pixels(made.output_path, collapse_legacy_false_color=False)[0]
    assert shown.ndim == 3
    assert (shown[..., 0].max(), shown[..., 1].max()) == (0, 0)
    assert shown[..., 2].max() > 0
