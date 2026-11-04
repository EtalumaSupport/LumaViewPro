# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A derived file is encoded from its own pixels, on every host.

A stitch, a z-projection and a merged composite write what their input
files hold: mono tiles give a mono output and false-coloured tiles a
coloured one, whatever the image mode is when the build runs. The build
reads no live mode, so the same folder gives the same files headless (REST,
the SDK, a script) and under the GUI, whose published application context
carries the mode.

The stitch and projection tests run the real builds through
``session.post_processing`` on a run folder written as a run of this
release saves it, at 12 bits: at 8 bits a mono and a false-colour encoding
write the same file, so only the 16-bit container tells them apart.
"""

import ast
import threading
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

import modules.app_context as _app_ctx
import modules.image_utils as image_utils
from modules.composite_generation import CompositeGeneration
from modules.layer_record import GREY
from modules.layer_record import release_colors
from tests.ast_seams import parse_module
from tests.stock_channels import LED0, LED2
from tests.test_a_derived_image_says_how_it_was_made import write_run_folder

RGB_MODE = '12bit_false_color_rgb'

_BUILDS = {
    'stitch': lambda post_processing, folder: post_processing.stitch(folder),
    'zproject': lambda post_processing, folder: post_processing.zproject(folder, method='Max'),
}


def _rgb_mode_context():
    """The GUI's published context with the false-colour RGB mode chosen."""
    # a stand-in by design: the GUI's application context, which the build
    # under test must not read; only its image mode is the GUI's to carry here.
    ctx = mock.MagicMock()
    ctx.settings = {'image_mode': RGB_MODE}
    ctx.settings_lock = threading.Lock()
    return ctx


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    yield session
    session.shutdown()


def _built(session, folder, build) -> dict:
    """{file name: pixels} of every file a build of ``folder`` wrote."""
    result = build(session.post_processing, folder)
    return {path.name: tf.imread(path) for path in result.artifact_paths}


def test_a_stitch_and_a_projection_write_the_same_files_under_the_gui(
    session, tmp_path, monkeypatch
):
    """Mono fluorescence tiles built headless and built with the GUI's context
    published in the RGB mode give the same mono files. Read from the
    context, the mode once widened the GUI's outputs to three channels."""
    for name, build in _BUILDS.items():
        headless = write_run_folder(tmp_path / name / 'headless', LED0, bits=12)
        under_gui = write_run_folder(tmp_path / name / 'gui', LED0, bits=12)
        monkeypatch.setattr(_app_ctx, 'ctx', None)
        expected = _built(session, headless, build)
        monkeypatch.setattr(_app_ctx, 'ctx', _rgb_mode_context())
        got = _built(session, under_gui, build)

        assert sorted(got) == sorted(expected), f'{name}: {sorted(got)} != {sorted(expected)}'
        assert any(f'_{LED0}_' in file for file in expected), f'{name}: no {LED0} output'
        for file, image in expected.items():
            assert image.ndim == 2, f'{name}: {file} from mono tiles is {image.shape}'
            assert got[file].shape == image.shape, (
                f'{name}: {file} is {got[file].shape} under the GUI, {image.shape} headless'
            )
            assert np.array_equal(got[file], image), f'{name}: {file} differs under the GUI'


def test_false_coloured_tiles_build_false_coloured_files(session, tmp_path):
    """Tiles a false-colour capture saved in three channels give three-channel
    fluorescence outputs, with no live mode to read; brightfield stays mono."""
    for name, build in _BUILDS.items():
        folder = write_run_folder(tmp_path / name, LED0, bits=12, fluorescence_encoding='rgb')
        built = _built(session, folder, build)

        fluorescence = {file: image for file, image in built.items() if f'_{LED0}_' in file}
        brightfield = {file: image for file, image in built.items() if '_BF_' in file}
        assert fluorescence and brightfield, f'{name}: built {sorted(built)}'
        for file, image in fluorescence.items():
            assert image.ndim == 3 and image.shape[-1] == 3, f'{name}: {file} is {image.shape}'
        for file, image in brightfield.items():
            assert image.ndim == 2, f'{name}: {file} is {image.shape}'


@pytest.fixture
def recorded_writes(monkeypatch):
    """Capture every ``write_tiff`` call's keywords instead of writing."""
    calls = []

    def _record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(image_utils, 'write_tiff', _record)
    return calls


def _channel_files(tmp_path, colors):
    rows = []
    for color in colors:
        arr = np.full((8, 8), 120, dtype=np.uint8)
        tf.imwrite(str(tmp_path / f'{color}.tiff'), arr, compression='lzw')
        rows.append({'Color': color, 'Filepath': f'{color}.tiff'})
    return pd.DataFrame(rows)


def _merge(tmp_path):
    df = _channel_files(tmp_path, ('BF', LED2))
    result = CompositeGeneration._create_composite_image(
        path=tmp_path,
        df=df,
        brightness_thresholds_percent={LED2: 25},
        output_file_loc=tmp_path / 'composite.tiff',
        channel_colors=release_colors(),
    )
    assert result['status'] is True, f'the merge failed: {result.get("error")}'


class TestTheCompositeMergeIsEightBitRgb:
    def test_the_merge_ignores_the_live_mode(self, tmp_path, recorded_writes, monkeypatch):
        """With the GUI's context in the RGB mode, the merge still writes the
        8-bit encoding of its RGB pixels."""
        monkeypatch.setattr(_app_ctx, 'ctx', _rgb_mode_context())
        _merge(tmp_path)

        assert len(recorded_writes) == 1
        assert recorded_writes[0]['color'] == GREY
        assert recorded_writes[0]['save_encoding'] == '8bit', recorded_writes[0]['save_encoding']

    def test_the_merge_writes_the_same_with_no_context(
        self, tmp_path, recorded_writes, monkeypatch
    ):
        """Headless, the merge writes the same encoding."""
        monkeypatch.setattr(_app_ctx, 'ctx', None)
        _merge(tmp_path)

        assert recorded_writes[0]['save_encoding'] == '8bit'


def test_image_utils_does_not_import_the_context():
    """Structural, at every scope: the writer the engine and every build share
    never reaches the GUI's context."""
    tree = parse_module('modules/image_utils.py')
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Import)
            and any(a.name == 'modules.app_context' for a in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and (
                node.module == 'modules.app_context'
                or (node.module == 'modules' and any(a.name == 'app_context' for a in node.names))
            )
        )
    ]
    assert not offenders, f'modules/image_utils.py imports the application context at {offenders}'
