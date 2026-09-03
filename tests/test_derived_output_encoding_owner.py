"""The derived-output encoding resolver lives outside the run engine.

The resolver that turns the user's live image mode into a save encoding for a
stitch or a z-projection reads the application context to find that mode. It
used to live in ``modules/image_utils.py``, a module the run engine imports,
and the composite merge -- which runs inside the engine on every run kind,
headless included -- called it too. Headless, the context is unset and the
resolver fell back to the array's own encoding; in the GUI it followed the
live mode. The composite's written file was byte-identical either way (the
merge always produces uint8 RGB), so nothing showed, but the engine was
reaching for a process-wide store to decide a value it already had a ruling
for: a merged composite is always 8-bit RGB.

Now the two composite writers resolve their encoding from that ruled constant,
and the context-reading resolver lives in its own module that only the two
GUI-driven post-processors import. The engine never learns the mode through
the context, and a stitch or z-projection still honours it.
"""

import ast
import pathlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

import modules.app_context as _app_ctx
import modules.image_utils as image_utils
from modules.composite_generation import CompositeGeneration
from tests.ast_seams import parse_module

RGB_MODE = '12bit_false_color_rgb'


@pytest.fixture
def recorded_writes(monkeypatch):
    """Capture every ``write_tiff`` call's keywords instead of writing."""
    calls = []

    def _record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(image_utils, 'write_tiff', _record)
    return calls


@pytest.fixture
def rgb_mode_context(monkeypatch):
    """A live context whose user has chosen the false-color RGB mode, the one
    mode under which a derived output's encoding differs from its dtype."""
    import threading

    ctx = mock.MagicMock()
    ctx.settings = {'image_mode': RGB_MODE}
    ctx.settings_lock = threading.Lock()
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    return ctx


@pytest.fixture
def no_context(monkeypatch):
    monkeypatch.setattr(_app_ctx, 'ctx', None)


def _channel_files(tmp_path, colors):
    rows = []
    for color in colors:
        arr = np.full((8, 8), 120, dtype=np.uint8)
        tf.imwrite(str(tmp_path / f'{color}.tiff'), arr, compression='lzw')
        rows.append({'Color': color, 'Filepath': f'{color}.tiff'})
    return pd.DataFrame(rows)


def _merge(tmp_path):
    df = _channel_files(tmp_path, ('BF', 'Red'))
    result = CompositeGeneration._create_composite_image(
        path=tmp_path,
        df=df,
        brightness_thresholds_percent={'Red': 25},
        output_file_loc=tmp_path / 'composite.tiff',
    )
    assert result['status'] is True, f'the merge failed: {result.get("error")}'


class TestTheCompositeMergeFollowsTheRuling:
    def test_the_merge_ignores_the_live_mode(self, tmp_path, recorded_writes, rgb_mode_context):
        """The discriminating case: with a live context in the RGB mode, the
        merge still writes the ruled 8-bit encoding. Resolving from the
        context gave 'rgb' here."""
        _merge(tmp_path)

        assert len(recorded_writes) == 1
        assert recorded_writes[0]['color'] == 'Composite'
        assert recorded_writes[0]['save_encoding'] == '8bit', recorded_writes[0]['save_encoding']

    def test_the_merge_resolves_with_no_context_in_the_process(
        self, tmp_path, recorded_writes, no_context
    ):
        """Preserved on both sides: headless, the merge writes the same
        encoding, now by ruling rather than by fallback."""
        _merge(tmp_path)

        assert recorded_writes[0]['save_encoding'] == '8bit'


class TestTheGuiPostProcessorsHonourTheLiveMode:
    """Preserved on both sides of the move: the two GUI-driven writers still
    follow the user's mode, through the resolver at its new home."""

    def test_a_stitch_widens_under_the_rgb_mode(self, tmp_path, recorded_writes, rgb_mode_context):
        from modules import stitching_core

        tile = tmp_path / 'tile.tiff'
        tf.imwrite(str(tile), np.full((8, 8), 3000, dtype=np.uint16))

        stitching_core._write_output(
            path=tmp_path,
            output_file_loc=pathlib.Path('stitched.tiff'),
            image=np.full((8, 8), 3000, dtype=np.uint16),
            first_tile_path=tile,
            color='Blue',
            center={'x': 0.0, 'y': 0.0},
            significant_bits=16,
            algorithm='test',
        )

        assert recorded_writes[0]['save_encoding'] == 'rgb'

    def test_a_stitch_keeps_its_dtype_with_no_context(self, tmp_path, recorded_writes, no_context):
        from modules import stitching_core

        tile = tmp_path / 'tile.tiff'
        tf.imwrite(str(tile), np.full((8, 8), 3000, dtype=np.uint16))

        stitching_core._write_output(
            path=tmp_path,
            output_file_loc=pathlib.Path('stitched.tiff'),
            image=np.full((8, 8), 3000, dtype=np.uint16),
            first_tile_path=tile,
            color='Blue',
            center={'x': 0.0, 'y': 0.0},
            significant_bits=16,
            algorithm='test',
        )

        assert recorded_writes[0]['save_encoding'] == 'right_aligned'

    def test_a_z_projection_widens_under_the_rgb_mode(
        self, tmp_path, recorded_writes, rgb_mode_context
    ):
        from modules.zprojector import ZProjector

        rows = []
        for z in range(2):
            name = f'A1_Blue_Z{z}.tiff'
            tf.imwrite(str(tmp_path / name), np.full((8, 8), 3000 + z, dtype=np.uint16))
            rows.append({'Color': 'Blue', 'Filepath': name})

        ZProjector(has_turret=False)._zproject(
            path=tmp_path,
            df=pd.DataFrame(rows),
            method='Max',
            output_file_loc=pathlib.Path('projected.tiff'),
        )

        assert recorded_writes[0]['save_encoding'] == 'rgb'


class TestTheEngineNeverReachesTheContextForAnEncoding:
    def test_the_resolver_lives_in_its_own_module(self):
        from modules import derived_output_encoding

        assert callable(derived_output_encoding.resolve_output_save_encoding)

    def test_image_utils_no_longer_imports_the_context(self):
        """Structural, at every scope: the resolver's function-local import
        was the module's only reach for the context."""
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
                    or (
                        node.module == 'modules'
                        and any(a.name == 'app_context' for a in node.names)
                    )
                )
            )
        ]
        assert not offenders, (
            f'modules/image_utils.py imports the application context at {offenders}'
        )

    def test_the_composite_writers_do_not_call_the_context_resolver(self):
        tree = parse_module('modules/composite_generation.py')
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == 'resolve_output_save_encoding'
        ]
        assert not offenders, (
            f'modules/composite_generation.py resolves an encoding from the live mode at '
            f'{offenders}; a merged composite is always 8-bit RGB by ruling'
        )
