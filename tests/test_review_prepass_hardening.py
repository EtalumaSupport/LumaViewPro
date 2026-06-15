# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression locks for the code-review pre-pass hardening fixes.

Each test pins one fix surfaced by the pre-merge review of the
mono-native + overlap bundle. Behavioral where the module imports under
the test harness (image_utils, stitch_algorithms); source/AST locks
where the carrier is Kivy-bound and cannot be instantiated (scope_display,
video_writer add_frame, image_save save_image).
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import tifffile as tf

from modules import image_utils


REPO = pathlib.Path(__file__).resolve().parent.parent


def _method_src(rel_path: str, class_name: str | None, func_name: str) -> str:
    source = (REPO / rel_path).read_text()
    tree = ast.parse(source)
    nodes = ast.walk(tree)
    if class_name is not None:
        cls = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        nodes = ast.walk(cls)
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == func_name)
    return ast.unparse(fn)


def _structured_metadata(missing: str | None = None) -> dict:
    """Build a structured-TIFF metadata dict matching write_tiff's shape.

    If ``missing`` names a Plane key, omit it to simulate an incomplete
    structured TIFF (older file / third-party producer).
    """
    plane = {
        'PositionX': 1.0,
        'PositionY': 2.0,
        'PositionZ': 3.0,
        'ExposureTime': 10.0,
        'Gain': 1.0,
        'Illumination': 50.0,
    }
    if missing is not None:
        plane.pop(missing, None)
    return {
        'Plane': plane,
        'PhysicalSizeX': 0.5,
        'Channel': {'Name': ['Blue']},
    }


class TestMetadataReadRobustness:
    """#8: read_postproc_input_metadata returns None on an incomplete
    structured TIFF instead of raising KeyError mid-postproc."""

    def test_complete_structured_tiff_returns_metadata(self, tmp_path):
        p = tmp_path / 'complete.tiff'
        tf.imwrite(str(p), np.zeros((4, 4), dtype=np.uint16), metadata=_structured_metadata())
        out = image_utils.read_postproc_input_metadata(p)
        assert out is not None
        assert out['exposure_time_ms'] == 10.0
        assert out['channel'] == 'Blue'

    def test_missing_plane_key_returns_none_not_keyerror(self, tmp_path):
        p = tmp_path / 'partial.tiff'
        tf.imwrite(
            str(p),
            np.zeros((4, 4), dtype=np.uint16),
            metadata=_structured_metadata(missing='ExposureTime'),
        )
        # Must not raise; falls back to None so the postproc job uses defaults.
        assert image_utils.read_postproc_input_metadata(p) is None


class TestStitchMixedChannelGuard:
    """#9: stitch_registered_tiles raises a clear error on a mix of mono
    and color tiles rather than a cryptic broadcast ValueError."""

    def test_source_has_ndim_consistency_guard(self):
        src = _method_src('modules/stitch_algorithms.py', None, 'stitch_registered_tiles')
        assert '.ndim != sample.ndim' in src, (
            'stitch_registered_tiles must reject a mix of mono/color tiles up front.'
        )
        assert 'raise ValueError' in src


class TestVideoTimestampAfterFalseColor:
    """#2: the timestamp overlay is drawn AFTER false-color conversion so
    the text stays neutral white instead of being tinted."""

    def test_add_timestamp_runs_after_mono_to_rgb_falsecolor(self):
        src = _method_src('modules/video_writer.py', 'VideoWriter', 'add_frame')
        fc_idx = src.find('mono_to_rgb_falsecolor')
        ts_idx = src.find('add_timestamp')
        assert fc_idx != -1 and ts_idx != -1
        assert fc_idx < ts_idx, (
            'add_timestamp must run after mono_to_rgb_falsecolor so the '
            'overlay is not false-colored.'
        )


class TestEngUiNoneGuard:
    """#7: set_engineering_ui returns early when the layer is no longer
    resolvable (accordion collapsed before the scheduled main-thread call)."""

    def test_source_guards_open_layer_none_before_ids(self):
        src = _method_src('ui/scope_display.py', 'ScopeDisplay', 'set_engineering_ui')
        guard_idx = src.find('open_layer_obj is None')
        ids_idx = src.find('.ids[')
        assert guard_idx != -1, 'set_engineering_ui must guard open_layer_obj is None.'
        assert ids_idx != -1 and guard_idx < ids_idx, (
            'the None-guard must precede any open_layer_obj.ids access.'
        )


class TestJpgSkipsMetadata:
    """#4: JPG export resolves only the save path and skips
    generate_image_metadata, so an ad-hoc snapshot works before a protocol
    (objective/labware) is configured."""

    def test_save_image_jpg_branch_uses_save_path_not_prepare(self):
        src = _method_src('modules/image_save.py', None, 'save_image')
        # The JPG branch must compute the path directly; prepare_image_for_saving
        # (which calls generate_image_metadata) is the else branch only.
        assert "if output_format == 'JPG':" in src
        jpg_idx = src.find("if output_format == 'JPG':")
        prepare_idx = src.find('prepare_image_for_saving', jpg_idx)
        save_path_idx = src.find('generate_image_save_path', jpg_idx)
        assert save_path_idx != -1, 'JPG branch must call generate_image_save_path.'
        # generate_image_save_path appears in the JPG branch, before
        # prepare_image_for_saving (which now lives in the else branch).
        assert save_path_idx < prepare_idx, (
            'JPG branch must resolve the path via generate_image_save_path '
            'before (and instead of) prepare_image_for_saving.'
        )
