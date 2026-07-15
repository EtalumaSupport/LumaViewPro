# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for architecture audit fixes (2026-03-12, expanded 2026-05-02 LAYER-G).

Covers:
  1. Layer violation fixes -- full directory scan per Architecture Rule 1
     (Firmware/docs/CLAUDE.md): lower layers must not import upward.
       - modules/*.py must not import from ui/
       - drivers/*.py must not import from modules/ or ui/
       - lib/*.py must not import from drivers/, modules/, or ui/
  2. config_getters -> config_ui_getters rename
  3. stitch_algorithms.py cleanup (feature_stitch, color_transfer, crop_to_content)
  4. Dead code removal (position_stitcher removed from stitcher.py)
  5. Tiny file consolidation -- enums/classes merged into parent modules
"""

import glob
import os
from typing import ClassVar

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Layer violations -- Architecture Rule 1: only call/import down one level
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _list_py_files(subdir):
    """Return sorted list of *.py files in <repo>/<subdir>/, excluding dunder
    files like __init__.py.
    """
    pattern = os.path.join(_REPO_ROOT, subdir, '*.py')
    return sorted(p for p in glob.glob(pattern) if not os.path.basename(p).startswith('__'))


def _check_no_toplevel_imports(module_path, forbidden_prefixes):
    """Read a source file and return [(line_no, line)] for top-level
    `from <prefix>...` or `import <prefix>...` imports where <prefix> is in
    forbidden_prefixes (e.g. ('ui.',) or ('modules.', 'ui.')).

    Skips: comments, indented imports (deferred inside functions/methods),
    and lines inside triple-quoted strings (rough heuristic).
    """
    with open(module_path) as f:
        lines = f.readlines()

    violations = []
    in_string = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if '"""' in stripped or "'''" in stripped:
            count = stripped.count('"""') + stripped.count("'''")
            if count % 2 == 1:
                in_string = not in_string
            continue
        if in_string:
            continue
        # Skip indented lines (inside functions/classes = deferred import)
        if line and line[0] in (' ', '\t'):
            continue
        for prefix in forbidden_prefixes:
            if f'from {prefix}' in stripped or f'import {prefix}' in stripped:
                violations.append((i, stripped))
                break
    return violations


# Files with known layer violations awaiting their structural fix.
# Each entry: filename -> (audit-cluster-or-LAYER-id, reason).
# When the structural fix lands, the file either disappears (e.g. shim
# deleted) or its imports change (xfail flips to xpass and the entry
# can be removed).
_KNOWN_MODULE_VIOLATIONS = {
    # Empty: modules/ui_helpers.py shim retired in LAYER-A' (deleted
    # along with modules/scope_commands.py). Add new entries here when
    # an audit surfaces a known-violating file with a stated retire-by
    # gate.
}


def _parametrized_with_known_violations(paths, known):
    """Build a parametrize list, attaching xfail(strict=False) marks to
    files in `known`. xfail flips to xpass when the violation is fixed
    elsewhere -- non-strict so it doesn't fail the run; the file is
    expected to disappear from `paths` once the structural fix lands.
    """
    out = []
    for p in paths:
        name = os.path.basename(p)
        if name in known:
            cluster_id, reason = known[name]
            out.append(
                pytest.param(
                    p,
                    id=name,
                    marks=pytest.mark.xfail(
                        strict=False,
                        reason=f'{cluster_id}: {reason}',
                    ),
                )
            )
        else:
            out.append(pytest.param(p, id=name))
    return out


_MODULES_FILES = _list_py_files('modules')
_DRIVERS_FILES = _list_py_files('drivers')
_LIB_FILES = _list_py_files('lib')


class TestLayerViolations:
    """Verify every source file respects Architecture Rule 1.

    Lower layers must not import upward. Higher layers may import down.
    Test parametrized over each *.py file in modules/, drivers/, lib/ --
    new files added under those directories are checked automatically.
    """

    @pytest.mark.parametrize(
        'module_path',
        _parametrized_with_known_violations(
            _MODULES_FILES,
            _KNOWN_MODULE_VIOLATIONS,
        ),
    )
    def test_modules_no_toplevel_ui_import(self, module_path):
        """modules/ must not import from ui/ (Rule 1: modules below ui)."""
        violations = _check_no_toplevel_imports(module_path, ('ui.',))
        assert not violations, (
            f'{os.path.basename(module_path)}: top-level ui/ imports found: {violations}'
        )

    @pytest.mark.parametrize(
        'driver_path',
        _DRIVERS_FILES,
        ids=lambda p: os.path.basename(p),
    )
    def test_drivers_no_toplevel_modules_or_ui_import(self, driver_path):
        """drivers/ must not import from modules/ or ui/ (Rule 1: drivers
        are below both). Shared utilities go in lib/ (e.g. lib/profile_trace).
        """
        violations = _check_no_toplevel_imports(
            driver_path,
            ('modules.', 'ui.'),
        )
        assert not violations, (
            f'{os.path.basename(driver_path)}: top-level modules/ or ui/ '
            f'imports found: {violations}'
        )

    @pytest.mark.parametrize(
        'lib_path',
        _LIB_FILES,
        ids=lambda p: os.path.basename(p),
    )
    def test_lib_no_toplevel_drivers_modules_or_ui_import(self, lib_path):
        """lib/ must be cross-layer-shared and dependency-free relative to
        drivers/, modules/, ui/. lib/ can only import stdlib + same-layer.
        """
        violations = _check_no_toplevel_imports(
            lib_path,
            ('drivers.', 'modules.', 'ui.'),
        )
        assert not violations, (
            f'{os.path.basename(lib_path)}: top-level drivers/, modules/, '
            f'or ui/ imports found: {violations}'
        )


# ---------------------------------------------------------------------------
# 2. config_getters renamed to config_ui_getters
# ---------------------------------------------------------------------------


class TestConfigGettersRename:
    """Verify old config_getters.py is gone and new name exists."""

    def test_old_file_does_not_exist(self):
        import os

        old_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'config_getters.py')
        assert not os.path.exists(old_path), 'Old config_getters.py still exists'

    def test_new_file_exists(self):
        import os

        new_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'config_ui_getters.py')
        assert os.path.exists(new_path), 'config_ui_getters.py not found'

    def test_no_imports_reference_old_name(self):
        """Scan all .py files for 'modules.config_getters' imports (should be zero)."""
        import os
        import glob

        old_module = 'modules.config_' + 'getters'  # avoid matching this test file
        root = os.path.join(os.path.dirname(__file__), '..')
        violations = []
        for py_file in glob.glob(os.path.join(root, '**', '*.py'), recursive=True):
            if '__pycache__' in py_file or 'test_architecture' in py_file:
                continue
            with open(py_file) as f:
                for i, line in enumerate(f, 1):
                    if old_module in line and not line.strip().startswith('#'):
                        violations.append(f'{os.path.relpath(py_file, root)}:{i}')
        assert not violations, f'Files still importing old name: {violations}'


# ---------------------------------------------------------------------------
# 3. stitch_algorithms.py -- cleaned up functions
# ---------------------------------------------------------------------------


class TestStitchAlgorithmsModule:
    """Verify stitch_algorithms.py exports the right functions."""

    def test_imports_succeed(self):
        from modules.stitch_algorithms import feature_stitch, color_transfer, crop_to_content

        assert callable(feature_stitch)
        assert callable(color_transfer)
        assert callable(crop_to_content)

    def test_old_module_gone(self):
        import os

        old_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'image_stitcher.py')
        assert not os.path.exists(old_path), 'Old image_stitcher.py still exists'

    def test_feature_stitch_rejects_single_image(self):
        from modules.stitch_algorithms import feature_stitch

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = feature_stitch([img])
        assert result is None

    def test_feature_stitch_rejects_empty_list(self):
        from modules.stitch_algorithms import feature_stitch

        result = feature_stitch([])
        assert result is None

    def test_color_transfer_preserves_shape(self):
        from modules.stitch_algorithms import color_transfer

        source = np.full((60, 80, 3), 200, dtype=np.uint8)
        target = np.full((40, 50, 3), 100, dtype=np.uint8)
        result = color_transfer(source, target)
        assert result.shape == target.shape
        assert result.dtype == np.uint8

    def test_color_transfer_shifts_brightness(self):
        from modules.stitch_algorithms import color_transfer

        bright = np.full((50, 50, 3), 220, dtype=np.uint8)
        dark = np.full((50, 50, 3), 50, dtype=np.uint8)
        result = color_transfer(bright, dark)
        assert result.mean() > dark.mean()

    def test_crop_to_content_removes_border(self):
        from modules.stitch_algorithms import crop_to_content

        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[40:160, 60:240] = 128
        result = crop_to_content(img)
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]
        assert result.mean() > 0


# ---------------------------------------------------------------------------
# 4. Dead code removal -- position_stitcher removed from stitcher.py
# ---------------------------------------------------------------------------


class TestStitcherDeadCodeRemoved:
    """Verify unused position_stitcher() was removed from stitcher.py."""

    def test_position_stitcher_not_in_stitcher(self):
        from modules.stitcher import Stitcher

        assert not hasattr(Stitcher, '_position_stitcher'), (
            'Unused _position_stitcher() should be removed from Stitcher'
        )

    def test_simple_position_stitcher_still_exists(self):
        from modules.stitcher import Stitcher

        assert hasattr(Stitcher, '_simple_position_stitcher'), (
            '_simple_position_stitcher() should still exist'
        )


# ---------------------------------------------------------------------------
# 5. Tiny file consolidation -- merged into parent modules
# ---------------------------------------------------------------------------


class TestTinyFileConsolidation:
    """Verify tiny files were deleted and their contents moved to parent modules."""

    DELETED_FILES: ClassVar[list] = [
        'stitcher_helper.py',
        'processing_utils.py',
        'protocol_step.py',
        'color_channels.py',
        'json_helper.py',
        'protocol_post_processing_functions.py',
        'sequenced_capture_run_modes.py',
    ]

    def test_deleted_files_are_gone(self):
        import os

        modules_dir = os.path.join(os.path.dirname(__file__), '..', 'modules')
        for filename in self.DELETED_FILES:
            path = os.path.join(modules_dir, filename)
            assert not os.path.exists(path), f'{filename} should be deleted'

    def test_color_channel_in_common_utils(self):
        from modules.common_utils import ColorChannel

        assert ColorChannel.Blue.value == 0
        assert ColorChannel.Lumi.value == 6
        assert len(ColorChannel) == 7

    def test_custom_jsonizer_in_common_utils(self):
        import json
        import numpy as np
        from modules.common_utils import CustomJSONizer

        data = {'a': np.int64(42), 'b': np.float64(3.14), 'c': np.bool_(True)}
        result = json.loads(json.dumps(data, cls=CustomJSONizer))
        assert result == {'a': 42, 'b': 3.14, 'c': True}

    def test_post_function_in_common_utils(self):
        from modules.common_utils import PostFunction

        assert PostFunction.COMPOSITE.value == 'Composite'
        assert PostFunction.HYPERSTACK.value == 'Hyperstack'
        assert 'Stitched' in PostFunction.list_values()

    def test_sequenced_capture_run_mode_importable_from_executor(self):
        """Verify SequencedCaptureRunMode is importable from sequenced_capture_runner."""
        from modules.sequenced_capture_runner import SequencedCaptureRunMode

        assert SequencedCaptureRunMode.FULL_PROTOCOL.value == 'full_protocol'
        assert SequencedCaptureRunMode.SINGLE_SCAN.value == 'single_scan'

    def test_no_imports_reference_old_modules(self):
        """Scan all .py files for imports of deleted modules (should be zero)."""
        import os
        import glob

        old_modules = [
            'modules.color_' + 'channels',
            'modules.json_' + 'helper',
            'modules.protocol_post_processing_' + 'functions',
            'modules.sequenced_capture_run_' + 'modes',
            'modules.stitcher_' + 'helper',
            'modules.processing_' + 'utils',
        ]
        # Deleted module that must not be confused with protocol_step_runner
        old_protocol_step = 'modules.protocol_' + 'step'
        root = os.path.join(os.path.dirname(__file__), '..')
        violations = []
        for py_file in glob.glob(os.path.join(root, '**', '*.py'), recursive=True):
            if '__pycache__' in py_file or 'test_architecture' in py_file:
                continue
            with open(py_file) as f:
                for i, line in enumerate(f, 1):
                    if line.strip().startswith('#'):
                        continue
                    for old_mod in old_modules:
                        if old_mod in line:
                            violations.append(f'{os.path.relpath(py_file, root)}:{i} ({old_mod})')
                    # Check for deleted modules.protocol_step but not protocol_step_runner
                    if old_protocol_step in line:
                        # Only flag if it's not followed by '_' (which would be protocol_step_runner)
                        import re

                        if re.search(r'modules\.protocol_step(?!_)', line):
                            violations.append(
                                f'{os.path.relpath(py_file, root)}:{i} ({old_protocol_step})'
                            )
        assert not violations, f'Files still importing deleted modules: {violations}'

    def test_all_python_files_compile(self):
        """Every .py file must be valid Python syntax -- catches refactor leftovers."""
        import os
        import py_compile

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        errors = []
        for dirpath, _, filenames in os.walk(root):
            if '__pycache__' in dirpath or '.git' in dirpath:
                continue
            for fn in filenames:
                if not fn.endswith('.py'):
                    continue
                filepath = os.path.join(dirpath, fn)
                try:
                    py_compile.compile(filepath, doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append(str(e))
        assert not errors, 'Syntax errors found:\n' + '\n'.join(errors)
