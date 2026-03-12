# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for architecture audit fixes (2026-03-12).

Covers:
  1. Layer violation fixes — modules/ no longer hard-depends on ui/
  2. config_getters → config_ui_getters rename
  3. stitch_algorithms.py cleanup (feature_stitch, color_transfer, crop_to_content)
  4. Dead code removal (position_stitcher removed from stitcher.py)
  5. Tiny file consolidation — enums/classes merged into parent modules
"""

import importlib
import sys
import types

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Layer violation: modules/ must not hard-import from ui/
# ---------------------------------------------------------------------------

class TestLayerViolations:
    """Verify modules/ files don't have hard (top-level) imports from ui/."""

    def _check_no_toplevel_ui_import(self, module_path):
        """Read a module file and check no top-level 'from ui.' imports exist."""
        with open(module_path) as f:
            lines = f.readlines()

        violations = []
        in_string = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith('#'):
                continue
            # Track triple-quote strings (rough heuristic)
            if '"""' in stripped or "'''" in stripped:
                count = stripped.count('"""') + stripped.count("'''")
                if count % 2 == 1:
                    in_string = not in_string
                continue
            if in_string:
                continue
            # Skip indented lines (inside functions/classes = deferred import)
            if line[0] in (' ', '\t'):
                continue
            # Check for top-level ui imports
            if 'from ui.' in stripped or 'import ui.' in stripped:
                violations.append((i, stripped))

        return violations

    def test_config_ui_getters_no_toplevel_ui_import(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'config_ui_getters.py')
        violations = self._check_no_toplevel_ui_import(path)
        assert not violations, f"Top-level ui/ imports found: {violations}"

    def test_step_navigation_no_toplevel_ui_import(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'step_navigation.py')
        violations = self._check_no_toplevel_ui_import(path)
        assert not violations, f"Top-level ui/ imports found: {violations}"

    def test_tech_support_report_no_toplevel_ui_import(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'tech_support_report.py')
        violations = self._check_no_toplevel_ui_import(path)
        assert not violations, f"Top-level ui/ imports found: {violations}"


# ---------------------------------------------------------------------------
# 2. config_getters renamed to config_ui_getters
# ---------------------------------------------------------------------------

class TestConfigGettersRename:
    """Verify old config_getters.py is gone and new name exists."""

    def test_old_file_does_not_exist(self):
        import os
        old_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'config_getters.py')
        assert not os.path.exists(old_path), "Old config_getters.py still exists"

    def test_new_file_exists(self):
        import os
        new_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'config_ui_getters.py')
        assert os.path.exists(new_path), "config_ui_getters.py not found"

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
                        violations.append(f"{os.path.relpath(py_file, root)}:{i}")
        assert not violations, f"Files still importing old name: {violations}"


# ---------------------------------------------------------------------------
# 3. stitch_algorithms.py — cleaned up functions
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
        assert not os.path.exists(old_path), "Old image_stitcher.py still exists"

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
# 4. Dead code removal — position_stitcher removed from stitcher.py
# ---------------------------------------------------------------------------

class TestStitcherDeadCodeRemoved:
    """Verify unused position_stitcher() was removed from stitcher.py."""

    def test_position_stitcher_not_in_stitcher(self):
        from modules.stitcher import Stitcher
        assert not hasattr(Stitcher, 'position_stitcher'), \
            "Unused position_stitcher() should be removed from Stitcher"

    def test_simple_position_stitcher_still_exists(self):
        from modules.stitcher import Stitcher
        assert hasattr(Stitcher, '_simple_position_stitcher'), \
            "_simple_position_stitcher() should still exist"


# ---------------------------------------------------------------------------
# 5. Tiny file consolidation — merged into parent modules
# ---------------------------------------------------------------------------

class TestTinyFileConsolidation:
    """Verify tiny files were deleted and their contents moved to parent modules."""

    DELETED_FILES = [
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
            assert not os.path.exists(path), f"{filename} should be deleted"

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
        assert PostFunction.COMPOSITE.value == "Composite"
        assert PostFunction.HYPERSTACK.value == "Hyperstack"
        assert "Stitched" in PostFunction.list_values()

    def test_sequenced_capture_run_mode_in_executor(self):
        """Verify SequencedCaptureRunMode is defined in sequenced_capture_executor.py."""
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'sequenced_capture_executor.py')
        with open(path) as f:
            content = f.read()
        assert 'class SequencedCaptureRunMode' in content
        assert "FULL_PROTOCOL = 'full_protocol'" in content
        assert "SINGLE_SCAN = 'single_scan'" in content

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
            'modules.protocol_' + 'step',
        ]
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
                            violations.append(f"{os.path.relpath(py_file, root)}:{i} ({old_mod})")
        assert not violations, f"Files still importing deleted modules: {violations}"
