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

from tests.ast_seams import REPO_ROOT


# ---------------------------------------------------------------------------
# 1. Layer violations -- Architecture Rule 1: only call/import down one level
# ---------------------------------------------------------------------------


def _list_py_files(subdir):
    """Return sorted list of *.py files in <repo>/<subdir>/, excluding dunder
    files like __init__.py.
    """
    pattern = os.path.join(REPO_ROOT, subdir, '*.py')
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
_DRIVERS_FILES = _list_py_files('drivers')
_LIB_FILES = _list_py_files('lib')


class TestLayerViolations:
    """Verify every source file respects Architecture Rule 1.

    Lower layers must not import upward. Higher layers may import down.
    Test parametrized over each *.py file in modules/, drivers/, lib/ --
    new files added under those directories are checked automatically.
    """

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

        old_path = os.path.join(REPO_ROOT, 'modules', 'config_getters.py')
        assert not os.path.exists(old_path), 'Old config_getters.py still exists'

    def test_new_file_exists(self):
        import os

        new_path = os.path.join(REPO_ROOT, 'modules', 'config_ui_getters.py')
        assert os.path.exists(new_path), 'config_ui_getters.py not found'

    def test_no_imports_reference_old_name(self):
        """Scan all .py files for 'modules.config_getters' imports (should be zero)."""
        import os
        import glob

        old_module = 'modules.config_' + 'getters'  # avoid matching this test file
        root = REPO_ROOT
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

        old_path = os.path.join(REPO_ROOT, 'modules', 'image_stitcher.py')
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

        modules_dir = os.path.join(REPO_ROOT, 'modules')
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
        root = REPO_ROOT
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

        root = REPO_ROOT
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


# ---------------------------------------------------------------------------
# 6. The GUI is display-only (Rule 2) -- the migration ratchet
# ---------------------------------------------------------------------------
#
# Rule 2's standing ruling: the API owns every piece of logic that can live
# there and the GUI renders API state, so the GUI stays replaceable and
# REST/headless callers get identical behaviour for free. The migration
# that moves existing logic down is in progress; nothing stopped NEW logic
# from landing in ui/ meanwhile. Two mechanical proxies for logic in the
# GUI, each pinned by site at introduction:
#
#   1. `ui/` importing `modules.*` directly (the architecture rule routes
#      the GUI to orchestration through the Session layer). Counted per
#      file, deferred imports included -- an import inside a method is the
#      commonest shape of GUI-owned logic.
#   2. `ui/` reaching a private attribute or `_impl` method on the scope
#      (`ctx.scope.motion._home_turret_impl`, `scope.imaging
#      ._set_frame_size_impl`): the GUI calling past the API surface into
#      its internals, which is a decision the API should expose instead.
#   3. an `except` handler in `ui/`: the GUI catching an outcome and
#      deciding what it means (which log level, which popup, which
#      sentence, whether to retry). Four commits in one fortnight each
#      added one that chose warning against error for a typed refusal,
#      after the plan for each said the GUI displays what the API decides;
#      the suite was green for all four because nothing counted them.
#      Counted per file and handler type, spelled by the exception's last
#      name so a module prefix cannot move a handler out of the census.
#
# Both pins are EQUALITIES, not ceilings. A count that RISES is new logic in
# the GUI: move it to the API and expose a getter/setter. A count that FALLS
# is the migration working: lower the pin in the same commit, so the number
# can never quietly grow back to a stale ceiling.
#
# Proxy 1 counted `modules.*` IMPORTS from ui/ until 2026-09-21, and it was
# measuring the wrong thing. Three facts retired it, all measured at
# `ace7b184`: of the 160 imports it counted, ONE was the API and none was
# the Session, while 135 were utility coupling (`notification_center`,
# `app_context`, `common_utils`, `exceptions`), which the migration will
# never remove; it counted the LEGAL direction, so it could not separate the
# GUI reaching past its boundary from the GUI using it; and it moved the
# WRONG WAY on a correct migration, since routing a widget at the Session
# means importing the Session -- the pin below carried a hand-written
# exemption for exactly that. Over the three months to 2026-09-21 it rose
# 145 -> 160 while orchestration went 16 -> 0 and widget reads 30 -> 5.
# History for any tree: `tools/ratchet_history.py`.
#
# None of these proxies sees a decision written in ui/ against widget state
# and the public scope API alone; that class is caught at review, not here.

import ast as _ast


def _ui_source_files():
    return _list_py_files('ui')


def _relpath(path):
    return os.path.relpath(path, REPO_ROOT)


# The answerer layer: the two modules that answer questions about the
# instrument and its configuration for the GUI. Every call into them from
# ui/ is a question the GUI answers below the API, and retiring one is the
# unit of migration work -- route the caller at its `ScopeSession` member
# and the call site goes away.
_ANSWERER_MODULES = frozenset({'config_ui_getters', 'config_helpers'})


def _ui_answerer_call_counts():
    """{'<gui file>.py': calls into the answerer layer}, both spellings.

    BOTH forms are counted together because counting either alone inverts
    the trend. Between 2026-06-22 and 2026-09-21 the attribute form
    (`config_helpers.get_x(...)`) rose 6 -> 24 while the imported-name form
    (`from modules.config_ui_getters import get_x` then `get_x()`) fell
    97 -> 70: a census of attribute calls alone would report a fourfold
    regression where the total in fact fell 103 -> 94.
    """
    counts = {}
    for path in _gui_source_files():
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        imported = set()
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.ImportFrom)
                and node.module
                and any(mod in node.module for mod in _ANSWERER_MODULES)
            ):
                imported |= {alias.asname or alias.name for alias in node.names}
        n = 0
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            func = node.func
            attribute_form = (
                isinstance(func, _ast.Attribute)
                and isinstance(func.value, _ast.Name)
                and func.value.id in _ANSWERER_MODULES
            )
            imported_name_form = isinstance(func, _ast.Name) and func.id in imported
            if attribute_form or imported_name_form:
                n += 1
        if n:
            counts[_relpath(path)] = n
    return counts


def _private_reaches_in_source(source, label):
    """{(label, '_private_name'): count} -- one file's reaches into an API private.

    A reach counts when EITHER the attribute chain written at the call site
    passes through `scope`, OR a single-underscore name ending `_impl` is read
    on a receiver that is not bare `self`.

    The second clause is here because the first pins the CALLER's spelling,
    which the caller owns and may change freely. Binding
    `imaging = ctx.lumaview.scope.imaging` and reaching through `imaging` left
    the chain without the word `scope`, which took two real reaches out of this
    census with no migration behind it. A member's name is the API's, not the
    caller's, and does not move when a local is renamed.

    A UNION, never a replacement. The chain clause is the only one that sees a
    reach into an API private that does not end `_impl` -- the LED-truth store
    and the active-objective store among them -- and 177 of the package's
    private members are of that kind.

    A bare `self` receiver is excluded from the name clause because `ui/` owns
    one `_impl` method of its own, `CompositeCapture._live_capture_impl`, bound
    as `action=self._live_capture_impl`. Reaching one's own body is not a reach
    into the API.
    """
    counts = {}
    for node in _ast.walk(_ast.parse(source)):
        if not isinstance(node, _ast.Attribute):
            continue
        if not node.attr.startswith('_') or node.attr.startswith('__'):
            continue
        chain = []
        value = node.value
        while isinstance(value, _ast.Attribute):
            chain.append(value.attr)
            value = value.value
        if isinstance(value, _ast.Name):
            chain.append(value.id)
        reaches_own_body = isinstance(node.value, _ast.Name) and node.value.id == 'self'
        if 'scope' in chain or (node.attr.endswith('_impl') and not reaches_own_body):
            key = (label, node.attr)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ui_private_reach_counts():
    """{('ui/<file>.py', '_private_name'): count} -- the GUI's reaches into an
    API private. The rule is `_private_reaches_in_source`."""
    counts = {}
    for path in _ui_source_files():
        with open(path) as fh:
            counts.update(_private_reaches_in_source(fh.read(), _relpath(path)))
    return counts


def _handler_type_name(node):
    """The census key for one `except` clause's type: the last name of each
    caught exception, sorted and joined, `<bare>` for a bare `except:`.

    `except exceptions.ConfigError` and `except ConfigError` are the same
    handler; keying on the spelling would let an import restyle move a
    handler out of the census, the failure the private-reach census had."""
    if node is None:
        return '<bare>'
    parts = node.elts if isinstance(node, _ast.Tuple) else [node]
    names = []
    for part in parts:
        names.append(
            part.attr
            if isinstance(part, _ast.Attribute)
            else getattr(part, 'id', _ast.unparse(part))
        )
    return ', '.join(sorted(names))


def _except_handlers_in_source(source, relpath):
    """{(relpath, '<type names>'): count} -- every `except` clause in one
    source file, by the census key of `_handler_type_name`."""
    counts = {}
    for node in _ast.walk(_ast.parse(source)):
        if isinstance(node, _ast.ExceptHandler):
            key = (relpath, _handler_type_name(node.type))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ui_except_counts():
    """{('ui/<file>.py', '<type names>'): count} -- the GUI's `except`
    handlers, each a place the GUI catches an outcome and decides what it
    means. The rule is `_except_handlers_in_source`."""
    counts = {}
    for path in _ui_source_files():
        with open(path) as fh:
            counts.update(_except_handlers_in_source(fh.read(), _relpath(path)))
    return counts


# Pinned at 5779bc04. Every entry is a question the GUI answers below the
# API. Lower a value in the same commit that routes its caller at the
# `ScopeSession` member; never raise one. Unlike the import count this
# replaced, this pin can only fall by the migration actually happening: a
# call site disappears when, and only when, the GUI stops answering.
_UI_ANSWERER_CALL_PIN = {
    'lumaviewpro.py': 1,
    'ui/advanced_settings.py': 3,
    'ui/image_settings.py': 5,
    'ui/layer_control.py': 6,
    'ui/microscope_settings.py': 6,
    'ui/motion_settings.py': 3,
    'ui/post_processing.py': 1,
    'ui/protocol_settings.py': 22,
    'ui/scope_display.py': 2,
    'ui/shader.py': 3,
    'ui/stage.py': 5,
    'ui/ui_helpers.py': 2,
    'ui/vertical_control.py': 9,
    'ui/zstack.py': 1,
}

# Every entry is a hardware write the GUI makes through the API's undispatched
# body, so the dispatcher's refusal, its typed error and its api.log line are
# skipped for that write and a REST caller gets them where the GUI does not.
# Lower a value in the same commit that routes its caller at a public member.
# A rise means a new reach and needs the same scrutiny as any other regression.
#
# The 12 -> 14 rise on 2026-09-22 was NOT a regression: no GUI code changed.
# The census keyed off the caller's spelling and could not see a reach written
# through a local alias, so `ui/microscope_settings.py`'s pixel-format and
# binning applies had never been counted. 12 and 14 are the same tree measured
# by a blind instrument and then a seeing one.
#
# `tools/ratchet_history.py` is unaffected and needs no annotation: it imports
# the detector by name and replays it over every past tree, so the whole series
# is recomputed by whichever rule ships today and gains no discontinuity here.
# `_private_reaches_in_source` carries the rule.
_UI_PRIVATE_REACH_PIN = {
    ('ui/advanced_settings.py', '_set_conversion_gain_mode_impl'): 1,
    ('ui/advanced_settings.py', '_set_line_noise_reduction_impl'): 1,
    ('ui/layer_control.py', '_apply_layer_camera_settings_impl'): 1,
    ('ui/microscope_settings.py', '_set_binning_size_impl'): 1,
    ('ui/microscope_settings.py', '_set_frame_size_impl'): 1,
    ('ui/microscope_settings.py', '_set_pixel_format_impl'): 1,
    ('ui/vertical_control.py', '_move_turret_impl'): 1,
}

# Every entry is an `except` handler in the GUI: a place the GUI catches an
# outcome and decides what it means. The API returns or raises a typed
# outcome and the GUI displays it; a handler that chooses a log level, a
# popup, a sentence or a retry is a decision that belongs below the API.
# Lower a value in the same commit that moves the decision down; a rise is
# a new decision in the GUI and needs Eric's word, not a pin edit. Pinned
# at 44f98226 from the tree, 151 handlers in 46 cells.
_UI_EXCEPT_PIN = {
    ('ui/advanced_settings.py', 'TypeError, ValueError'): 3,
    ('ui/composite_capture.py', 'Exception'): 1,
    ('ui/composite_capture.py', 'HardwareCommandRefusedError'): 1,
    ('ui/file_dialogs.py', 'Exception'): 5,
    ('ui/histogram.py', 'AttributeError, KeyError'): 1,
    ('ui/image_settings.py', 'Exception'): 2,
    ('ui/image_settings.py', 'KeyError'): 1,
    ('ui/layer_control.py', 'AttributeError, ImportError'): 1,
    ('ui/layer_control.py', 'Exception'): 17,
    ('ui/layer_control.py', 'KeyError'): 1,
    ('ui/layer_control.py', 'ProtocolError'): 1,
    ('ui/layer_control.py', 'TypeError, ValueError'): 1,
    ('ui/listener_bridge.py', 'Exception'): 2,
    ('ui/main_display.py', 'Exception'): 4,
    ('ui/main_display.py', 'RecordingRefusedError'): 1,
    ('ui/microscope_settings.py', 'Exception'): 12,
    ('ui/microscope_settings.py', 'FileNotFoundError'): 2,
    ('ui/microscope_settings.py', 'JSONDecodeError'): 2,
    ('ui/microscope_settings.py', 'KeyError'): 1,
    ('ui/microscope_settings.py', 'ValueError'): 1,
    ('ui/motion_settings.py', 'Exception'): 7,
    ('ui/motion_settings.py', 'ObjectiveUnknownError'): 1,
    ('ui/notification_popup.py', 'Exception'): 6,
    ('ui/post_processing.py', 'Exception'): 9,
    ('ui/post_processing.py', 'FileNotFoundError, ValueError'): 1,
    ('ui/post_processing.py', 'ValueError'): 1,
    ('ui/protocol_settings.py', 'ConfigError'): 3,
    ('ui/protocol_settings.py', 'Exception'): 18,
    ('ui/protocol_settings.py', 'OSError'): 1,
    ('ui/protocol_settings.py', 'ObjectiveUnknownError'): 1,
    ('ui/protocol_settings.py', 'ProtocolRunRefusedError'): 5,
    ('ui/protocol_settings.py', 'TypeError, ValueError'): 2,
    ('ui/protocol_settings.py', 'ValueError'): 2,
    ('ui/scope_display.py', 'Exception'): 3,
    ('ui/shader.py', 'Exception'): 2,
    ('ui/shader.py', 'ObjectiveUnknownError'): 1,
    ('ui/stage.py', 'Exception'): 7,
    ('ui/step_navigation.py', 'ProtocolRunRefusedError'): 1,
    ('ui/ui_helpers.py', 'AxisStateUnknownError'): 1,
    ('ui/ui_helpers.py', 'Exception'): 1,
    ('ui/ui_helpers.py', 'ObjectiveUnknownError'): 1,
    ('ui/ui_helpers.py', 'ProtocolRunRefusedError'): 2,
    ('ui/ui_helpers.py', 'RunAlreadyEndedError'): 1,
    ('ui/vertical_control.py', 'Exception'): 10,
    ('ui/vertical_control.py', 'ObjectiveUnknownError'): 1,
    ('ui/zstack.py', 'Exception'): 3,
}


# Proxy 3: the GUI calling the scope's LIFECYCLE and orchestration members
# directly -- bring-up, streaming, disconnect, the runtime-state setters, and
# constructing the scope, the session or an autofocus runner. Every one is a
# step the Session owns for a headless caller; a GUI that performs it is the
# reason a headless session cannot run the same step. This catches the
# settings-to-scope bring-up that the import and private-reach pins cannot:
# it reaches the scope through PUBLIC members with imports the file keeps
# for other uses. lumaviewpro.py is the GUI's entry point and is counted
# with ui/.
_ORCHESTRATION_MEMBERS = frozenset(
    {'initialize', 'start_streaming', 'stop_streaming', 'disconnect', 'set_acceleration_limit'}
)
_ORCHESTRATION_CONSTRUCTORS = frozenset(
    {
        'Lumascope',
        'ScopeSession',
        'AutofocusRunner',
        'AutofocusThread',
        'create_default',
    }
)


def _gui_source_files():
    return [*_ui_source_files(), os.path.join(REPO_ROOT, 'lumaviewpro.py')]


def _attribute_chain(node):
    chain = []
    value = node
    while isinstance(value, _ast.Attribute):
        chain.append(value.attr)
        value = value.value
    if isinstance(value, _ast.Name):
        chain.append(value.id)
    return chain


def _gui_orchestration_counts():
    """{('<gui file>', '<member or constructor>'): count} -- calls from the
    GUI to an orchestration member reached through the scope, the session or
    the app's `lumaview` handle, a `runtime_state.set_*` setter, or one of
    the orchestration constructors."""
    counts = {}
    for path in _gui_source_files():
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            fn = node.func
            name = None
            if isinstance(fn, _ast.Name) and fn.id in _ORCHESTRATION_CONSTRUCTORS:
                name = fn.id
            elif isinstance(fn, _ast.Attribute):
                chain = _attribute_chain(fn.value)
                through_scope = 'scope' in chain or 'session' in chain or 'lumaview' in chain
                if (
                    fn.attr in _ORCHESTRATION_CONSTRUCTORS
                    or (fn.attr in _ORCHESTRATION_MEMBERS and through_scope)
                    or (fn.attr.startswith('set_') and 'runtime_state' in chain)
                ):
                    name = fn.attr
            if name is not None:
                key = (_relpath(path), name)
                counts[key] = counts.get(key, 0) + 1
    return counts


# Proxy 4: the lower layer reaching UP into the GUI's context. A `modules/`
# file that reads `_app_ctx.ctx` is orchestration code that only works when
# a Kivy app has been built around it -- the mirror of proxy 3, and the
# other half of what a headless caller trips over. Counted per file.


def _modules_context_read_counts():
    """{'modules/<file>.py': number of `_app_ctx.ctx` attribute reads}."""
    counts = {}
    for path in _list_py_files('modules'):
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        n = 0
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.Attribute)
                and node.attr == 'ctx'
                and isinstance(node.value, _ast.Name)
                and node.value.id == '_app_ctx'
            ):
                n += 1
        if n:
            counts[_relpath(path)] = n
    return counts


# Pinned at 38f9a81c. Lower a value in the same commit that moves the step
# into the Session; never raise one.
_GUI_ORCHESTRATION_PIN = {
    ('lumaviewpro.py', 'AutofocusRunner'): 0,
    ('lumaviewpro.py', 'AutofocusThread'): 0,
    ('lumaviewpro.py', 'ScopeSession'): 0,
    ('lumaviewpro.py', 'disconnect'): 0,
    ('ui/main_display.py', 'Lumascope'): 0,
    ('ui/microscope_settings.py', 'Lumascope'): 0,
    ('ui/microscope_settings.py', 'disconnect'): 0,
    ('ui/microscope_settings.py', 'initialize'): 0,
    ('ui/microscope_settings.py', 'start_streaming'): 0,
    ('ui/protocol_settings.py', 'set_labware'): 0,
}

_MODULES_CONTEXT_READ_PIN = {
    'modules/config_helpers.py': 4,
    'modules/config_ui_getters.py': 12,
    'modules/derived_output_encoding.py': 3,
    'modules/executor_registry.py': 0,
    'modules/metrics_logger.py': 2,
    'modules/scope_session.py': 1,
}


# Proxy 5: the lower layer reading WIDGETS. A `modules/` file that subscripts
# `.ids[...]` is answering a question about the instrument by looking at a
# Kivy widget -- the API reading its settings off the GUI. Pin 4 counts the
# context reads that open the door; this counts every widget read behind
# them, alias or not.


def _modules_widget_read_counts():
    """{'modules/<file>.py': number of `<expr>.ids[...]` subscripts}."""
    counts = {}
    for path in _list_py_files('modules'):
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        n = sum(
            1
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Subscript)
            and isinstance(node.value, _ast.Attribute)
            and node.value.attr == 'ids'
        )
        if n:
            counts[_relpath(path)] = n
    return counts


# Proxy 6: the lower layer importing the GUI at ANY depth. The layer test
# above deliberately skips deferred imports; a `from ui...` inside a function
# is still an upward dependency, and a module that imports a popup cannot
# run headless. A rule with one existing violation, expressed as a ratchet
# that falls to zero.


def _lower_layer_ui_import_counts():
    """{'<modules|drivers>/<file>.py': number of `ui` / `ui.*` import
    statements at any nesting depth}."""
    counts = {}
    for path in _list_py_files('modules') + _list_py_files('drivers'):
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        n = 0
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                if node.module and (node.module == 'ui' or node.module.startswith('ui.')):
                    n += 1
            elif isinstance(node, _ast.Import) and any(
                a.name == 'ui' or a.name.startswith('ui.') for a in node.names
            ):
                n += 1
        if n:
            counts[_relpath(path)] = n
    return counts


# Pinned at 74153fad. Lower a value in the same commit that hands the value
# in as an argument or moves the popup to the caller; never raise one.
_MODULES_WIDGET_READ_PIN = {
    'modules/config_ui_getters.py': 2,
}


# Proxy 7: TWO ANSWERERS for one question. The five proxies above all count
# REACH -- who touches whom, in which direction. None of them can see the
# failure that actually stalls the migration: the same question answered in
# two places, one for the GUI and one for L2, so the GUI never goes through
# the Session and the two answers are free to drift.
#
# The measurement is the shape, not the names: a `config_ui_getters`
# function and a `ScopeSession` method that both forward to the SAME
# `config_helpers` function are two routes to one answer. Moving logic out
# of ui/ into a modules/ helper LOWERS proxies 1-3 and 5 while creating one
# of these, which is how a commit can score on every instrument and still
# leave the API not owning the answer.


def _twin_answerer_names():
    """{'<config_helpers function>': 1} for each one reached by BOTH a
    `config_ui_getters` function and a `ScopeSession` method."""

    def helper_calls_in(node):
        return {
            n.func.attr
            for n in _ast.walk(node)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and isinstance(n.func.value, _ast.Name)
            and n.func.value.id == 'config_helpers'
        }

    with open(os.path.join(REPO_ROOT, 'modules', 'config_ui_getters.py')) as fh:
        gui_tree = _ast.parse(fh.read())
    gui = set()
    for node in gui_tree.body:
        if isinstance(node, _ast.FunctionDef):
            gui |= helper_calls_in(node)

    with open(os.path.join(REPO_ROOT, 'modules', 'scope_session.py')) as fh:
        session_tree = _ast.parse(fh.read())
    session = set()
    for cls in session_tree.body:
        if isinstance(cls, _ast.ClassDef):
            for node in cls.body:
                if isinstance(node, _ast.FunctionDef):
                    session |= helper_calls_in(node)

    return dict.fromkeys(gui & session, 1)


# Pinned at 0e4a609e. Every entry is one question with a GUI answerer and an
# L2 answerer; the set may only shrink. Retire one by deleting the
# `config_ui_getters` forwarder and routing its GUI callers at the
# `ScopeSession` member -- never by adding a third.
#
# `get_sequenced_capture_config_from_settings` is the worked example, both
# ways: it entered this set at `ee765db0` (pre-REST item 1), in the commit
# that added `ScopeSession.get_sequenced_capture_config` beside the GUI's
# existing `get_sequenced_capture_config_from_ui` rather than routing the
# GUI at it -- the set went 5 -> 6 there and no instrument in this file
# moved -- and it left when the protocol panel was routed at the Session
# member and the forwarder deleted, the remedy below applied as written.
_TWIN_ANSWERER_PIN = {
    'get_auto_gain_settings': 1,
    'get_layer_configs': 1,
    'get_selected_labware_from_settings': 1,
}

# Empty, and empty is the achieved state: nothing under modules/ or
# drivers/ imports ui/. The pin stays so the next one is a rise from zero.
_LOWER_LAYER_UI_IMPORT_PIN: dict[str, int] = {}


_GUI_REMEDY = 'New logic in the GUI: move it to the API and expose a getter/setter (Rule 2).'
_EXCEPT_REMEDY = (
    'The GUI catches an outcome and decides what it means. The API returns or '
    'raises a typed outcome that already says its severity and its sentence; '
    'the GUI displays it (Rule 2).'
)
_ANSWERER_REMEDY = (
    'The GUI answers this below the API. Route the caller at the Lumascope '
    'API or at its ScopeSession member -- CLAUDE.md allows the GUI either -- '
    'and delete the call, rather than moving it to another helper '
    '(goal 1, Rule 35).'
)
_MODULES_REMEDY = (
    'The lower layer is reaching up into the GUI: take the value as an argument (Rule 2).'
)
_TWIN_REMEDY = (
    'Two answerers for one question. Moving logic out of ui/ is only half the '
    'migration -- it has to land ON the API. Route the GUI at the ScopeSession '
    'member and delete the config_ui_getters forwarder (goal 1, Rule 35).'
)


def _ratchet_report(pin, actual, what, remedy=_GUI_REMEDY):
    """The differences between a pin and the tree, each with its remedy."""
    lines = []
    for key in sorted(set(pin) | set(actual), key=str):
        before, now = pin.get(key, 0), actual.get(key, 0)
        if now > before:
            lines.append(f'{key}: {what} rose {before} -> {now}. {remedy}')
        elif now < before:
            lines.append(f'{key}: {what} fell {before} -> {now}. Lower the pin in this commit.')
    return lines


class TestGuiIsDisplayOnly:
    """Rule 2's migration ratchet: logic in ui/ does not grow, and the pin
    tracks every drop."""

    def test_ui_answerer_calls_match_the_pin(self):
        report = _ratchet_report(
            _UI_ANSWERER_CALL_PIN,
            _ui_answerer_call_counts(),
            'answerer calls',
            _ANSWERER_REMEDY,
        )
        assert report == [], '\n'.join(report)

    def test_ui_private_reaches_match_the_pin(self):
        report = _ratchet_report(
            _UI_PRIVATE_REACH_PIN, _ui_private_reach_counts(), 'private reaches'
        )
        assert report == [], '\n'.join(report)

    def test_ui_except_handlers_match_the_pin(self):
        report = _ratchet_report(
            _UI_EXCEPT_PIN, _ui_except_counts(), 'except handlers', _EXCEPT_REMEDY
        )
        assert report == [], '\n'.join(report)

    def test_gui_orchestration_calls_match_the_pin(self):
        report = _ratchet_report(
            _GUI_ORCHESTRATION_PIN, _gui_orchestration_counts(), 'orchestration calls'
        )
        assert report == [], '\n'.join(report)

    def test_modules_context_reads_match_the_pin(self):
        report = _ratchet_report(
            _MODULES_CONTEXT_READ_PIN,
            _modules_context_read_counts(),
            '_app_ctx.ctx reads',
            _MODULES_REMEDY,
        )
        assert report == [], '\n'.join(report)

    def test_modules_widget_reads_match_the_pin(self):
        report = _ratchet_report(
            _MODULES_WIDGET_READ_PIN, _modules_widget_read_counts(), '.ids[ reads', _MODULES_REMEDY
        )
        assert report == [], '\n'.join(report)

    def test_lower_layer_ui_imports_match_the_pin(self):
        report = _ratchet_report(
            _LOWER_LAYER_UI_IMPORT_PIN,
            _lower_layer_ui_import_counts(),
            'ui imports',
            _MODULES_REMEDY,
        )
        assert report == [], '\n'.join(report)

    def test_twin_answerers_match_the_pin(self):
        report = _ratchet_report(
            _TWIN_ANSWERER_PIN, _twin_answerer_names(), 'GUI+Session answerers', _TWIN_REMEDY
        )
        assert report == [], '\n'.join(report)


class TestThePrivateReachCensusCountsReachesNotSpellings:
    """A reach must not leave the census by being renamed.

    The census keyed off the caller's spelling, asking whether the literal
    word `scope` appeared in the attribute chain written at the call site.
    Binding the sub-object to a local first took two real reaches out of the
    count with no migration -- and one of them, `ui/microscope_settings.py`'s
    binning apply, is the site whose own comment records the production
    incident the whole cluster exists because of.
    """

    _LABEL = 'ui/fixture.py'
    _DIRECT = 'def f(ctx):\n    ctx.lumaview.scope.imaging._set_pixel_format_impl(fmt)\n'
    _ALIASED = (
        'def f(ctx):\n'
        '    imaging = ctx.lumaview.scope.imaging\n'
        '    imaging._set_pixel_format_impl(fmt)\n'
    )

    def test_an_aliased_reach_counts_the_same_as_a_direct_one(self):
        direct = _private_reaches_in_source(self._DIRECT, self._LABEL)
        aliased = _private_reaches_in_source(self._ALIASED, self._LABEL)
        # Asserted non-zero first: a census that counts nothing would satisfy
        # the equality below while seeing neither reach.
        assert sum(direct.values()) == 1, direct
        assert aliased == direct, f'aliased {aliased} != direct {direct}'

    def test_a_non_impl_api_private_still_counts(self):
        """The clause that sees the stores, which the `_impl` clause cannot.

        `_led_state` is the LED-truth store and `_objective_id` is the
        active-objective store; neither ends in `_impl`. A later
        simplification that drops the chain clause in favour of the name
        clause alone would stop seeing both, so this pins the union.
        """
        for member in ('_led_state', '_objective_id'):
            source = f'def f(ctx):\n    return ctx.scope.illumination.{member}\n'
            counts = _private_reaches_in_source(source, self._LABEL)
            assert counts == {(self._LABEL, member): 1}, counts

    def test_a_widgets_own_impl_method_is_not_a_reach(self):
        """`ui/` owns exactly one `_impl` method of its own.

        `CompositeCapture._live_capture_impl`, bound as
        `action=self._live_capture_impl`. Reaching one's own body is not
        reaching into the API, so the name clause excludes a bare `self`
        receiver.
        """
        source = 'def f(self):\n    return self._live_capture_impl\n'
        assert _private_reaches_in_source(source, self._LABEL) == {}


# Announced at the end of every run (tests/ratchets.py).
from tests import ratchets as _ratchets

_ratchets.register(
    'GUI: questions the GUI answers below the API',
    lambda: sum(_ui_answerer_call_counts().values()),
    sum(_UI_ANSWERER_CALL_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: private API reaches from ui/',
    lambda: sum(_ui_private_reach_counts().values()),
    sum(_UI_PRIVATE_REACH_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: except handlers in ui/',
    lambda: sum(_ui_except_counts().values()),
    sum(_UI_EXCEPT_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: orchestration calls from the GUI',
    lambda: sum(_gui_orchestration_counts().values()),
    sum(_GUI_ORCHESTRATION_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: _app_ctx.ctx reads in modules/',
    lambda: sum(_modules_context_read_counts().values()),
    sum(_MODULES_CONTEXT_READ_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: widget .ids[ reads in modules/',
    lambda: sum(_modules_widget_read_counts().values()),
    sum(_MODULES_WIDGET_READ_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: ui imports from modules/ and drivers/',
    lambda: sum(_lower_layer_ui_import_counts().values()),
    sum(_LOWER_LAYER_UI_IMPORT_PIN.values()),
    'equal',
)
_ratchets.register(
    'GUI: questions with a GUI answerer and an L2 answerer',
    lambda: sum(_twin_answerer_names().values()),
    sum(_TWIN_ANSWERER_PIN.values()),
    'equal',
)
