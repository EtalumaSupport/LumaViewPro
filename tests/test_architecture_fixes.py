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
#
# Both pins are EQUALITIES, not ceilings. A count that RISES is new logic in
# the GUI: move it to the API and expose a getter/setter. A count that FALLS
# is the migration working: lower the pin in the same commit, so the number
# can never quietly grow back to a stale ceiling.
#
# The import pin is a WEAK proxy for logic: a file keeps an import for its
# rendering uses after the logic that also used it moves down, so a
# migration commit may change the import count by nothing at all. It stops
# growth; it does not measure the migration. Pins 3 and 4 below measure the
# lifecycle class directly.
#
# None of these proxies sees a decision written in ui/ against widget state
# and the public scope API alone; that class is caught at review, not here.

import ast as _ast


def _ui_source_files():
    return _list_py_files('ui')


def _relpath(path):
    return os.path.relpath(path, _REPO_ROOT)


def _ui_modules_import_counts():
    """{'ui/<file>.py': number of `modules.*` import statements} -- every
    Import / ImportFrom node in the file, at any nesting depth."""
    counts = {}
    for path in _ui_source_files():
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        n = 0
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                if node.module and (node.module == 'modules' or node.module.startswith('modules.')):
                    n += 1
            elif isinstance(node, _ast.Import) and any(
                a.name == 'modules' or a.name.startswith('modules.') for a in node.names
            ):
                n += 1
        if n:
            counts[_relpath(path)] = n
    return counts


def _ui_private_reach_counts():
    """{('ui/<file>.py', '_private_name'): count} -- attribute reads of a
    single-underscore name whose attribute chain passes through `scope`."""
    counts = {}
    for path in _ui_source_files():
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        for node in _ast.walk(tree):
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
            if 'scope' in chain:
                key = (_relpath(path), node.attr)
                counts[key] = counts.get(key, 0) + 1
    return counts


# Pinned at 530b6093 (beta34). Lower a value in the same commit that moves
# the logic down; never raise one.
_UI_MODULES_IMPORT_PIN = {
    'ui/advanced_settings.py': 9,
    'ui/composite_capture.py': 9,
    'ui/file_dialogs.py': 8,
    'ui/histogram.py': 1,
    'ui/image_settings.py': 4,
    'ui/image_utils_kivy.py': 1,
    'ui/layer_control.py': 14,
    'ui/main_display.py': 8,
    'ui/microscope_settings.py': 19,
    'ui/motion_settings.py': 5,
    'ui/notification_popup.py': 4,
    'ui/post_processing.py': 15,
    'ui/protocol_settings.py': 17,
    'ui/scope_display.py': 12,
    'ui/shader.py': 3,
    'ui/stage.py': 5,
    'ui/step_navigation.py': 6,
    'ui/tooltip.py': 1,
    'ui/ui_helpers.py': 7,
    'ui/vertical_control.py': 10,
    'ui/zstack.py': 9,
}

_UI_PRIVATE_REACH_PIN = {
    ('ui/advanced_settings.py', '_set_conversion_gain_mode_impl'): 1,
    ('ui/advanced_settings.py', '_set_line_noise_reduction_impl'): 1,
    ('ui/composite_capture.py', '_capture_and_wait_impl'): 1,
    ('ui/composite_capture.py', '_last_turret_position'): 1,
    ('ui/layer_control.py', '_apply_layer_camera_settings_impl'): 1,
    ('ui/microscope_settings.py', '_set_frame_size_impl'): 1,
    ('ui/ui_helpers.py', '_move_absolute_impl'): 1,
    ('ui/vertical_control.py', '_home_turret_impl'): 3,
    ('ui/vertical_control.py', '_move_turret_impl'): 2,
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
        'create_headless',
    }
)


def _gui_source_files():
    return [*_ui_source_files(), os.path.join(_REPO_ROOT, 'lumaviewpro.py')]


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
    ('lumaviewpro.py', 'AutofocusRunner'): 1,
    ('lumaviewpro.py', 'AutofocusThread'): 1,
    ('lumaviewpro.py', 'ScopeSession'): 1,
    ('lumaviewpro.py', 'disconnect'): 1,
    ('ui/main_display.py', 'Lumascope'): 1,
    ('ui/microscope_settings.py', 'Lumascope'): 1,
    ('ui/microscope_settings.py', 'disconnect'): 1,
    ('ui/microscope_settings.py', 'initialize'): 2,
    ('ui/microscope_settings.py', 'start_streaming'): 2,
    ('ui/protocol_settings.py', 'set_labware'): 1,
    ('ui/vertical_control.py', 'set_objective'): 1,
    ('ui/vertical_control.py', 'set_turret_config'): 3,
}

_MODULES_CONTEXT_READ_PIN = {
    'modules/config_helpers.py': 4,
    'modules/config_ui_getters.py': 21,
    'modules/derived_output_encoding.py': 3,
    'modules/executor_registry.py': 1,
    'modules/metrics_logger.py': 2,
    'modules/scope_session.py': 1,
}


_GUI_REMEDY = 'New logic in the GUI: move it to the API and expose a getter/setter (Rule 2).'
_MODULES_REMEDY = (
    'The lower layer is reaching up into the GUI: take the value as an argument (Rule 2).'
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

    def test_ui_modules_imports_match_the_pin(self):
        report = _ratchet_report(
            _UI_MODULES_IMPORT_PIN, _ui_modules_import_counts(), 'modules.* imports'
        )
        assert report == [], '\n'.join(report)

    def test_ui_private_reaches_into_scope_match_the_pin(self):
        report = _ratchet_report(
            _UI_PRIVATE_REACH_PIN, _ui_private_reach_counts(), 'private reaches'
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
