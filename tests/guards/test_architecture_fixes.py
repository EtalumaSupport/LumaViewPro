# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for architecture audit fixes (2026-03-12, expanded 2026-05-02 LAYER-G).

Covers (the layering of src/_lumascope/lib/ and src/_lumascope/drivers/ is
tests/guards/test_the_lower_layers_import_nothing_above_them.py's):
  2. config_getters retired (its successor, config_ui_getters, retired in turn)
  3. stitch_algorithms.py replaces image_stitcher.py
  4. Dead code removal (position_stitcher removed from stitcher.py)
  5. Tiny file consolidation -- enums/classes merged into parent modules
"""

import os
import sys
from typing import ClassVar

from tests import ratchets as _ratchets
from tests.ast_seams import REPO_ROOT, MemberNames, Packages, tracked_files


def _list_py_files(subdir):
    """Every tracked .py under <repo>/<subdir>/ at any depth, a package's
    __init__.py included: a read added in a subpackage or an __init__ is a
    read like any other, sorted."""
    return [os.path.join(REPO_ROOT, rel) for rel in tracked_files((subdir,))]


# ---------------------------------------------------------------------------
# 2. config_getters retired
# ---------------------------------------------------------------------------


class TestConfigGettersRename:
    """Verify the old config_getters.py is gone and nothing imports it."""

    def test_old_file_does_not_exist(self):
        import os

        old_path = os.path.join(REPO_ROOT, 'src/_lumascope/modules', 'config_getters.py')
        assert not os.path.exists(old_path), 'Old config_getters.py still exists'

    def test_no_imports_reference_old_name(self):
        """Scan every tracked .py file for '_lumascope.modules.config_getters' imports (should be zero)."""
        old_module = '_lumascope.modules.config_' + 'getters'  # avoid matching this test file
        violations = []
        for rel_path in tracked_files():
            if 'test_architecture' in rel_path:
                continue
            with open(REPO_ROOT / rel_path, encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if old_module in line and not line.strip().startswith('#'):
                        violations.append(f'{rel_path}:{i}')
        assert not violations, f'Files still importing old name: {violations}'


# ---------------------------------------------------------------------------
# 3. stitch_algorithms.py -- cleaned up functions
# ---------------------------------------------------------------------------


class TestStitchAlgorithmsModule:
    """stitch_algorithms.py replaced image_stitcher.py."""

    def test_old_module_gone(self):
        import os

        old_path = os.path.join(REPO_ROOT, 'src/_lumascope/modules', 'image_stitcher.py')
        assert not os.path.exists(old_path), 'Old image_stitcher.py still exists'


# ---------------------------------------------------------------------------
# 4. Dead code removal -- position_stitcher removed from stitcher.py
# ---------------------------------------------------------------------------


class TestStitcherDeadCodeRemoved:
    """Verify unused position_stitcher() was removed from stitcher.py."""

    def test_position_stitcher_not_in_stitcher(self):
        from _lumascope.modules.stitcher import Stitcher

        assert not hasattr(Stitcher, '_position_stitcher'), (
            'Unused _position_stitcher() should be removed from Stitcher'
        )

    def test_simple_position_stitcher_still_exists(self):
        from _lumascope.modules.stitcher import Stitcher

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

        modules_dir = os.path.join(REPO_ROOT, 'src/_lumascope/modules')
        for filename in self.DELETED_FILES:
            path = os.path.join(modules_dir, filename)
            assert not os.path.exists(path), f'{filename} should be deleted'

    def test_post_function_in_common_utils(self):
        from _lumascope.modules.common_utils import PostFunction

        assert PostFunction.COMPOSITE.value == 'Composite'
        assert PostFunction.HYPERSTACK.value == 'Hyperstack'
        assert 'Stitched' in PostFunction.list_values()

    def test_sequenced_capture_run_mode_importable_from_executor(self):
        """Verify SequencedCaptureRunMode is importable from sequenced_capture_runner."""
        from _lumascope.modules.sequenced_capture_runner import SequencedCaptureRunMode

        assert SequencedCaptureRunMode.FULL_PROTOCOL.value == 'full_protocol'
        assert SequencedCaptureRunMode.SINGLE_SCAN.value == 'single_scan'

    def test_no_imports_reference_old_modules(self):
        """Scan every tracked .py file for imports of deleted modules (should be zero)."""
        old_modules = [
            '_lumascope.modules.color_' + 'channels',
            '_lumascope.modules.json_' + 'helper',
            '_lumascope.modules.protocol_post_processing_' + 'functions',
            '_lumascope.modules.sequenced_capture_run_' + 'modes',
            '_lumascope.modules.stitcher_' + 'helper',
            '_lumascope.modules.processing_' + 'utils',
        ]
        # Deleted module that must not be confused with protocol_step_runner
        old_protocol_step = '_lumascope.modules.protocol_' + 'step'
        violations = []
        for rel_path in tracked_files():
            if 'test_architecture' in rel_path:
                continue
            with open(REPO_ROOT / rel_path, encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if line.strip().startswith('#'):
                        continue
                    for old_mod in old_modules:
                        if old_mod in line:
                            violations.append(f'{rel_path}:{i} ({old_mod})')
                    # Check for deleted _lumascope.modules.protocol_step but not protocol_step_runner
                    if old_protocol_step in line:
                        # Only flag if it's not followed by '_' (which would be protocol_step_runner)
                        import re

                        if re.search(r'_lumascope\.modules\.protocol_step(?!_)', line):
                            violations.append(f'{rel_path}:{i} ({old_protocol_step})')
        assert not violations, f'Files still importing deleted modules: {violations}'

    def test_all_python_files_compile(self):
        """Every tracked .py file is valid Python syntax -- catches refactor leftovers."""
        import py_compile

        errors = []
        for rel_path in tracked_files():
            try:
                py_compile.compile(str(REPO_ROOT / rel_path), doraise=True)
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


# The answerer layer: the module that answers questions about the
# instrument and its configuration below the API. Every call into it from
# ui/ is a question the GUI answers below the API, and retiring one is the
# unit of migration work -- route the caller at its `ScopeSession` member
# and the call site goes away.
_ANSWERER_MODULES = Packages({'config_helpers'}, under='src/_lumascope/modules')


def _ui_answerer_call_counts():
    """{'<gui file>.py': calls into the answerer layer}, both spellings.

    BOTH forms are counted together because counting either alone inverts
    the trend. Between 2026-06-22 and 2026-09-21 the attribute form
    (`config_helpers.get_x(...)`) rose 6 -> 24 while the imported-name form
    (`from _lumascope.modules.config_ui_getters import get_x` then `get_x()`) fell
    97 -> 70: a census of attribute calls alone would report a fourfold
    regression where the total in fact fell 103 -> 94.
    """
    counts = {}
    for path in _gui_source_files():
        with open(path, encoding='utf-8') as fh:
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
        with open(path, encoding='utf-8') as fh:
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
        with open(path, encoding='utf-8') as fh:
            counts.update(_except_handlers_in_source(fh.read(), _relpath(path)))
    return counts


# Pinned at 5779bc04. Every entry is a question the GUI answers below the
# API. Lower a value in the same commit that routes its caller at the
# `ScopeSession` member; never raise one. Unlike the import count this
# replaced, this pin can only fall by the migration actually happening: a
# call site disappears when, and only when, the GUI stops answering.
_UI_ANSWERER_CALL_PIN = {
    'ui/advanced_settings.py': 1,
    'ui/microscope_settings.py': 0,
    'ui/ui_helpers.py': 1,
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
    ('ui/microscope_settings.py', '_set_binning_impl'): 0,
    ('ui/microscope_settings.py', '_set_frame_size_impl'): 0,
    ('ui/microscope_settings.py', '_set_pixel_format_impl'): 0,
}

# Every entry is an `except` handler in the GUI: a place the GUI catches an
# outcome and decides what it means. The API returns or raises a typed
# outcome and the GUI displays it; a handler that chooses a log level, a
# popup, a sentence or a retry is a decision that belongs below the API.
# Lower a value in the same commit that moves the decision down; a rise is
# a new decision in the GUI and needs Eric's word, not a pin edit. Pinned
# at 44f98226 from the tree, 151 handlers in 46 cells.
_UI_EXCEPT_PIN = {
    ('ui/advanced_settings.py', 'TypeError, ValueError'): 0,
    ('ui/file_dialogs.py', 'Exception'): 1,
    ('ui/histogram.py', 'AttributeError, KeyError'): 0,
    ('ui/image_settings.py', 'Exception'): 0,
    ('ui/image_settings.py', 'KeyError'): 0,
    ('ui/channel_control.py', 'Exception'): 0,
    ('ui/channel_control.py', 'ProtocolError'): 0,
    ('ui/channel_control.py', 'TypeError, ValueError'): 0,
    ('ui/listener_bridge.py', 'Exception'): 0,
    ('ui/main_display.py', 'Exception'): 0,
    ('ui/microscope_settings.py', 'Exception'): 1,
    ('ui/microscope_settings.py', 'ValueError'): 0,
    ('ui/motion_settings.py', 'Exception'): 0,
    ('ui/notification_popup.py', 'Exception'): 0,
    ('ui/post_processing.py', 'Exception'): 0,
    ('ui/post_processing.py', 'FileNotFoundError, ValueError'): 0,
    ('ui/protocol_settings.py', 'ConfigError'): 0,
    ('ui/protocol_settings.py', 'Exception'): 1,
    ('ui/protocol_settings.py', 'TypeError, ValueError'): 0,
    ('ui/protocol_settings.py', 'ValueError'): 0,
    ('ui/scope_display.py', 'Exception'): 0,
    ('ui/shader.py', 'Exception'): 0,
    ('ui/stage.py', 'Exception'): 0,
    ('ui/step_navigation.py', 'ProtocolRunRefusedError'): 0,
    # The walk driver's own stop (a step it cannot perform, or a fault it
    # injected refused), not an API outcome; one catch where the driver runs
    # a step (Eric, 2026-10-04; the refusal folded in, 2026-10-30).
    ('ui/walk.py', 'Refusal, WalkStepError'): 1,
    ('ui/ui_helpers.py', 'Exception'): 1,
    ('ui/ui_helpers.py', 'ValueError'): 1,
    ('ui/vertical_control.py', 'Exception'): 0,
    ('ui/zstack.py', 'Exception'): 0,
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
_ORCHESTRATION_MEMBERS = MemberNames(
    (
        'src/_lumascope/modules/lumascope_api/_lumascope.py',
        'src/_lumascope/modules/lumascope_api/imaging.py',
        'src/_lumascope/modules/lumascope_api/motion.py',
    ),
    {
        'initialize',
        'enable_streaming',
        'disable_streaming',
        'disconnect',
        'set_acceleration_limit_pct',
    },
)
_ORCHESTRATION_CONSTRUCTORS = MemberNames(
    (
        'src/_lumascope/modules/lumascope_api/_lumascope.py',
        'src/_lumascope/modules/scope_session.py',
        'src/_lumascope/modules/autofocus_runner.py',
        'src/_lumascope/modules/autofocus_thread.py',
        'src/_lumascope/modules/executor_registry.py',
    ),
    {
        'Lumascope',
        'ScopeSession',
        'AutofocusRunner',
        'AutofocusThread',
        'create_default',
    },
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
        with open(path, encoding='utf-8') as fh:
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
                # A member called on the Session itself is the owner doing its
                # own step for the GUI, as it does for a headless caller; only
                # a reach past it to the scope is the GUI orchestrating.
                on_the_session = chain[:1] == ['session']
                through_scope = not on_the_session and (
                    'scope' in chain or 'session' in chain or 'lumaview' in chain
                )
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


# Proxy 4: the lower layer reaching UP into the GUI's context. A `src/_lumascope/modules/`
# file that reads `_app_ctx.ctx` is orchestration code that only works when
# a Kivy app has been built around it -- the mirror of proxy 3, and the
# other half of what a headless caller trips over. Counted per file.


def _modules_context_read_counts():
    """{'src/_lumascope/modules/<file>.py': number of `_app_ctx.ctx` attribute reads}."""
    counts = {}
    for path in _list_py_files('src/_lumascope/modules'):
        with open(path, encoding='utf-8') as fh:
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
    ('ui/microscope_settings.py', 'enable_streaming'): 0,
    ('ui/protocol_settings.py', 'set_labware'): 0,
}

_MODULES_CONTEXT_READ_PIN = {
    'src/_lumascope/modules/executor_registry.py': 0,
    'src/_lumascope/modules/metrics_logger.py': 0,
    'src/_lumascope/modules/scope_session.py': 0,
}


# Proxy 5: the lower layer reading WIDGETS. A `src/_lumascope/modules/` file that subscripts
# `.ids[...]` is answering a question about the instrument by looking at a
# Kivy widget -- the API reading its settings off the GUI. Pin 4 counts the
# context reads that open the door; this counts every widget read behind
# them, alias or not.


def _modules_widget_read_counts():
    """{'src/_lumascope/modules/<file>.py': number of `<expr>.ids[...]` subscripts}."""
    counts = {}
    for path in _list_py_files('src/_lumascope/modules'):
        with open(path, encoding='utf-8') as fh:
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


# Pinned at 74153fad. Lower a value in the same commit that hands the value
# in as an argument or moves the popup to the caller; never raise one.
_MODULES_WIDGET_READ_PIN = {}


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


def _ratchet_report(name, pin, actual, what, remedy=_GUI_REMEDY):
    """The differences between a pin and the tree, each with its remedy.

    The totals are observed first (tests/ratchets.py), so the run's summary
    carries the ratchet whether or not the differences fail the test.
    """
    _ratchets.observe(name, sum(actual.values()), sum(pin.values()), 'equal')
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
            'GUI: questions the GUI answers below the API',
            _UI_ANSWERER_CALL_PIN,
            _ui_answerer_call_counts(),
            'answerer calls',
            _ANSWERER_REMEDY,
        )
        assert report == [], '\n'.join(report)

    def test_ui_private_reaches_match_the_pin(self):
        report = _ratchet_report(
            'GUI: private API reaches from ui/',
            _UI_PRIVATE_REACH_PIN,
            _ui_private_reach_counts(),
            'private reaches',
        )
        assert report == [], '\n'.join(report)

    def test_ui_except_handlers_match_the_pin(self):
        report = _ratchet_report(
            'GUI: except handlers in ui/',
            _UI_EXCEPT_PIN,
            _ui_except_counts(),
            'except handlers',
            _EXCEPT_REMEDY,
        )
        assert report == [], '\n'.join(report)

    def test_gui_orchestration_calls_match_the_pin(self):
        report = _ratchet_report(
            'GUI: orchestration calls from the GUI',
            _GUI_ORCHESTRATION_PIN,
            _gui_orchestration_counts(),
            'orchestration calls',
        )
        assert report == [], '\n'.join(report)

    def test_modules_context_reads_match_the_pin(self):
        report = _ratchet_report(
            'GUI: _app_ctx.ctx reads in src/_lumascope/modules/',
            _MODULES_CONTEXT_READ_PIN,
            _modules_context_read_counts(),
            '_app_ctx.ctx reads',
            _MODULES_REMEDY,
        )
        assert report == [], '\n'.join(report)

    def test_modules_widget_reads_match_the_pin(self):
        report = _ratchet_report(
            'GUI: widget .ids[ reads in src/_lumascope/modules/',
            _MODULES_WIDGET_READ_PIN,
            _modules_widget_read_counts(),
            '.ids[ reads',
            _MODULES_REMEDY,
        )
        assert report == [], '\n'.join(report)


def test_the_modules_guards_read_every_file_at_every_depth(monkeypatch, tmp_path):
    """A context read or a widget read in a subpackage, or in a package's
    __init__.py, is counted: the lower layer reaching up is the same defect
    one folder down."""
    import subprocess

    from tests import ast_seams

    reads = 'def f(_app_ctx):\n    return _app_ctx.ctx.ids["w"]\n'
    (tmp_path / 'src/_lumascope/modules' / 'pkg').mkdir(parents=True)
    (tmp_path / 'src/_lumascope/modules' / 'pkg' / 'x.py').write_text(reads)
    (tmp_path / 'src/_lumascope/modules' / 'pkg' / '__init__.py').write_text(reads)
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    # pointed at the tree as tools/ratchet_history points a sampled one
    monkeypatch.setattr(ast_seams, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(sys.modules[__name__], 'REPO_ROOT', str(tmp_path))

    expected = {'src/_lumascope/modules/pkg/__init__.py': 1, 'src/_lumascope/modules/pkg/x.py': 1}
    assert _modules_context_read_counts() == expected
    assert _modules_widget_read_counts() == expected


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
    _DIRECT = 'def f(ctx):\n    ctx.lumaview.scope.camera._set_pixel_format_impl(fmt)\n'
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
