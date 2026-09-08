# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session composes and decomposes the instrument; no host keeps a copy.

AST pins over ``tests.ast_seams`` (never a direct source read: the
source-pin ratchet counts those). Each pin names one copy of the
bring-up or the teardown that used to live in a host and the reason it
must not come back.
"""

import ast
import inspect
import warnings
from unittest.mock import MagicMock

import pytest

import tests.ast_seams as ast_seams
from modules.scope_session import ScopeSession
from tests.scope_fakes import spec_scope
from tests.settings_fixtures import complete_settings


class TestNoHostCopyOfTheBringUp:
    def test_microscope_settings_has_no_reconnect_handler(self):
        # The handler was an unbound 131-line copy of the settings-to-scope
        # bring-up: it took the registry's unvalidated explicit-name path
        # and stopped the display thread last, after the scope swap. The
        # scope-swap seams it called (set_scope, rebind) stay for the
        # auto-reconnect item; a new handler would be a second bring-up.
        node = ast_seams.find_def(
            'ui/microscope_settings.py', 'reconnect', class_name='MicroscopeSettings'
        )
        assert node is None, f'MicroscopeSettings.reconnect is back at line {node.lineno}'


# ===========================================================================
# create(...) takes the host's injections by name and wires each to its one
# consumer; the ownership facts are constructor state
# ===========================================================================

HOST_INJECTIONS = (
    'simulate',
    'warn_pre_release',
    'ui_dispatcher',
    'af_ui_update_func',
    'settings_saved_hook',
    'engineering_mode',
    'display_ctx_provider',
)


@pytest.fixture
def fresh_warning_latch():
    # The pre-release warning fires once per process; each test here needs
    # a not-yet-fired latch and must not poison the tests that follow.
    import modules.lumascope_api._lumascope as lm_mod

    previous = lm_mod._PRE_RELEASE_WARNING_FIRED
    lm_mod._PRE_RELEASE_WARNING_FIRED = False
    yield
    lm_mod._PRE_RELEASE_WARNING_FIRED = previous


class TestCreateTakesTheHostInjections:
    def test_the_injections_are_keyword_only(self):
        params = inspect.signature(ScopeSession.create).parameters
        missing = [n for n in HOST_INJECTIONS if n not in params]
        assert missing == [], f'create() lacks host injections {missing}'
        not_kw_only = [
            n for n in HOST_INJECTIONS if params[n].kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        assert not_kw_only == [], 'a host injection is positional; each is named at the call'

    def test_each_injection_lands_on_its_consumer(self, tmp_path, fresh_warning_latch):
        af_ui = MagicMock(name='af_ui')
        dispatcher = MagicMock(name='ui_dispatcher')
        hook = MagicMock(name='settings_saved_hook')
        provider = MagicMock(name='display_ctx_provider', return_value=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            session = ScopeSession.create(
                settings=complete_settings(live_folder=str(tmp_path)),
                simulate=True,
                warn_pre_release=False,
                ui_dispatcher=dispatcher,
                af_ui_update_func=af_ui,
                settings_saved_hook=hook,
                engineering_mode=True,
                display_ctx_provider=provider,
            )
        try:
            assert [w.category for w in caught if w.category is FutureWarning] == [], (
                'warn_pre_release=False gates the factory call AND the constructor call'
            )
            # One callable, two consumers: the AF runner's Z readout and the
            # capture engine's.
            assert session.autofocus_runner.ui_update_func is af_ui
            assert session.sequenced_capture_runner._z_ui_update_func is af_ui
            bundle = session.executor_bundle
            for lane in (
                bundle.io_executor,
                bundle.camera_executor,
                bundle.file_io_executor,
                bundle.worker_pool,
            ):
                assert lane._ui_dispatch is dispatcher, f'{lane.executor_name} marshals elsewhere'
            assert bundle.scope_display_thread._ctx_provider is provider
            assert session._settings_saved_hook is hook
            assert session.engineering_mode is True
            assert session.scope.no_hardware is False, 'simulate=True reached the scope'
            assert session.scope.diagnostics.get_microscope_model() is not None
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_a_lane_the_caller_did_not_pass_takes_the_dispatcher(self, tmp_path):
        from modules.sequential_io_executor import SequentialIOExecutor

        dispatcher = MagicMock(name='ui_dispatcher')
        cam = SequentialIOExecutor(name='CAMERA_CONSTRUCTION_TEST')
        cam.start()
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            camera_executor=cam,
            simulate=True,
            warn_pre_release=False,
            ui_dispatcher=dispatcher,
        )
        try:
            assert session.executor_bundle is None
            assert session.io_executor._ui_dispatch is dispatcher
            assert session.io_executor.worker_alive is False, (
                'the factory never starts a lane it built beside a caller lane'
            )
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_the_default_still_warns_once(self, tmp_path, fresh_warning_latch):
        # The preserved half: a caller that shipped separately is warned.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            session = ScopeSession.create_headless(
                settings=complete_settings(live_folder=str(tmp_path))
            )
        try:
            assert [w.category for w in caught if w.category is FutureWarning] == [FutureWarning]
        finally:
            session.shutdown()
            session.scope.disconnect()


class TestScopeOwnershipIsConstructorState:
    """Whether shutdown() may tear the scope down is decided where the
    scope came from, at construction: a factory-built scope is the
    session's, a passed or directly constructed one is the caller's."""

    def _settings(self, tmp_path):
        return complete_settings(live_folder=str(tmp_path))

    def test_a_factory_built_scope_over_the_bundle_is_owned(self, tmp_path):
        session = ScopeSession.create(
            settings=self._settings(tmp_path), simulate=True, warn_pre_release=False
        )
        try:
            assert session._owns_scope is True
            assert session._owns_executors is True
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_create_headless_is_the_same_row(self, tmp_path):
        session = ScopeSession.create_headless(settings=self._settings(tmp_path))
        try:
            assert session._owns_scope is True
            assert session._owns_executors is True
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_a_factory_built_scope_over_caller_lanes_is_owned(self, tmp_path):
        from modules.sequential_io_executor import SequentialIOExecutor

        io, cam = (
            SequentialIOExecutor(name='IO_OWN_TEST'),
            SequentialIOExecutor(name='CAM_OWN_TEST'),
        )
        io.start()
        cam.start()
        session = ScopeSession.create(
            settings=self._settings(tmp_path),
            io_executor=io,
            camera_executor=cam,
            simulate=True,
            warn_pre_release=False,
        )
        try:
            assert session._owns_scope is True
            assert session._owns_executors is False
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_a_passed_scope_is_not_owned(self):
        session = ScopeSession.create(
            settings=complete_settings(),
            scope=spec_scope(),
            io_executor=MagicMock(),
            camera_executor=MagicMock(),
        )
        try:
            assert session._owns_scope is False
            assert session._owns_executors is False
        finally:
            session.shutdown()

    def test_a_passed_scope_over_the_bundle_is_not_owned(self):
        session = ScopeSession.create(settings=complete_settings(), scope=spec_scope())
        try:
            assert session._owns_scope is False
            assert session._owns_executors is True
        finally:
            session.shutdown()

    def test_a_direct_construction_is_not_owned(self):
        session = ScopeSession(
            settings=complete_settings(),
            scope=spec_scope(),
            io_executor=MagicMock(),
            camera_executor=MagicMock(),
        )
        try:
            assert session._owns_scope is False
            assert session._owns_executors is False
            assert session._shut_down is False
        finally:
            session.shutdown()


# ===========================================================================
# The GUI takes the factory: build() composes nothing itself, the widget
# takes the scope, load_settings renders what the Session already did
# ===========================================================================


def _calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _chain(node):
    out = []
    value = node
    while isinstance(value, ast.Attribute):
        out.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        out.append(value.id)
    return out


class TestTheGuiTakesTheFactory:
    def test_lumaviewpro_constructs_no_session_directly(self):
        # The ratchet cannot see this: a call through the factory is an
        # attribute call, so the scanner's constructor count reads zero
        # either way. This is the exit criterion for the direct call.
        build = ast_seams.find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')
        assert build is not None
        direct = [
            c.lineno
            for c in _calls(build)
            if isinstance(c.func, ast.Name) and c.func.id == 'ScopeSession'
        ]
        assert direct == [], f'build() still constructs ScopeSession( directly at {direct}'
        factory = [
            c
            for c in _calls(build)
            if isinstance(c.func, ast.Attribute)
            and c.func.attr == 'create'
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == 'ScopeSession'
        ]
        assert len(factory) == 1, 'build() composes the session through the one factory'
        passed = {kw.arg for kw in factory[0].keywords}
        assert set(HOST_INJECTIONS) <= passed, (
            f'the host passes every injection by name; missing {set(HOST_INJECTIONS) - passed}'
        )

    def test_no_ui_module_constructs_a_lumascope(self):
        hits = []
        for rel, tree in ast_seams.iter_package_modules(('ui',)):
            for c in _calls(tree):
                fn = c.func
                if (isinstance(fn, ast.Name) and fn.id == 'Lumascope') or (
                    isinstance(fn, ast.Attribute) and fn.attr == 'Lumascope'
                ):
                    hits.append((rel, c.lineno))
        assert hits == [], f'Lumascope( constructed under ui/ at {hits}'

    def test_the_executor_registry_reaches_no_app_context(self):
        tree = ast_seams.parse_module('modules/executor_registry.py')
        hits = [
            n.lineno
            for n in ast.walk(tree)
            if (isinstance(n, ast.Import) and any(a.name == 'modules.app_context' for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module == 'modules.app_context')
        ]
        assert hits == [], (
            f'executor_registry imports modules.app_context at {hits}; the provider is handed in'
        )

    def test_main_display_takes_the_scope_before_its_own_init(self):
        init = ast_seams.find_def('ui/main_display.py', '__init__', class_name='MainDisplay')
        assert init is not None
        names = [a.arg for a in init.args.args]
        assert 'scope' in names and 'camera_type' not in names and 'simulate' not in names, names
        assign = next(
            (
                i
                for i, stmt in enumerate(init.body)
                if isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'scope' for t in stmt.targets)
                and isinstance(stmt.value, ast.Name)
                and stmt.value.id == 'scope'
            ),
            None,
        )
        super_init = next(
            (
                i
                for i, stmt in enumerate(init.body)
                if any(
                    isinstance(c.func, ast.Attribute)
                    and c.func.attr == '__init__'
                    and isinstance(c.func.value, ast.Call)
                    and isinstance(c.func.value.func, ast.Name)
                    and c.func.value.func.id == 'super'
                    for c in _calls(stmt)
                )
            ),
            None,
        )
        assert assign is not None, 'no `self.scope = scope` in MainDisplay.__init__'
        assert super_init is not None and assign < super_init, (
            'the scope goes on before the kv tree is built'
        )

    def test_the_context_handles_are_the_sessions_objects(self):
        build = ast_seams.find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')
        ctx_calls = [
            c for c in _calls(build) if isinstance(c.func, ast.Name) and c.func.id == 'AppContext'
        ]
        assert len(ctx_calls) == 1
        composed = {
            'scope',
            'io_executor',
            'camera_executor',
            'file_io_executor',
            'worker_pool',
            'protocol_thread',
            'scope_display_thread',
            'autofocus_thread',
            'autofocus_runner',
            'sequenced_capture_runner',
            'wellplate_loader',
            'coordinate_transformer',
            'objective_helper',
        }
        wrong = [
            (kw.arg, ast.unparse(kw.value))
            for kw in ctx_calls[0].keywords
            if kw.arg in composed and 'scope_session' not in _chain(kw.value)
        ]
        assert wrong == [], f'context handles not read off the session: {wrong}'
        seen = {kw.arg for kw in ctx_calls[0].keywords}
        assert composed <= seen, f'the context lost a handle: {composed - seen}'

    def test_the_context_has_no_field_only_the_dead_handler_read(self):
        from modules.app_context import AppContext

        fields = set(AppContext.__dataclass_fields__)
        assert 'simulate_mode' not in fields and 'disable_homing' not in fields

    def test_load_settings_renders_and_no_longer_brings_up(self):
        load = ast_seams.find_def(
            'ui/microscope_settings.py', 'load_settings', class_name='MicroscopeSettings'
        )
        assert load is not None
        hits = sorted(
            {c.func.attr for c in _calls(load) if isinstance(c.func, ast.Attribute)}
            & {'get_microscope_model', 'configure_scope', 'start_streaming'}
        )
        assert hits == [], f'load_settings still runs bring-up steps: {hits}'
        assert any(
            isinstance(c.func, ast.Attribute) and c.func.attr == 'reconfigure_for_scope'
            for c in _calls(load)
        ), 'load_settings renders the model the Session wrote'
