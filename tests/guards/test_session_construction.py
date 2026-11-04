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
from modules.exceptions import ArgumentRefusedError
from modules.scope_session import ScopeSession
from tests.ast_seams import MemberNames
from tests.scope_fakes import real_executor_bundle, spec_scope
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
# Every scope is built by ScopeSession.create: no other production file
# constructs a Lumascope
# ===========================================================================

# The one file that constructs a scope, and the one exception with its
# retire-by gate (Rule 35 exception 3). The Pylon probe tool does not run
# (its ImagingAPI call predates the frame_validity argument) and is rebuilt
# on ScopeSession.create with #727; its entry goes then.
_SCOPE_BUILDERS = {
    'modules/scope_session.py': 'ScopeSession.create, the one bring-up',
    'tools/pylon_probe_sweep.py': '#727 rebuilds it on ScopeSession.create',
}


def _named(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (
        isinstance(node, ast.Attribute) and node.attr == name
    )


def _lumascope_constructions(tree: ast.AST) -> list[tuple[int, str]]:
    """Every place *tree* builds a Lumascope: ``(line, the form)``.

    The forms: a call naming ``Lumascope``; ``Lumascope.__new__(...)``;
    ``cls(...)`` or ``cls.__new__(...)`` inside ``class Lumascope`` (a
    classmethod constructor); a class deriving from ``Lumascope``. Not
    seen: an alias (``L = Lumascope; L()``), ``type(x)()``, ``copy``,
    ``pickle`` and ``functools.partial``; production uses none.
    """
    found = []

    def visit(node: ast.AST, in_lumascope: bool) -> None:
        if isinstance(node, ast.ClassDef):
            if any(_named(base, 'Lumascope') for base in node.bases):
                found.append((node.lineno, f'class {node.name}(Lumascope)'))
            in_lumascope = in_lumascope or node.name == 'Lumascope'
        if isinstance(node, ast.Call):
            fn = node.func
            is_new = isinstance(fn, ast.Attribute) and fn.attr == '__new__'
            if _named(fn, 'Lumascope'):
                found.append((node.lineno, 'Lumascope('))
            elif is_new and _named(fn.value, 'Lumascope'):
                found.append((node.lineno, 'Lumascope.__new__('))
            elif in_lumascope and _named(fn, 'cls'):
                found.append((node.lineno, 'cls('))
            elif in_lumascope and is_new and _named(fn.value, 'cls'):
                found.append((node.lineno, 'cls.__new__('))
        for child in ast.iter_child_nodes(node):
            visit(child, in_lumascope)

    visit(tree, False)
    return found


class TestOnlyTheSessionBuildsAScope:
    def test_no_production_file_but_the_session_constructs_a_lumascope(self):
        # A second constructor drifts from the first: the support report's
        # hand-wired copy had no camera, no model and no settings.
        built_in = {}
        for rel in ast_seams.tracked_files():
            if rel.startswith('tests/'):
                continue
            found = _lumascope_constructions(ast_seams.parse_module(rel))
            if found:
                built_in[rel] = found
        others = {rel: found for rel, found in built_in.items() if rel not in _SCOPE_BUILDERS}
        assert others == {}, f'a Lumascope is built outside ScopeSession.create: {others}'
        stale = sorted(set(_SCOPE_BUILDERS) - set(built_in))
        assert stale == [], f'these no longer build a Lumascope; drop their entries: {stale}'

    def test_every_form_of_construction_is_seen(self):
        planted = {
            'a call': 'scope = Lumascope(simulate=True)\n',
            'an attribute call': 'scope = _lumascope.Lumascope()\n',
            'a bare instance': 'scope = Lumascope.__new__(Lumascope)\n',
            'a classmethod constructor': (
                'class Lumascope:\n'
                '    @classmethod\n'
                '    def made(cls):\n'
                '        instance = cls.__new__(cls)\n'
                '        return instance\n'
            ),
            'a classmethod calling cls': (
                'class Lumascope:\n    @classmethod\n    def made(cls):\n        return cls()\n'
            ),
            'a subclass': 'class Diagnostic(Lumascope):\n    pass\n',
        }
        unseen = [
            form
            for form, source in planted.items()
            if not _lumascope_constructions(ast.parse(source))
        ]
        assert unseen == []


# ===========================================================================
# create(...) takes the host's injections by name and wires each to its one
# consumer; the ownership facts are constructor state
# ===========================================================================

HOST_INJECTIONS = MemberNames(
    'modules/scope_session.py',
    {
        'simulate',
        'warn_pre_release',
        'engineering_mode',
        'no_engineering',
    },
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            session = ScopeSession.create(
                settings=complete_settings(live_folder=str(tmp_path)),
                simulate=True,
                warn_pre_release=False,
                engineering_mode=True,
                no_engineering=True,
            )
        try:
            assert [w.category for w in caught if w.category is FutureWarning] == [], (
                'warn_pre_release=False gates the factory call AND the constructor call'
            )
            assert session.engineering_mode is True
            assert session.no_engineering is True
            assert session.scope.no_hardware is False, 'simulate=True reached the scope'
            assert session.scope.diagnostics.get_motor_info().model is not None
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_create_takes_no_lanes(self):
        # The scope builds its own; a caller's lane would be a second copy.
        params = inspect.signature(ScopeSession.create).parameters
        assert 'io_executor' not in params and 'camera_executor' not in params
        with pytest.raises(TypeError):
            ScopeSession.create(settings=complete_settings(), io_executor=MagicMock())

    def test_create_takes_no_scope(self):
        # The factory builds the scope, so every session's is brought up
        # the one way; a caller's scope would be a second bring-up. The
        # name is refused whatever it is given.
        with pytest.raises(TypeError):
            ScopeSession.create(settings=complete_settings(), scope=None)

    def test_a_simulated_scope_is_refused_for_a_real_scope_before_anything_is_built(self):
        # One keyword carries every part a simulated scope is built with, so
        # a real scope refuses the record whatever it holds -- an empty one
        # included, since only a simulated session is built from it.
        from modules.lumascope_api.simulation import (
            SimulatedCameraWriteRefusal,
            SimulatedModelAnswer,
            SimulatedScope,
        )

        wrong = {}
        for label, record in (
            ('empty', SimulatedScope()),
            (
                'every part',
                SimulatedScope(
                    model='LS850T',
                    model_answer=SimulatedModelAnswer('LS850T'),
                    refused_stream_starts=1,
                    refused_camera_writes=(SimulatedCameraWriteRefusal('gain', 'false'),),
                    camera_present=False,
                ),
            ),
        ):
            with pytest.raises(ArgumentRefusedError) as refused:
                ScopeSession.create(settings=complete_settings(), simulated=record)
            if (refused.value.reason, refused.value.argument) != (
                'needs_simulated_scope',
                'simulated',
            ):
                wrong[label] = (refused.value.reason, refused.value.argument)
        assert not wrong, wrong

    def test_the_default_still_warns_once(self, tmp_path, fresh_warning_latch):
        # The preserved half: a caller that shipped separately is warned.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            session = ScopeSession.create(
                complete_settings(live_folder=str(tmp_path)), simulate=True
            )
        try:
            assert [w.category for w in caught if w.category is FutureWarning] == [FutureWarning]
        finally:
            session.shutdown()
            session.scope.disconnect()


class TestScopeOwnershipIsConstructorState:
    """Whether shutdown() may tear the scope down is decided where the
    scope came from, at construction: a factory-built scope is the
    session's, one handed to the constructor is the caller's. The bundle
    the session holds is always its own to stop."""

    def _settings(self, tmp_path):
        return complete_settings(live_folder=str(tmp_path))

    def test_a_factory_built_scope_is_owned(self, tmp_path):
        session = ScopeSession.create(
            settings=self._settings(tmp_path), simulate=True, warn_pre_release=False
        )
        try:
            assert session._owns_scope is True
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_a_direct_construction_is_not_owned(self):
        session = ScopeSession(
            settings=complete_settings(), scope=spec_scope(), executor_bundle=real_executor_bundle()
        )
        try:
            assert session._owns_scope is False
            assert session._shut_down is False
        finally:
            session.shutdown()

    def test_a_session_needs_a_bundle(self):
        # FILE, the worker pool and the run threads come from it; a session
        # without one had five branches reading a FILE lane it did not have.
        with pytest.raises(TypeError):
            ScopeSession(settings=complete_settings(), scope=spec_scope())


class TestTheLanesAreTheScopes:
    """The scope builds its IO and CAMERA lanes; the session, its bundle and
    its run engine read them from it and hold no copy."""

    def test_the_sessions_lanes_are_its_scopes(self, tmp_path):
        session = ScopeSession.create(
            settings=complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            warn_pre_release=False,
        )
        try:
            scope = session.scope
            for lanes in (
                (session.io_executor, session.camera_executor),
                (session.executor_bundle.io_executor, session.executor_bundle.camera_executor),
                (
                    session.sequenced_capture_runner._io_executor,
                    session.sequenced_capture_runner.camera_executor,
                ),
            ):
                assert lanes == (scope.io_lane(), scope.camera_lane())
        finally:
            session.shutdown()


# ===========================================================================
# The GUI takes the factory: build() composes nothing itself, the widget
# takes the scope, load_settings renders what the Session already did
# ===========================================================================


def _calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


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

    def test_the_gui_builds_its_context_over_its_session(self):
        """build() takes its context from ``AppContext.over_session`` and hands
        in none of the session's handles itself: the builder reads them off the
        session (tests/test_the_app_context_is_the_sessions.py), and a handle
        handed in beside it would be a second construction."""
        build = ast_seams.find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')
        direct = [
            c.lineno
            for c in _calls(build)
            if isinstance(c.func, ast.Name) and c.func.id == 'AppContext'
        ]
        assert direct == [], f'build() constructs AppContext( directly at {direct}'
        built = [
            c
            for c in _calls(build)
            if isinstance(c.func, ast.Attribute)
            and c.func.attr == 'over_session'
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == 'AppContext'
        ]
        assert len(built) == 1, 'build() builds its context through the one builder'
        assert [ast.unparse(a) for a in built[0].args] == ['scope_session']
        builder = ast_seams.find_def(
            'modules/app_context.py', 'over_session', class_name='AppContext'
        )
        (returned,) = [n for n in ast.walk(builder) if isinstance(n, ast.Return)]
        from_the_session = {kw.arg for kw in returned.value.keywords if kw.arg}
        handed_in = sorted({kw.arg for kw in built[0].keywords} & from_the_session)
        assert handed_in == [], f'build() hands in handles the session owns: {handed_in}'

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
            & {'get_motor_info', 'configure_scope', 'enable_streaming'}
        )
        assert hits == [], f'load_settings still runs bring-up steps: {hits}'
        assert any(
            isinstance(c.func, ast.Attribute) and c.func.attr == 'reconfigure_for_scope'
            for c in _calls(load)
        ), 'load_settings renders the model the Session wrote'


# ===========================================================================
# One factory: create(settings, ..., simulate=...) for every host. Reading
# the user's configuration from disk is its own call, never a default of it
# ===========================================================================
class TestOneSessionFactory:
    def test_there_is_no_second_factory(self):
        # create_headless was create(simulate=True) plus a disk read behind
        # a None default: a second way to build a session, and "headless"
        # read as "simulated". Every host picks hardware with simulate=.
        assert not hasattr(ScopeSession, 'create_headless')

    def test_create_requires_settings(self):
        # A default here would configure a caller that forgot its settings
        # from whatever is on disk instead of refusing the call.
        param = inspect.signature(ScopeSession.create).parameters['settings']
        assert param.default is inspect.Parameter.empty

    def test_load_user_settings_refuses_a_directory_that_is_not_an_installation(
        self, monkeypatch, tmp_path
    ):
        import modules.settings_init as settings_init
        from modules.exceptions import ArgumentRefusedError

        # No settings loaded in this process: the resolver, not the store,
        # answers.
        monkeypatch.setattr(settings_init, 'settings', None)

        with pytest.raises(ArgumentRefusedError) as refused:
            ScopeSession.load_user_settings(str(tmp_path))
        assert (refused.value.reason, refused.value.argument) == (
            'not_an_installation',
            'source_path',
        )
