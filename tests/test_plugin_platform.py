# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the plugin platform itself (modules/plugins/__init__.py).

Covers:
    - PluginSpec construction + immutability
    - UIRegistry: mount-point validation, duplicate-name rejection
    - PostProcessingRegistry: processor registration + lookup
    - LiveProcessingRegistry: forwards to the scope's frame listeners
    - RESTRegistry: stub raises with clear message
    - PluginRegistry.all_health(): per-namespace snapshot shape
    - is_version_compatible: semver compare with pre-release suffixes
    - PluginRegistry.load / unload: discovery + lifecycle (mocked entry points)
"""

from __future__ import annotations

import re
import types
from unittest.mock import MagicMock, patch

import pytest

from modules.exceptions import PluginNotLoadedError, RefusalCause
from modules.notification_center import OutcomeKind, outcome_of
from modules.plugins import (
    PluginRegistrationError,
    PluginSpec,
    ProcessorResult,
    UI_MOUNT_POINTS,
    is_version_compatible,
)
from tests.plugin_test_harness import harness_ctx  # noqa: F401


# ---------------------------------------------------------------------------
# PluginSpec
# ---------------------------------------------------------------------------


def _make_spec(name: str = 'demo', version: str = '0.1.0', requires: str = '>=4.0.0') -> PluginSpec:
    return PluginSpec(
        name=name,
        version=version,
        requires_lvp_version=requires,
        description='Demo plugin',
        capabilities=('scope.imaging',),
        author='test',
    )


def test_plugin_spec_is_frozen():
    spec = _make_spec()
    with pytest.raises((AttributeError, Exception)):
        spec.name = 'modified'


def test_plugin_spec_defaults():
    spec = PluginSpec(
        name='x',
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description='d',
    )
    assert spec.capabilities == ()
    assert spec.subscribes_to == ()
    assert spec.author == ''
    assert spec.url == ''
    assert spec.auto_run_on_protocol_complete is False


def test_plugin_spec_auto_run_opt_in():
    spec = PluginSpec(
        name='x',
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description='d',
        auto_run_on_protocol_complete=True,
    )
    assert spec.auto_run_on_protocol_complete is True


# ---------------------------------------------------------------------------
# UIRegistry
# ---------------------------------------------------------------------------


def test_ui_register_known_mount(harness_ctx):
    spec = _make_spec(name='ui_demo')
    builder = lambda: MagicMock(name='widget')
    harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', builder)
    mounts = harness_ctx.plugins.ui.mounts()
    assert len(mounts) == 1
    name, mp, b = mounts[0]
    assert name == 'ui_demo'
    assert mp == 'left_sidebar.accordion'
    assert b is builder


def test_ui_register_unknown_mount_raises(harness_ctx):
    spec = _make_spec(name='ui_demo')
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.ui.register(
            spec,
            'nonexistent.mount',
            lambda: None,
        )
    assert 'nonexistent.mount' in str(exc.value)
    assert 'left_sidebar.accordion' in str(exc.value)
    # The plugin asked for something this LumaViewPro does not have; asking
    # again unchanged cannot succeed.
    outcome = outcome_of(exc.value)
    assert (outcome.kind, outcome.cause, outcome.reason) == (
        OutcomeKind.REFUSAL,
        RefusalCause.REQUEST,
        'unknown_mount_point',
    )


def test_ui_register_duplicate_name_raises(harness_ctx):
    spec = _make_spec(name='dup')
    harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: None)
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: None)
    assert "'dup'" in str(exc.value)
    assert "'ui'" in str(exc.value)
    outcome = outcome_of(exc.value)
    assert (outcome.kind, outcome.cause, outcome.reason) == (
        OutcomeKind.REFUSAL,
        RefusalCause.REQUEST,
        'name_taken',
    )


def test_ui_health_after_register(harness_ctx):
    spec = _make_spec(name='healthy')
    harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: None)
    health = harness_ctx.plugins.ui.health()
    assert health.namespace == 'ui'
    assert len(health.loaded) == 1
    assert health.loaded[0].name == 'healthy'
    assert harness_ctx.plugins.not_loaded() == ()


def test_ui_mount_points_constant_is_frozen():
    assert isinstance(UI_MOUNT_POINTS, frozenset)
    assert 'left_sidebar.accordion' in UI_MOUNT_POINTS


# ---------------------------------------------------------------------------
# PostProcessingRegistry
# ---------------------------------------------------------------------------


def _trivial_processor(input_dir: str, manifest: dict, output_dir: str):
    return ProcessorResult(success=True, outputs=(), message='ok')


def test_post_processing_register_and_get(harness_ctx):
    spec = _make_spec(name='pp_demo')
    harness_ctx.plugins.post_processing.register(spec, _trivial_processor)
    fetched = harness_ctx.plugins.post_processing.get('pp_demo')
    assert fetched is _trivial_processor
    assert harness_ctx.plugins.post_processing.names() == ('pp_demo',)


def test_post_processing_duplicate_name_raises(harness_ctx):
    spec = _make_spec(name='dup_pp')
    harness_ctx.plugins.post_processing.register(spec, _trivial_processor)
    with pytest.raises(PluginRegistrationError):
        harness_ctx.plugins.post_processing.register(spec, _trivial_processor)


def test_post_processing_get_missing_returns_none(harness_ctx):
    assert harness_ctx.plugins.post_processing.get('not_registered') is None


def test_post_processing_handlers_returns_spec_processor_pairs(harness_ctx):
    spec_a = _make_spec(name='pp_a')
    spec_b = _make_spec(name='pp_b')
    harness_ctx.plugins.post_processing.register(spec_a, _trivial_processor)
    harness_ctx.plugins.post_processing.register(spec_b, _trivial_processor)
    handlers = harness_ctx.plugins.post_processing.handlers()
    assert len(handlers) == 2
    names = {h[0].name for h in handlers}
    assert names == {'pp_a', 'pp_b'}
    for spec, processor in handlers:
        assert isinstance(spec, PluginSpec)
        assert callable(processor)


# ---------------------------------------------------------------------------
# run_protocol_complete_processors() dispatcher
# ---------------------------------------------------------------------------


def test_auto_run_dispatcher_invokes_only_opted_in(harness_ctx):
    calls = []

    def opted_in(input_dir, manifest, output_dir):
        calls.append(('opted_in', input_dir, manifest, output_dir))
        return ProcessorResult(success=True, message='did the work')

    def not_opted_in(input_dir, manifest, output_dir):
        calls.append(('not_opted_in',))
        return ProcessorResult(success=True, message='should not run')

    spec_opt = _make_spec(name='opted')
    # Re-create with auto_run set since _make_spec doesn't take it
    spec_opt = PluginSpec(
        name='opted',
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description='d',
        auto_run_on_protocol_complete=True,
    )
    spec_skip = _make_spec(name='skipped')

    harness_ctx.plugins.post_processing.register(spec_opt, opted_in)
    harness_ctx.plugins.post_processing.register(spec_skip, not_opted_in)
    # Loaded, as the session's load records a plugin: only a loaded
    # plugin's processor is handed the folder.
    for name in ('opted', 'skipped'):
        harness_ctx.plugins._track(name, types.ModuleType(name))

    harness_ctx.plugins.run_protocol_complete_processors(
        input_dir='/in',
        manifest={'k': 'v'},
        output_dir='/out',
        files='written',
    )

    assert len(calls) == 1
    assert calls[0] == ('opted_in', '/in', {'k': 'v'}, '/out')


def test_auto_run_dispatcher_swallows_processor_exception(harness_ctx, caplog):
    calls = []

    def boom(input_dir, manifest, output_dir):
        raise RuntimeError('processor exploded')

    def runs_after(input_dir, manifest, output_dir):
        calls.append('runs_after')
        return ProcessorResult(success=True, message='ok')

    spec_a = PluginSpec(
        name='boomer',
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description='d',
        auto_run_on_protocol_complete=True,
    )
    spec_b = PluginSpec(
        name='good_citizen',
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description='d',
        auto_run_on_protocol_complete=True,
    )
    harness_ctx.plugins.post_processing.register(spec_a, boom)
    harness_ctx.plugins.post_processing.register(spec_b, runs_after)
    for name in ('boomer', 'good_citizen'):
        harness_ctx.plugins._track(name, types.ModuleType(name))

    # Should not raise; runs_after should still execute despite boomer failing.
    harness_ctx.plugins.run_protocol_complete_processors(
        input_dir='/in',
        manifest={},
        output_dir='/out',
        files='written',
    )

    assert calls == ['runs_after']


def test_auto_run_dispatcher_skips_when_all_opt_out(harness_ctx):
    calls = []

    def proc(input_dir, manifest, output_dir):
        calls.append('ran')
        return ProcessorResult(success=True, message='ok')

    spec = _make_spec(name='not_opted')  # default False
    harness_ctx.plugins.post_processing.register(spec, proc)
    harness_ctx.plugins.run_protocol_complete_processors(
        input_dir='/in',
        manifest={},
        output_dir='/out',
        files='written',
    )
    assert calls == []


# ---------------------------------------------------------------------------
# LiveProcessingRegistry + RESTRegistry stubs
# ---------------------------------------------------------------------------


def test_live_processing_register_forwards_to_imaging(harness_ctx):
    """register(spec, handler) forwards to scope.imaging.add_frame_listener
    with the plugin name attached."""
    spec = _make_spec(name='live_demo')

    def handler(frame):
        return None

    harness_ctx.plugins.live_processing.register(spec, handler)
    harness_ctx.scope.imaging.add_frame_listener.assert_called_once_with(handler, name='live_demo')
    assert harness_ctx.plugins.live_processing.names() == ('live_demo',)


def test_live_processing_unregister_forwards_to_imaging(harness_ctx):
    """unregister(name) forwards to scope.imaging.remove_frame_listener
    with the original handler."""
    spec = _make_spec(name='live_demo')

    def handler(frame):
        return None

    harness_ctx.plugins.live_processing.register(spec, handler)
    harness_ctx.plugins.live_processing.unregister('live_demo')
    harness_ctx.scope.imaging.remove_frame_listener.assert_called_once_with(handler)
    assert harness_ctx.plugins.live_processing.names() == ()


def test_live_processing_unregister_unknown_name_is_noop(harness_ctx):
    """unregister(name) for an un-registered plugin is a silent no-op."""
    harness_ctx.plugins.live_processing.unregister('not_registered')
    harness_ctx.scope.imaging.remove_frame_listener.assert_not_called()


def test_live_processing_register_duplicate_name_raises(harness_ctx):
    """Re-registering with the same plugin name raises (same shape as
    UI / post_processing registries)."""
    spec = _make_spec(name='live_dup')
    harness_ctx.plugins.live_processing.register(spec, lambda i, t, c: None)
    with pytest.raises(PluginRegistrationError):
        harness_ctx.plugins.live_processing.register(spec, lambda i, t, c: None)


def test_rest_register_is_refused_with_what_the_author_can_do(harness_ctx):
    """The words name the stub-mode opt-in, the one thing a plugin author
    can do, and point at no document LVP does not ship."""
    spec = _make_spec(name='rest_attempt')
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.rest.register(spec, MagicMock(name='router'))
    msg = str(exc.value)
    assert 'enable_stub_mode' in msg
    assert re.search(r'\.md\b', msg) is None, msg
    outcome = outcome_of(exc.value)
    assert (outcome.kind, outcome.cause, outcome.reason) == (
        OutcomeKind.REFUSAL,
        RefusalCause.REQUEST,
        'rest_namespace_reserved',
    )


# ---------------------------------------------------------------------------
# RESTRegistry stub mode
# ---------------------------------------------------------------------------


def test_rest_stub_mode_accepts_registration(harness_ctx):
    """enable_stub_mode() flips register() from raising to recording."""
    harness_ctx.plugins.rest.enable_stub_mode()
    spec = _make_spec(name='stubbed')
    router = MagicMock(name='router')
    # Should not raise.
    harness_ctx.plugins.rest.register(spec, router)
    # Health reports the registration.
    health = harness_ctx.plugins.rest.health()
    assert len(health.loaded) == 1
    assert health.loaded[0].name == 'stubbed'
    assert health.loaded[0].namespace == 'rest'


def test_rest_stub_mode_does_not_mount_route(harness_ctx):
    """Stub-mode registration records the spec but does not mount a route.

    The router object is stored for introspection but never called/mounted.
    """
    harness_ctx.plugins.rest.enable_stub_mode()
    spec = _make_spec(name='stubbed_no_mount')
    router = MagicMock(name='router')
    harness_ctx.plugins.rest.register(spec, router)
    # Router is never invoked: no .include_router / .add_api_route / etc.
    router.include_router.assert_not_called()
    router.add_api_route.assert_not_called()
    # But it is stored, for tests/introspection.
    assert 'stubbed_no_mount' in harness_ctx.plugins.rest._stubbed_routers


def test_rest_stub_mode_duplicate_registration_raises(harness_ctx):
    """Stub-mode honors the _assert_unique contract from _BaseNamespace."""
    harness_ctx.plugins.rest.enable_stub_mode()
    spec = _make_spec(name='dupe')
    harness_ctx.plugins.rest.register(spec, MagicMock(name='router1'))
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.rest.register(spec, MagicMock(name='router2'))
    assert 'already registered' in str(exc.value)


def test_rest_stub_mode_disabled_restores_fail_loud(harness_ctx):
    """disable_stub_mode() reverts to the default raise behavior;
    previously-stubbed registrations remain visible in health()."""
    harness_ctx.plugins.rest.enable_stub_mode()
    spec_a = _make_spec(name='stub_a')
    harness_ctx.plugins.rest.register(spec_a, MagicMock(name='router_a'))

    harness_ctx.plugins.rest.disable_stub_mode()

    spec_b = _make_spec(name='post_disable')
    with pytest.raises(PluginRegistrationError):
        harness_ctx.plugins.rest.register(spec_b, MagicMock(name='router_b'))

    # The stubbed registration from before disable is still visible.
    health = harness_ctx.plugins.rest.health()
    names = [s.name for s in health.loaded]
    assert 'stub_a' in names
    assert 'post_disable' not in names


def test_rest_stub_mode_default_off(harness_ctx):
    """Stub mode is opt-in; default behavior is unchanged."""
    spec = _make_spec(name='default_off')
    with pytest.raises(PluginRegistrationError):
        harness_ctx.plugins.rest.register(spec, MagicMock(name='router'))


# ---------------------------------------------------------------------------
# PluginRegistry.all_health
# ---------------------------------------------------------------------------


def test_all_health_returns_four_namespaces(harness_ctx):
    h = harness_ctx.plugins.all_health()
    namespaces = [n.namespace for n in h]
    assert sorted(namespaces) == ['live_processing', 'post_processing', 'rest', 'ui']


def test_all_health_initial_state_empty(harness_ctx):
    for ns in harness_ctx.plugins.all_health():
        assert ns.loaded == ()
        assert ns.last_runtime_errors == ()
    assert harness_ctx.plugins.not_loaded() == ()


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'requires,host,expected',
    [
        ('>=4.0.0', '4.0.0', True),
        ('>=4.0.0', '4.0.0-beta8', True),
        ('>=4.0.0', '3.9.9', False),
        ('>=4.1.0', '4.0.0', False),
        ('==4.0.0', '4.0.0', True),
        ('==4.0.0', '4.0.1', False),
        ('<5.0.0', '4.5.0', True),
        ('<5.0.0', '5.0.0', False),
        ('~=4.0.0', '4.5.0', True),
        ('~=4.0.0', '5.0.0', False),
        ('garbage', '4.0.0', False),
        ('>=4.0.0', 'garbage', False),
        ('>=4.0.0', '', False),
    ],
)
def test_is_version_compatible(requires, host, expected):
    assert is_version_compatible(requires, host) is expected


# ---------------------------------------------------------------------------
# load_plugins / unload_plugins
# ---------------------------------------------------------------------------


def _make_plugin_module(
    name: str,
    version: str = '0.1.0',
    requires: str = '>=4.0.0',
    register_raises: bool = False,
    unregister_raises: bool = False,
):
    """Build an in-memory module that looks like a plugin to load_plugins."""
    mod = types.ModuleType(f'fake_plugin_{name}')
    mod.__version__ = version
    mod.spec = PluginSpec(
        name=name,
        version=version,
        requires_lvp_version=requires,
        description=f'fake {name}',
    )
    mod._register_calls = []
    mod._unregister_calls = []

    def register(ctx):
        mod._register_calls.append(ctx)
        if register_raises:
            raise RuntimeError(f'{name} register intentionally failed')
        ctx.plugins.post_processing.register(mod.spec, _trivial_processor)

    def unregister(ctx):
        mod._unregister_calls.append(ctx)
        if unregister_raises:
            raise RuntimeError(f'{name} unregister intentionally failed')

    mod.register = register
    mod.unregister = unregister
    return mod


class _FakeEntryPoint:
    def __init__(self, name: str, module):
        self.name = name
        self._module = module

    def load(self):
        return self._module


def test_load_plugins_with_no_entry_points_loads_only_the_built_ins(harness_ctx):
    with patch('importlib.metadata.entry_points', return_value=[]):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert harness_ctx.plugins.post_processing.names() == ('stitcher',)


def test_load_plugins_registers_a_valid_plugin(harness_ctx):
    mod = _make_plugin_module('valid_pp')
    eps = [_FakeEntryPoint('valid_pp', mod)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert len(mod._register_calls) == 1
    assert mod._register_calls[0] is harness_ctx
    assert 'valid_pp' in harness_ctx.plugins.post_processing.names()


def test_load_plugins_skips_version_incompatible(harness_ctx):
    mod = _make_plugin_module('too_new', requires='>=5.0.0')
    eps = [_FakeEntryPoint('too_new', mod)]
    harness_ctx.version = '4.0.0'
    with patch('importlib.metadata.entry_points', return_value=eps):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert mod._register_calls == []
    assert 'too_new' in [s.name for s in harness_ctx.plugins.not_loaded()]


@pytest.mark.parametrize(
    ('version', 'reason'),
    [
        ('1.0.5', 'version 1.0.5 is older than the 1.0.6 this LumaViewPro needs'),
        ('bogus', "its version 'bogus' cannot be read, and this LumaViewPro needs 1.0.6 or later"),
    ],
)
def test_load_plugins_refuses_a_plugin_older_than_the_hosts_minimum(harness_ctx, version, reason):
    mod = _make_plugin_module('etaluma_engineering', version=version)
    eps = [_FakeEntryPoint('etaluma_engineering', mod)]
    with (
        patch.dict('modules.plugins.MINIMUM_PLUGIN_VERSIONS', {'etaluma_engineering': '1.0.6'}),
        patch('importlib.metadata.entry_points', return_value=eps),
        patch('modules.plugins.notifications.report_outcome') as report,
    ):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert mod._register_calls == []
    assert 'etaluma_engineering' not in harness_ctx.plugins.post_processing.names()
    report.assert_called_once()
    outcome = report.call_args.args[0]
    assert isinstance(outcome, PluginNotLoadedError)
    assert outcome.reason == 'plugin_not_loaded'
    assert [(f.name, f.reason) for f in outcome.failures] == [('etaluma_engineering', reason)]


@pytest.mark.parametrize('version', ['1.0.6', '1.1.0'])
def test_load_plugins_loads_a_plugin_at_or_above_the_hosts_minimum(harness_ctx, version):
    mod = _make_plugin_module('etaluma_engineering', version=version)
    eps = [_FakeEntryPoint('etaluma_engineering', mod)]
    with (
        patch.dict('modules.plugins.MINIMUM_PLUGIN_VERSIONS', {'etaluma_engineering': '1.0.6'}),
        patch('importlib.metadata.entry_points', return_value=eps),
    ):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert len(mod._register_calls) == 1
    assert 'etaluma_engineering' in harness_ctx.plugins.post_processing.names()


def test_load_plugins_continues_past_register_failure(harness_ctx):
    bad = _make_plugin_module('bad', register_raises=True)
    good = _make_plugin_module('good')
    eps = [_FakeEntryPoint('bad', bad), _FakeEntryPoint('good', good)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert 'good' in harness_ctx.plugins.post_processing.names()
    assert 'bad' not in harness_ctx.plugins.post_processing.names()
    # Failed plugin's unregister was called for cleanup attempt
    assert len(bad._unregister_calls) == 1


def test_load_plugins_skips_module_without_spec(harness_ctx):
    mod = types.ModuleType('no_spec_plugin')
    mod.__version__ = '0.1.0'
    mod.register = lambda ctx: None
    # No mod.spec attribute.
    eps = [_FakeEntryPoint('no_spec_plugin', mod)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    assert harness_ctx.plugins.post_processing.names() == ('stitcher',)


def test_unload_plugins_calls_unregister_in_reverse_order(harness_ctx):
    call_order = []

    def make_tracking(name):
        mod = types.ModuleType(f'track_{name}')
        mod.__version__ = '0.1.0'
        mod.spec = PluginSpec(
            name=name,
            version='0.1.0',
            requires_lvp_version='>=4.0.0',
            description='tracking',
        )
        mod.register = lambda ctx: None
        mod.unregister = lambda ctx: call_order.append(name)
        return mod

    a, b, c = make_tracking('a'), make_tracking('b'), make_tracking('c')
    eps = [_FakeEntryPoint('a', a), _FakeEntryPoint('b', b), _FakeEntryPoint('c', c)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    harness_ctx.plugins.unload(harness_ctx)
    assert call_order == ['c', 'b', 'a']


def test_unload_plugins_swallows_unregister_failure(harness_ctx):
    bad = _make_plugin_module('bad_unload', unregister_raises=True)
    good = _make_plugin_module('good_unload')
    eps = [_FakeEntryPoint('bad', bad), _FakeEntryPoint('good', good)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        harness_ctx.plugins.load(harness_ctx, '4.0.0')
    # Should not raise even though bad.unregister will throw.
    harness_ctx.plugins.unload(harness_ctx)
    assert len(bad._unregister_calls) == 1
    assert len(good._unregister_calls) == 1


class TestAttributeException:
    """attribute_exception names the plugin whose code appears in a
    traceback, so the app-level crash guard can contain plugin
    exceptions (popup + log, app continues) while non-plugin crashes
    keep their normal loud failure path. One bad plugin button handler
    must never take down the host.
    """

    def _make_plugin_module(self, tmp_path, pkg_name):
        import sys
        import textwrap

        pkg_dir = tmp_path / pkg_name
        pkg_dir.mkdir()
        (pkg_dir / '__init__.py').write_text(
            textwrap.dedent("""
            def boom():
                raise RuntimeError('plugin blew up')
            """)
        )
        sys.path.insert(0, str(tmp_path))
        try:
            import importlib

            return importlib.import_module(pkg_name)
        finally:
            sys.path.remove(str(tmp_path))

    def test_names_plugin_for_plugin_frame(self, tmp_path):
        import sys

        from modules.plugins import PluginRegistry

        module = self._make_plugin_module(tmp_path, 'fake_bench_plugin')
        registry = PluginRegistry(object())  # a stand-in by design: no scope member is reached
        registry._track('fake-bench-plugin', module)

        try:
            module.boom()
        except RuntimeError:
            tb = sys.exc_info()[2]
            assert registry.attribute_exception(tb) == 'fake-bench-plugin'

    def test_none_for_non_plugin_frame(self):
        import sys

        from modules.plugins import PluginRegistry

        registry = PluginRegistry(object())  # a stand-in by design: no scope member is reached
        try:
            raise RuntimeError('host code failure')
        except RuntimeError:
            tb = sys.exc_info()[2]
            assert registry.attribute_exception(tb) is None

    def test_none_when_no_plugins_loaded(self, tmp_path):
        import sys

        from modules.plugins import PluginRegistry

        module = self._make_plugin_module(tmp_path, 'fake_unloaded_plugin')
        # nothing tracked; a stand-in scope by design: no scope member is reached
        registry = PluginRegistry(object())
        try:
            module.boom()
        except RuntimeError:
            tb = sys.exc_info()[2]
            assert registry.attribute_exception(tb) is None
