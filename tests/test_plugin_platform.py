# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the plugin platform itself (modules/plugins/__init__.py).

Covers:
    - PluginSpec construction + immutability
    - UIRegistry: mount-point validation, duplicate-name rejection
    - PostProcessingRegistry: processor registration + lookup
    - LiveProcessingRegistry: stub raises with clear message
    - RESTRegistry: stub raises with clear message
    - PluginRegistry.all_health(): per-namespace snapshot shape
    - is_version_compatible: semver compare with pre-release suffixes
    - load_plugins / unload_plugins: discovery + lifecycle (mocked entry points)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from modules.plugins import (
    PluginRegistrationError,
    PluginRegistry,
    PluginSpec,
    ProcessorResult,
    UI_MOUNT_POINTS,
    is_version_compatible,
    load_plugins,
    unload_plugins,
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
        subscribes_to=(),
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


def test_ui_register_duplicate_name_raises(harness_ctx):
    spec = _make_spec(name='dup')
    harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: None)
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: None)
    assert "'dup'" in str(exc.value)
    assert "'ui'" in str(exc.value)


def test_ui_health_after_register(harness_ctx):
    spec = _make_spec(name='healthy')
    harness_ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: None)
    health = harness_ctx.plugins.ui.health()
    assert health.namespace == 'ui'
    assert len(health.loaded) == 1
    assert health.loaded[0].name == 'healthy'
    assert health.loaded[0].loaded is True
    assert health.failed == ()


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
    from modules.plugins import run_protocol_complete_processors

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

    run_protocol_complete_processors(
        harness_ctx,
        input_dir='/in',
        manifest={'k': 'v'},
        output_dir='/out',
    )

    assert len(calls) == 1
    assert calls[0] == ('opted_in', '/in', {'k': 'v'}, '/out')


def test_auto_run_dispatcher_swallows_processor_exception(harness_ctx, caplog):
    from modules.plugins import run_protocol_complete_processors

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

    # Should not raise; runs_after should still execute despite boomer failing.
    run_protocol_complete_processors(
        harness_ctx,
        input_dir='/in',
        manifest={},
        output_dir='/out',
    )

    assert calls == ['runs_after']


def test_auto_run_dispatcher_handles_no_ctx_plugins():
    from modules.plugins import run_protocol_complete_processors

    # Should silently return without raising when ctx has no .plugins attr.
    from types import SimpleNamespace

    ctx = SimpleNamespace()
    run_protocol_complete_processors(
        ctx,
        input_dir='/in',
        manifest={},
        output_dir='/out',
    )
    # No assertion needed -- absence of exception is the contract.


def test_auto_run_dispatcher_skips_when_all_opt_out(harness_ctx):
    from modules.plugins import run_protocol_complete_processors

    calls = []

    def proc(input_dir, manifest, output_dir):
        calls.append('ran')
        return ProcessorResult(success=True, message='ok')

    spec = _make_spec(name='not_opted')  # default False
    harness_ctx.plugins.post_processing.register(spec, proc)
    run_protocol_complete_processors(
        harness_ctx,
        input_dir='/in',
        manifest={},
        output_dir='/out',
    )
    assert calls == []


# ---------------------------------------------------------------------------
# LiveProcessingRegistry + RESTRegistry stubs
# ---------------------------------------------------------------------------


def test_live_processing_register_forwards_to_imaging(harness_ctx):
    """register(spec, handler) forwards to scope.imaging.add_frame_listener
    with the plugin name attached."""
    spec = _make_spec(name='live_demo')

    def handler(image, ts, chunks):
        return None

    harness_ctx.plugins.live_processing.register(spec, handler)
    harness_ctx.scope.imaging.add_frame_listener.assert_called_once_with(handler, name='live_demo')
    assert harness_ctx.plugins.live_processing.names() == ('live_demo',)


def test_live_processing_unregister_forwards_to_imaging(harness_ctx):
    """unregister(name) forwards to scope.imaging.remove_frame_listener
    with the original handler."""
    spec = _make_spec(name='live_demo')

    def handler(image, ts, chunks):
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


def test_live_processing_register_without_bind_raises(harness_ctx):
    """register() on an unbound registry raises a PluginRegistrationError
    naming the bind_scope contract."""
    harness_ctx.plugins.live_processing._scope = None
    spec = _make_spec(name='unbound')
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.live_processing.register(spec, lambda i, t, c: None)
    assert 'bind_scope' in str(exc.value)


def test_rest_register_raises_with_design_session_message(harness_ctx):
    spec = _make_spec(name='rest_attempt')
    with pytest.raises(PluginRegistrationError) as exc:
        harness_ctx.plugins.rest.register(spec, MagicMock(name='router'))
    msg = str(exc.value)
    assert 'rest' in msg.lower()
    assert 'REST design session' in msg or 'REST_API_PLAN.md' in msg
    # error message points plugin authors to the stub-mode opt-in
    assert 'enable_stub_mode' in msg


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
        assert ns.failed == ()
        assert ns.last_runtime_errors == ()


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


def test_load_plugins_with_no_entry_points_is_noop(harness_ctx):
    with patch('importlib.metadata.entry_points', return_value=[]):
        load_plugins(harness_ctx)
    assert harness_ctx.plugins.post_processing.names() == ()


def test_load_plugins_registers_a_valid_plugin(harness_ctx):
    mod = _make_plugin_module('valid_pp')
    eps = [_FakeEntryPoint('valid_pp', mod)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        load_plugins(harness_ctx)
    assert len(mod._register_calls) == 1
    assert mod._register_calls[0] is harness_ctx
    assert 'valid_pp' in harness_ctx.plugins.post_processing.names()


def test_load_plugins_skips_version_incompatible(harness_ctx):
    mod = _make_plugin_module('too_new', requires='>=5.0.0')
    eps = [_FakeEntryPoint('too_new', mod)]
    harness_ctx.version = '4.0.0'
    with patch('importlib.metadata.entry_points', return_value=eps):
        load_plugins(harness_ctx)
    assert mod._register_calls == []
    health = harness_ctx.plugins.ui.health()
    failed_names = [s.name for s in health.failed]
    assert 'too_new' in failed_names


def test_load_plugins_continues_past_register_failure(harness_ctx):
    bad = _make_plugin_module('bad', register_raises=True)
    good = _make_plugin_module('good')
    eps = [_FakeEntryPoint('bad', bad), _FakeEntryPoint('good', good)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        load_plugins(harness_ctx)
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
        load_plugins(harness_ctx)
    assert harness_ctx.plugins.post_processing.names() == ()


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
        load_plugins(harness_ctx)
    unload_plugins(harness_ctx)
    assert call_order == ['c', 'b', 'a']


def test_unload_plugins_swallows_unregister_failure(harness_ctx):
    bad = _make_plugin_module('bad_unload', unregister_raises=True)
    good = _make_plugin_module('good_unload')
    eps = [_FakeEntryPoint('bad', bad), _FakeEntryPoint('good', good)]
    with patch('importlib.metadata.entry_points', return_value=eps):
        load_plugins(harness_ctx)
    # Should not raise even though bad.unregister will throw.
    unload_plugins(harness_ctx)
    assert len(bad._unregister_calls) == 1
    assert len(good._unregister_calls) == 1


def test_unload_plugins_no_ctx_plugins_attr_is_noop():
    fake_ctx = types.SimpleNamespace()
    unload_plugins(fake_ctx)  # no exception expected


def test_load_plugins_no_ctx_is_noop():
    load_plugins(None)  # no exception expected


# ---------------------------------------------------------------------------
# on_settings_changed dispatch
# ---------------------------------------------------------------------------
#
# PluginSpec.subscribes_to declares dot-path settings keys; the host fires
# on_settings_changed when one of those keys (or a key under a subscribed
# subtree, by prefix match) changes between successive saves. Pre-F21,
# subscribes_to was declared but no dispatcher existed, so plugins got
# silently no settings reactivity.


def _make_plugin_with_settings_hook(
    name: str,
    subscribes_to: tuple[str, ...],
    on_change_raises: bool = False,
):
    """Build an in-memory plugin module with on_settings_changed wired."""
    mod = types.ModuleType(f'fake_settings_plugin_{name}')
    mod.__version__ = '0.1.0'
    mod.spec = PluginSpec(
        name=name,
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description=f'fake {name}',
        subscribes_to=subscribes_to,
    )
    mod._on_change_calls = []

    def _register(ctx):
        ctx.plugins.post_processing.register(mod.spec, _trivial_processor)

    def _unregister(ctx):
        pass

    def on_settings_changed(ctx, settings):
        mod._on_change_calls.append((ctx, settings))
        if on_change_raises:
            raise RuntimeError(f'{name} on_settings_changed intentionally failed')

    mod.register = _register
    mod.unregister = _unregister
    mod.on_settings_changed = on_settings_changed
    return mod


def test_diff_settings_keys_no_change_returns_empty():
    from modules.plugins import _diff_settings_keys

    assert _diff_settings_keys({'a': 1}, {'a': 1}) == set()


def test_diff_settings_keys_leaf_change():
    from modules.plugins import _diff_settings_keys

    assert _diff_settings_keys({'a': 1}, {'a': 2}) == {'a'}


def test_diff_settings_keys_nested_change_returns_dotted_path():
    from modules.plugins import _diff_settings_keys

    old = {'manual_video': {'max_fps': 10, 'max_duration': 60}}
    new = {'manual_video': {'max_fps': 30, 'max_duration': 60}}
    assert _diff_settings_keys(old, new) == {'manual_video.max_fps'}


def test_diff_settings_keys_added_and_removed_keys():
    from modules.plugins import _diff_settings_keys

    old = {'a': 1, 'b': 2}
    new = {'a': 1, 'c': 3}
    assert _diff_settings_keys(old, new) == {'b', 'c'}


def test_diff_settings_keys_none_baseline_flattens_all_leaves():
    from modules.plugins import _diff_settings_keys

    new = {'a': {'b': 1, 'c': 2}, 'd': 3}
    assert _diff_settings_keys(None, new) == {'a.b', 'a.c', 'd'}


def test_any_prefix_match_exact_key():
    from modules.plugins import _any_prefix_match

    assert _any_prefix_match(('manual_video.max_fps',), {'manual_video.max_fps'})
    assert not _any_prefix_match(('manual_video.max_fps',), {'manual_video.max_duration'})


def test_any_prefix_match_subtree():
    from modules.plugins import _any_prefix_match

    # subscribes to 'manual_video' subtree -> any descendant matches.
    assert _any_prefix_match(('manual_video',), {'manual_video.max_fps'})
    assert _any_prefix_match(('manual_video',), {'manual_video.codec.bitrate'})
    assert _any_prefix_match(('manual_video',), {'manual_video'})


def test_any_prefix_match_does_not_match_unrelated_prefix():
    from modules.plugins import _any_prefix_match

    # 'manual' must not match 'manual_video.X' -- prefix is full dot-path
    # component, not arbitrary string prefix.
    assert not _any_prefix_match(('manual',), {'manual_video.max_fps'})


def test_notify_settings_changed_fires_subscribed_plugin(harness_ctx):
    mod = _make_plugin_with_settings_hook(
        'video_listener',
        subscribes_to=('manual_video.max_fps',),
    )
    ep = _FakeEntryPoint('video_listener', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    settings = {'manual_video': {'max_fps': 30}}
    harness_ctx.plugins.notify_settings_changed(
        harness_ctx,
        settings,
        {'manual_video.max_fps'},
    )
    assert len(mod._on_change_calls) == 1
    ctx_arg, settings_arg = mod._on_change_calls[0]
    assert ctx_arg is harness_ctx
    assert settings_arg is settings


def test_notify_settings_changed_skips_non_subscribed(harness_ctx):
    mod = _make_plugin_with_settings_hook(
        'camera_listener',
        subscribes_to=('camera.gain',),
    )
    ep = _FakeEntryPoint('camera_listener', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    # Change a key the plugin did NOT subscribe to.
    harness_ctx.plugins.notify_settings_changed(
        harness_ctx,
        {'manual_video': {'max_fps': 30}},
        {'manual_video.max_fps'},
    )
    assert mod._on_change_calls == []


def test_notify_settings_changed_skips_plugin_with_empty_subscribes_to(harness_ctx):
    mod = _make_plugin_with_settings_hook('no_subs', subscribes_to=())
    ep = _FakeEntryPoint('no_subs', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    harness_ctx.plugins.notify_settings_changed(
        harness_ctx,
        {'a': 1},
        {'a'},
    )
    assert mod._on_change_calls == []


def test_notify_settings_changed_empty_changed_keys_is_noop(harness_ctx):
    mod = _make_plugin_with_settings_hook(
        'subscriber',
        subscribes_to=('a',),
    )
    ep = _FakeEntryPoint('subscriber', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    harness_ctx.plugins.notify_settings_changed(harness_ctx, {'a': 1}, set())
    assert mod._on_change_calls == []


def test_notify_settings_changed_swallows_handler_exception(harness_ctx):
    bad = _make_plugin_with_settings_hook(
        'bad_handler',
        subscribes_to=('a',),
        on_change_raises=True,
    )
    good = _make_plugin_with_settings_hook(
        'good_handler',
        subscribes_to=('a',),
    )
    eps = [
        _FakeEntryPoint('bad_handler', bad),
        _FakeEntryPoint('good_handler', good),
    ]
    with patch('importlib.metadata.entry_points', return_value=eps):
        load_plugins(harness_ctx)
    # bad raises, good must still fire, dispatcher must not propagate.
    harness_ctx.plugins.notify_settings_changed(
        harness_ctx,
        {'a': 1},
        {'a'},
    )
    assert len(bad._on_change_calls) == 1
    assert len(good._on_change_calls) == 1
    # Runtime error recorded against bad's namespace.
    health = harness_ctx.plugins.post_processing.health()
    errors = [e for e in health.last_runtime_errors if e.plugin_name == 'bad_handler']
    assert len(errors) == 1
    assert errors[0].hook == 'on_settings_changed'


def test_notify_settings_changed_prefix_subtree_subscription(harness_ctx):
    mod = _make_plugin_with_settings_hook(
        'subtree_listener',
        subscribes_to=('manual_video',),
    )
    ep = _FakeEntryPoint('subtree_listener', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    # Any key under the manual_video subtree must fire the handler.
    harness_ctx.plugins.notify_settings_changed(
        harness_ctx,
        {'manual_video': {'codec': {'bitrate': 5000}}},
        {'manual_video.codec.bitrate'},
    )
    assert len(mod._on_change_calls) == 1


def test_fire_settings_save_hooks_first_call_caches_without_firing(harness_ctx):
    from modules.plugins import fire_settings_save_hooks

    mod = _make_plugin_with_settings_hook('first_call', subscribes_to=('a',))
    ep = _FakeEntryPoint('first_call', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    fire_settings_save_hooks(harness_ctx, {'a': 1})
    assert mod._on_change_calls == [], (
        'First call after startup must cache the baseline without firing '
        'so plugins do not get spurious notifications for the boot-time state.'
    )
    assert harness_ctx._last_saved_settings_snapshot == {'a': 1}


def test_fire_settings_save_hooks_second_call_fires_on_diff(harness_ctx):
    from modules.plugins import fire_settings_save_hooks

    mod = _make_plugin_with_settings_hook('second_call', subscribes_to=('a',))
    ep = _FakeEntryPoint('second_call', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    fire_settings_save_hooks(harness_ctx, {'a': 1})  # cache
    fire_settings_save_hooks(harness_ctx, {'a': 2})  # fire
    assert len(mod._on_change_calls) == 1
    _, settings_arg = mod._on_change_calls[0]
    assert settings_arg == {'a': 2}
    assert harness_ctx._last_saved_settings_snapshot == {'a': 2}


def test_fire_settings_save_hooks_no_diff_does_not_fire(harness_ctx):
    from modules.plugins import fire_settings_save_hooks

    mod = _make_plugin_with_settings_hook('no_diff', subscribes_to=('a',))
    ep = _FakeEntryPoint('no_diff', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    fire_settings_save_hooks(harness_ctx, {'a': 1})
    fire_settings_save_hooks(harness_ctx, {'a': 1})  # same -> no fire
    assert mod._on_change_calls == []


def test_fire_settings_save_hooks_baseline_isolation_from_mutation(harness_ctx):
    """The cached baseline must be a deep copy so subsequent in-memory
    mutations to the settings dict do not poison the next diff."""
    from modules.plugins import fire_settings_save_hooks

    mod = _make_plugin_with_settings_hook('isolation', subscribes_to=('a',))
    ep = _FakeEntryPoint('isolation', mod)
    with patch('importlib.metadata.entry_points', return_value=[ep]):
        load_plugins(harness_ctx)
    settings = {'a': {'b': 1}}
    fire_settings_save_hooks(harness_ctx, settings)  # cache
    # Mutate the dict in place after caching.
    settings['a']['b'] = 999
    # Fire with a NEW dict that has a different value -- baseline should
    # still reflect the original cached state (1), so this fires.
    fire_settings_save_hooks(harness_ctx, {'a': {'b': 2}})
    assert len(mod._on_change_calls) == 1


def test_fire_settings_save_hooks_no_ctx_plugins_is_noop():
    from modules.plugins import fire_settings_save_hooks

    fake_ctx = types.SimpleNamespace()
    fire_settings_save_hooks(fake_ctx, {'a': 1})  # no exception expected


def test_fire_settings_save_hooks_none_ctx_is_noop():
    from modules.plugins import fire_settings_save_hooks

    fire_settings_save_hooks(None, {'a': 1})  # no exception expected


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
        registry = PluginRegistry()
        registry._track('fake-bench-plugin', module)

        try:
            module.boom()
        except RuntimeError:
            tb = sys.exc_info()[2]
            assert registry.attribute_exception(tb) == 'fake-bench-plugin'

    def test_none_for_non_plugin_frame(self):
        import sys

        from modules.plugins import PluginRegistry

        registry = PluginRegistry()
        try:
            raise RuntimeError('host code failure')
        except RuntimeError:
            tb = sys.exc_info()[2]
            assert registry.attribute_exception(tb) is None

    def test_none_when_no_plugins_loaded(self, tmp_path):
        import sys

        from modules.plugins import PluginRegistry

        module = self._make_plugin_module(tmp_path, 'fake_unloaded_plugin')
        registry = PluginRegistry()  # nothing tracked
        try:
            module.boom()
        except RuntimeError:
            tb = sys.exc_info()[2]
            assert registry.attribute_exception(tb) is None
