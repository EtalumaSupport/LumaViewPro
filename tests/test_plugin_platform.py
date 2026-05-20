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


def _make_spec(name: str = 'demo', version: str = '0.1.0',
               requires: str = '>=4.0.0') -> PluginSpec:
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
        name='x', version='0.1.0', requires_lvp_version='>=4.0.0',
        description='d',
    )
    assert spec.capabilities == ()
    assert spec.subscribes_to == ()
    assert spec.author == ''
    assert spec.url == ''
    assert spec.auto_run_on_protocol_complete is False


def test_plugin_spec_auto_run_opt_in():
    spec = PluginSpec(
        name='x', version='0.1.0', requires_lvp_version='>=4.0.0',
        description='d', auto_run_on_protocol_complete=True,
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
            spec, 'nonexistent.mount', lambda: None,
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
        name='opted', version='0.1.0', requires_lvp_version='>=4.0.0',
        description='d', auto_run_on_protocol_complete=True,
    )
    spec_skip = _make_spec(name='skipped')

    harness_ctx.plugins.post_processing.register(spec_opt, opted_in)
    harness_ctx.plugins.post_processing.register(spec_skip, not_opted_in)

    run_protocol_complete_processors(
        harness_ctx, input_dir='/in', manifest={'k': 'v'}, output_dir='/out',
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
        name='boomer', version='0.1.0', requires_lvp_version='>=4.0.0',
        description='d', auto_run_on_protocol_complete=True,
    )
    spec_b = PluginSpec(
        name='good_citizen', version='0.1.0', requires_lvp_version='>=4.0.0',
        description='d', auto_run_on_protocol_complete=True,
    )
    harness_ctx.plugins.post_processing.register(spec_a, boom)
    harness_ctx.plugins.post_processing.register(spec_b, runs_after)

    # Should not raise; runs_after should still execute despite boomer failing.
    run_protocol_complete_processors(
        harness_ctx, input_dir='/in', manifest={}, output_dir='/out',
    )

    assert calls == ['runs_after']


def test_auto_run_dispatcher_handles_no_ctx_plugins():
    from modules.plugins import run_protocol_complete_processors
    # Should silently return without raising when ctx has no .plugins attr.
    from types import SimpleNamespace
    ctx = SimpleNamespace()
    run_protocol_complete_processors(
        ctx, input_dir='/in', manifest={}, output_dir='/out',
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
        harness_ctx, input_dir='/in', manifest={}, output_dir='/out',
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
    harness_ctx.scope.imaging.add_frame_listener.assert_called_once_with(
        handler, name='live_demo'
    )
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


@pytest.mark.parametrize('requires,host,expected', [
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
])
def test_is_version_compatible(requires, host, expected):
    assert is_version_compatible(requires, host) is expected


# ---------------------------------------------------------------------------
# load_plugins / unload_plugins
# ---------------------------------------------------------------------------


def _make_plugin_module(name: str, version: str = '0.1.0',
                        requires: str = '>=4.0.0',
                        register_raises: bool = False,
                        unregister_raises: bool = False):
    """Build an in-memory module that looks like a plugin to load_plugins."""
    mod = types.ModuleType(f'fake_plugin_{name}')
    mod.__version__ = version
    mod.spec = PluginSpec(
        name=name, version=version, requires_lvp_version=requires,
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
            name=name, version='0.1.0', requires_lvp_version='>=4.0.0',
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
