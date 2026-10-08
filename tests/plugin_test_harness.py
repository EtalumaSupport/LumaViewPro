# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Plugin author's test harness.

Pytest fixtures that give a plugin author "a configured ctx" without
spinning up Kivy / LumaViewPro / hardware. Use these to assert that
your plugin's register/unregister/on_settings_changed hooks behave
correctly in isolation.

Usage in your plugin's test file:

    from tests.plugin_test_harness import harness_ctx

    def test_my_plugin_registers(harness_ctx):
        import my_plugin
        my_plugin.register(harness_ctx)
        loaded = harness_ctx.plugins.post_processing.names()
        assert 'my_plugin' in loaded

The harness ctx stands in for the ScopeSession the host hands a plugin,
and has only what a session has:
    ctx.plugins              -- real PluginRegistry
    ctx.scope                -- mocked Lumascope; attribute access does not raise
    ctx.post_processing      -- mocked post-processing builds
    ctx.engineering_mode     -- False
    ctx.no_engineering       -- False
    ctx.get_settings_snapshot() -- a copy of the settings, empty to begin with
    ctx.update_settings      -- mocked

A plugin that reaches for anything else a session does not have fails
here as it would in LumaViewPro. The mocks are intentionally minimal:
plugins that exercise scope methods should set attributes on ctx.scope
explicitly per test.
"""

from __future__ import annotations

import copy
import types
from unittest.mock import MagicMock

import pytest

# Re-export PluginSpec / ProcessorResult so plugin authors can build
# specs in their tests without importing modules.plugins directly.
from modules.plugins import (
    PluginRegistry,
    PluginSpec,
    PluginRegistrationError,
    ProcessorResult,
)


__all__ = [
    'PluginRegistrationError',
    'PluginSpec',
    'ProcessorResult',
    'harness_ctx',
]


def _make_ctx() -> types.SimpleNamespace:
    """Build a fresh session-shaped ctx with a real PluginRegistry."""
    ctx = types.SimpleNamespace()
    ctx.plugins = PluginRegistry()
    ctx.scope = MagicMock(name='scope')
    ctx.post_processing = MagicMock(name='post_processing')
    ctx.engineering_mode = False
    ctx.no_engineering = False
    settings: dict = {}
    ctx.get_settings_snapshot = lambda: copy.deepcopy(settings)
    ctx.update_settings = MagicMock(name='update_settings')
    # live_processing registry needs scope wired (PluginRegistry.load does
    # this in production; tests use the harness without loading, so do it
    # here). Tests that need an unbound registry can reset
    # ctx.plugins.live_processing._scope = None.
    ctx.plugins.live_processing.bind_scope(ctx.scope)
    return ctx


@pytest.fixture
def harness_ctx() -> types.SimpleNamespace:
    """Fresh ctx per test. Plugin registrations do not leak across tests."""
    return _make_ctx()


@pytest.fixture
def harness_ctx_factory():
    """Lets a test build multiple isolated ctx instances (e.g. to test that
    two ctx instances don't share state)."""
    return _make_ctx
