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

The harness ctx exposes:
    ctx.plugins          -- real PluginRegistry
    ctx.scope            -- mocked Lumascope; attribute access does not raise
    ctx.settings         -- empty dict
    ctx.version          -- the LVP version string the host would pass
    ctx.engineering_mode -- False

The scope mock is intentionally minimal. Plugins that exercise scope
methods should set attributes on ctx.scope explicitly per test.
"""

from __future__ import annotations

import sys
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
    'harness_ctx',
    'PluginSpec',
    'PluginRegistrationError',
    'ProcessorResult',
]


def _make_ctx(version: str = '4.0.0') -> types.SimpleNamespace:
    """Build a fresh ctx-shaped object with a real PluginRegistry."""
    ctx = types.SimpleNamespace()
    ctx.plugins = PluginRegistry()
    ctx.scope = MagicMock(name='scope')
    ctx.settings = {}
    ctx.version = version
    ctx.engineering_mode = False
    ctx.lumaview = MagicMock(name='lumaview')
    ctx.session = MagicMock(name='session')
    # live_processing registry needs scope wired (load_plugins does this
    # in production; tests use the harness without load_plugins so do it
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
