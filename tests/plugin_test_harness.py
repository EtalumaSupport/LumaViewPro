# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""LumaViewPro's own test harness for its plugins.

Pytest fixtures that give a test of a built-in plugin, or of the plugin
platform, "a configured ctx" without spinning up Kivy / LumaViewPro /
hardware. It is this suite's, not published: a plugin author's tests build
their own stand-in over ``lumascope.PluginRegistry``, as
``docs/plugin-tutorial.md`` teaches.

Usage in a test file of this suite:

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
    ctx.engineering_mode     -- False, a read; ctx.set_engineering_mode(enabled) writes it
    ctx.no_engineering       -- False
    ctx.get_settings_snapshot() -- a copy of the settings, the shipped template to begin with
    ctx.update_settings      -- the Session's own check against the shipped template, then the
                                store, so a write the Session would refuse is refused here too

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

# Re-exported so a test can build specs from the harness's one import.
from _lumascope.modules.plugins import (
    PluginRegistry,
    PluginSpec,
    PluginRegistrationError,
    ProcessorResult,
)
from tests.settings_fixtures import complete_settings, settings_writer


__all__ = [
    'PluginRegistrationError',
    'PluginSpec',
    'ProcessorResult',
    'harness_ctx',
]


class _HarnessSession(types.SimpleNamespace):
    """The session's shape where a plain namespace would differ: its engineering
    mode is a read, so a plugin storing to it fails here as in LumaViewPro."""

    _engineering_mode = False

    @property
    def engineering_mode(self) -> bool:
        return self._engineering_mode

    def set_engineering_mode(self, enabled: bool) -> None:
        self._engineering_mode = enabled


def _make_ctx() -> _HarnessSession:
    """Build a fresh session-shaped ctx with a real PluginRegistry."""
    ctx = _HarnessSession()
    ctx.scope = MagicMock(name='scope')
    ctx.plugins = PluginRegistry(ctx.scope)
    ctx.post_processing = MagicMock(name='post_processing')
    ctx.no_engineering = False
    settings = complete_settings()
    ctx.get_settings_snapshot = lambda: copy.deepcopy(settings)
    ctx.update_settings = settings_writer(settings)
    return ctx


@pytest.fixture
def harness_ctx() -> _HarnessSession:
    """Fresh ctx per test. Plugin registrations do not leak across tests."""
    return _make_ctx()


@pytest.fixture
def harness_ctx_factory():
    """Lets a test build multiple isolated ctx instances (e.g. to test that
    two ctx instances don't share state)."""
    return _make_ctx
