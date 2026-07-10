# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stim UI is gated on the LED firmware's stim capability.

firmware_stim_supported() is the single gate for all stimulation UI: True only
when the connected scope reports firmware stim support, and it fails safe to
False (hide) when the scope or capability surface is unavailable. Stim controls
appearing on firmware that cannot drive stim is the defect this guards against.
"""

from unittest.mock import MagicMock

import modules.app_context as _app_ctx
from modules.lumascope_api import Lumascope
from modules.config_ui_getters import firmware_stim_supported


def _ctx_reporting_stim(supported):
    # The gate reads the LIVE scope (ctx.lumaview.scope), the reference reconnect
    # rebuilds -- not the build-time ctx.scope registry field.
    ctx = MagicMock()
    ctx.lumaview.scope.capabilities.supports.return_value = supported
    return ctx


def test_gate_true_when_firmware_supports(monkeypatch):
    ctx = _ctx_reporting_stim(True)
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    assert firmware_stim_supported() is True
    ctx.lumaview.scope.capabilities.supports.assert_called_with('firmware_stim')


def test_gate_false_when_firmware_lacks_support(monkeypatch):
    monkeypatch.setattr(_app_ctx, 'ctx', _ctx_reporting_stim(False))
    assert firmware_stim_supported() is False


def test_gate_fails_safe_to_hidden_when_no_scope(monkeypatch):
    ctx = MagicMock()
    ctx.lumaview.scope = None
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    assert firmware_stim_supported() is False


def test_default_simulated_scope_reports_no_firmware_stim():
    # The pre-3.0.8-firmware case (a scope that cannot stim): the capability
    # must read False so the gate hides stim. This is the real-path anchor for
    # the mocked gate tests above.
    scope = Lumascope(simulate=True)
    assert scope.capabilities.supports('firmware_stim') is False
