# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the live-preview blit gate (`_downscale_for_blit`).

The default preview path uploads the full-resolution frame and lets the GPU
minify it to the widget. A host-side `cv2.resize` at a fractional contain-fit
ratio costs roughly a full core (measured on the bench: fit-to-window ~2x the
process CPU of a 1:1 view, which does no resize), while the full-res upload is
a small fraction of a core -- so the resize is off by default and only turns
back on for a machine that needs it, via the `preview_host_downscale` setting.

These lock the contract:
  * default (setting off): the frame is returned unchanged for a full-res blit;
  * setting on: the frame is host-downscaled to the on-screen widget size;
  * no app context yet (early startup): full-res, never a crash.

The real ScopeDisplay is a Kivy widget needing a GL context; the gate touches
only a few instance attributes, so a minimal stand-in borrowing the real
methods exercises the exact code without constructing the widget.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


for _name in (
    'kivy.uix',
    'kivy.graphics',
    'kivy.graphics.texture',
    'kivy.metrics',
    'kivy.properties',
    'kivy.input',
    'kivy.clock',
):
    sys.modules.setdefault(_name, MagicMock())

_real_base_module('kivy.uix.image', Image=_StubWidget)
_real_base_module('kivy.uix.widget', Widget=_StubWidget)

import modules.app_context as app_context
from ui.scope_display import ScopeDisplay


class _Ctx:
    def __init__(self, host_downscale):
        self.settings = {'preview_host_downscale': host_downscale}


class _Stand:
    """Carries only the state the blit gate touches + the real methods."""

    _downscale_for_blit = ScopeDisplay._downscale_for_blit
    _current_preview_target = ScopeDisplay._current_preview_target
    _log_preview_downscale = ScopeDisplay._log_preview_downscale

    def __init__(self, target_wh=(700, 700)):
        self.parent = None  # no enclosing Scatter -> scale 1.0 (fit-to-window)
        self._preview_target_wh = target_wh
        self._preview_downscale_logged = None


def test_default_uploads_full_resolution(monkeypatch):
    # Setting off (default): the oversized frame is passed through untouched so
    # the GPU minifies it -- no host cv2.resize. Fails on the pre-gate code,
    # which always downscaled to the widget size.
    monkeypatch.setattr(app_context, 'ctx', _Ctx(host_downscale=False))
    img = np.zeros((1900, 1900), dtype=np.uint8)
    assert _Stand()._downscale_for_blit(img) is img


def test_host_downscale_setting_shrinks_to_widget(monkeypatch):
    # Setting on: the fallback path host-downscales to the on-screen widget box
    # (700x700 here, scale 1.0), preserving the aspect-fit contract.
    monkeypatch.setattr(app_context, 'ctx', _Ctx(host_downscale=True))
    img = np.zeros((1900, 1900), dtype=np.uint8)
    out = _Stand(target_wh=(700, 700))._downscale_for_blit(img)
    assert out.shape == (700, 700)


def test_bullseye_stays_full_resolution_by_default(monkeypatch):
    # The categorical (bullseye) path takes the same gate: full-res by default,
    # so the GPU (nearest min-filter) preserves the contour colors.
    monkeypatch.setattr(app_context, 'ctx', _Ctx(host_downscale=False))
    img = np.zeros((900, 900, 3), dtype=np.uint8)
    assert _Stand()._downscale_for_blit(img, categorical=True) is img


def test_no_app_context_uploads_full_resolution(monkeypatch):
    # Very early startup: no ctx yet. Full-res blit, never a crash.
    monkeypatch.setattr(app_context, 'ctx', None)
    img = np.zeros((1900, 1900), dtype=np.uint8)
    assert _Stand()._downscale_for_blit(img) is img


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
