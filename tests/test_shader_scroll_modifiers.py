"""Regression: ShaderViewer reads live modifier state, not a private mirror.

Over the live image, a plain scroll digitally zooms and ctrl+scroll adjusts
focus. ShaderViewer used to gate that on a PRIVATE mirror of the ctrl/shift
state (`_active_key_presses`) that it maintained itself from key events. When a
real ctrl transition never reached its handlers -- window focus lost/regained
with ctrl held, or a popup consuming the event -- the mirror went stale, so
ctrl+scroll silently zoomed instead of focusing (and it self-recovered on the
next clean ctrl press).

Fix: query the live `Window.modifiers` (the OS-backed single owner, rebuilt
from the modifier state on every key event) so there is no mirror to desync.

Structural (code-shape) invariant, asserted from source: instantiating the GL
ShaderViewer needs a real window, unavailable headless.
"""

from __future__ import annotations

import re
from pathlib import Path

_SHADER = Path(__file__).resolve().parents[1] / 'ui' / 'shader.py'


def _method_body(src: str, name: str) -> str:
    m = re.search(
        rf'def {re.escape(name)}\b.*?(?=\n    def |\n    @|\nclass |\Z)',
        src,
        re.DOTALL,
    )
    assert m is not None, f'{name}() not found'
    return m.group(0)


def test_shader_has_no_private_modifier_mirror():
    src = _SHADER.read_text()
    for leftover in ('_active_key_presses', '_track_keys', '_key_down', '_key_up'):
        assert leftover not in src, (
            f'ShaderViewer still references {leftover!r}; the private modifier '
            f'mirror must be gone -- it desynced from the real ctrl/shift state '
            f'and broke ctrl+scroll focus. Query Window.modifiers instead.'
        )


def test_shader_scroll_reads_window_modifiers():
    body = _method_body(_SHADER.read_text(), 'on_touch_down')
    for key in ('ctrl', 'shift'):
        assert re.search(rf'[\'"]{key}[\'"]\s+in\s+Window\.modifiers', body), (
            f'ShaderViewer.on_touch_down must gate on "{key}" in '
            f'Window.modifiers (the live single-owner state), not a mirror.'
        )
