# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The display path must never write a layer's saved settings.

Bug
---
While a layer's auto-gain was armed, every tenth display frame queued a read of
the API's gain/exposure CACHE and pushed the result into that layer's sliders.
A slider write fires LayerControl.gain_slider() / exp_slider(), which commit the
slider's value into settings[layer][...] -- so a display refresh silently
overwrote the user's saved gain and exposure with whatever the camera happened
to be doing, and kept doing it ten times a second while the arm was up.

The display is a rendering of state, never an author of it. The committed gain
and exposure change when the user changes them, or when the auto-gain lock
writes back at toggle-off; a poll that exists to show a value must not be able
to commit one.

Fix
---
The poll is deleted rather than guarded: showing the camera's live values while
armed was the only thing it did, and the achieved values already reach the user
through the lock's write-back at toggle-off. While an arm is up the controls now
hold the user's committed values.

Test approach
-------------
Source-level locks. The UI modules touch Kivy widgets and cannot be imported
under the test mocks, so this asserts the shape at the source: the display
module defines no per-frame camera-readback push, and writes no layer slider at
all. The second assertion is the durable one -- it catches a reintroduction
under any new name.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SCOPE_DISPLAY = REPO / 'ui' / 'scope_display.py'


def _tree() -> ast.Module:
    return ast.parse(SCOPE_DISPLAY.read_text(encoding='utf-8'))


def test_the_display_writes_no_layer_slider():
    """The durable lock: nothing in the display path assigns a layer control."""
    writes = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == 'value'
                and isinstance(target.value, ast.Subscript)
                and isinstance(target.value.slice, ast.Constant)
                and isinstance(target.value.slice.value, str)
                and target.value.slice.value.endswith('_slider')
            ):
                writes.append(f'{target.value.slice.value} at line {node.lineno}')
    assert writes == [], (
        'ui/scope_display.py writes a layer slider: '
        + ', '.join(writes)
        + '. A slider write fires the layer handler, which commits the value to '
        "settings -- the display would be authoring the user's saved state."
    )


def test_the_per_frame_camera_readback_push_is_gone():
    """The specific shape that caused it, so a revert is caught by name."""
    defined = {
        node.name
        for node in ast.walk(_tree())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for gone in ('get_true_gain_exp', 'update_auto_gain_ui'):
        assert gone not in defined, (
            f'{gone} is back in ui/scope_display.py. It read the API gain/exposure '
            "cache every tenth frame and pushed it into the armed layer's sliders, "
            "which committed it to that layer's saved settings."
        )


def test_the_display_loop_queues_no_camera_work_on_auto_gain():
    """The poll's trigger, not just its body."""
    src = SCOPE_DISPLAY.read_text(encoding='utf-8')
    assert 'camera_executor.put' not in src, (
        'ui/scope_display.py queues camera-lane work again. The display loop ran '
        'one such task per ten frames purely to refresh two sliders; a display '
        'path that needs a camera round trip is the shape this removed.'
    )
