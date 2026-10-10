# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live view's slow-frame check is told the camera's TRACKED settled state.

``ScopeDisplay._check_slow_frame`` takes ``settled`` as an argument and does
not judge a frame delivered while a camera change is still switching over.
Whether the camera has settled is frame validity's answer
(``ScopeDisplay._camera_settled`` asks the API's ``frames_until_valid`` with
stage motion left out), never something inferred from the frame cadence.
The detector's own tests feed ``settled`` by hand, and ``_camera_settled``
is tested against a real ``FrameValidity``; the one line joining them is the
render loop's call, ``ScopeDisplay._render_one_frame``.

That loop is a Kivy path no test can run headless, so the structural fact is
pinned here: the render loop calls the slow-frame check exactly once, and
its ``settled=`` keyword is a call to ``_camera_settled`` -- never a
constant, never a value computed in the widget. A render loop that passed
``settled=True`` would warn on every exposure change again; one that passed
``False`` would never report a real stall.
"""

import ast

from tests.ast_seams import find_def


def test_the_render_loop_hands_the_slow_frame_check_the_camera_s_settled_state():
    render = find_def('ui/scope_display.py', '_render_one_frame', class_name='ScopeDisplay')
    assert render is not None, 'ScopeDisplay._render_one_frame is gone; re-pin the wiring'
    calls = [
        node
        for node in ast.walk(render)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == '_check_slow_frame'
    ]
    assert len(calls) == 1, f'the render loop calls _check_slow_frame {len(calls)} times'
    settled = [kw.value for kw in calls[0].keywords if kw.arg == 'settled']
    assert len(settled) == 1, 'the slow-frame check is not given settled by keyword'
    value = settled[0]
    assert isinstance(value, ast.Call), f'settled= is {ast.unparse(value)}, not a call'
    assert isinstance(value.func, ast.Attribute)
    assert value.func.attr == '_camera_settled', (
        f'settled= is {ast.unparse(value)}, not the tracked validity read'
    )
