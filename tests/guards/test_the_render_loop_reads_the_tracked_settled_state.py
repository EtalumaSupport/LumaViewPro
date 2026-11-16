# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The live view's slow-frame check is told the camera's TRACKED settled state, and judges every live draw.

The display thread adds each frame it picks up to the record the next live
draw is judged by (``ScopeDisplay._note_pickup``), with ``settled``: a frame
delivered while a camera change is still switching over is not judged.
Whether the camera has settled is frame validity's answer
(``ScopeDisplay._camera_settled`` asks the API's ``changes_in_force`` with
stage motion left out), never something inferred from the frame cadence.
The UI thread judges at each live frame drawn (``_judge_live_draw``). The
detector's own tests drive those seams by hand, and ``_camera_settled`` is
tested against a real ``FrameValidity``; the lines joining them are the
render loop's call and the two live draw sites.

Those are Kivy paths no test can run headless, so the structural facts are
pinned here: the render loop adds each frame once, its ``settled=`` keyword
a call to ``_camera_settled`` -- never a constant, never a value computed in
the widget -- and both live draw sites, the mono frame and the bullseye,
judge the draw. A render loop that passed ``settled=True`` would warn on
every exposure change again; one that passed ``False`` would never report a
real stall; a draw site that did not judge would leave its stutters
unreported.
"""

import ast

from tests.ast_seams import find_def


def _calls(func, name):
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_the_render_loop_hands_the_record_the_camera_s_settled_state():
    render = find_def('ui/scope_display.py', '_render_one_frame', class_name='ScopeDisplay')
    assert render is not None, 'ScopeDisplay._render_one_frame is gone; re-pin the wiring'
    calls = _calls(render, '_note_pickup')
    assert len(calls) == 1, f'the render loop calls _note_pickup {len(calls)} times'
    settled = [kw.value for kw in calls[0].keywords if kw.arg == 'settled']
    assert len(settled) == 1, 'the record is not given settled by keyword'
    value = settled[0]
    assert isinstance(value, ast.Call), f'settled= is {ast.unparse(value)}, not a call'
    assert isinstance(value.func, ast.Attribute)
    assert value.func.attr == '_camera_settled', (
        f'settled= is {ast.unparse(value)}, not the tracked validity read'
    )


def test_every_live_draw_site_judges_the_draw():
    unjudged = [
        site
        for site in ('create_and_set_texture', 'create_and_set_bullseye_texture')
        if len(
            _calls(
                find_def('ui/scope_display.py', site, class_name='ScopeDisplay'), '_judge_live_draw'
            )
        )
        != 1
    ]
    assert unjudged == [], f'live draw sites that do not judge the draw once: {unjudged}'
