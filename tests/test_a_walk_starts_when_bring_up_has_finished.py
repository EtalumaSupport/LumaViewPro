# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A walk's step 1 runs once bring-up has finished, by its tracked facts.

The walk is started at the startup motion's end, however long the home
takes: a Classic homes in about a minute, and a walk started beside the
motion stopped before step 1 when its 30 s bound ran out mid-home (an LS720,
2026-11-17). The motion is bounded by its drivers, so the walk's bound covers
only what follows it: the saved protocol loads after the objective question
resolves, and the display shows its first camera frame whenever the display
thread delivers one. A walk started before them pressed a control mid-layout
("the touch did not reach it") and shot an all-black window (docs track,
2026-10-05). The app hands the walk what bring-up still owes; the walk starts
on nothing owed and stops, saying what was owed, when that never comes. A
scope with no camera connected owes no frame: it has none to show.
"""

from __future__ import annotations

import ast

from tests.ast_seams import direct_call_names, find_def


def _app_owing(
    *,
    protocol_loaded: bool,
    frames_shown: int,
    popups: list | None = None,
    camera_present: bool = True,
) -> list[str]:
    from unittest import mock

    import _lumascope.modules.app_context as _app_ctx
    import ui.walk as walk_driver
    from tests.scope_fakes import build_scope

    scope = build_scope(simulate=True, register_atexit=False, sim_camera_present=camera_present)

    class _App:
        _persisted_protocol_loaded = protocol_loaded
        _bring_up_owes = __import__(
            'lumaviewpro', fromlist=['LumaViewProApp']
        ).LumaViewProApp._bring_up_owes

    original = _app_ctx.ctx
    # The predicate asks the driver one thing, which popups are open.
    try:
        _app_ctx.ctx = type(
            'C',
            (),
            {
                'scope': scope,
                'scope_display': type('D', (), {'frames_shown': frames_shown})(),
            },
        )()
        import lumaviewpro

        lumaviewpro.ctx = _app_ctx.ctx
        with mock.patch.object(walk_driver, 'open_popups', lambda: list(popups or [])):
            return _App()._bring_up_owes()
    finally:
        _app_ctx.ctx = original


class TestWhatBringUpOwes:
    def test_nothing_once_every_fact_holds(self):
        assert _app_owing(protocol_loaded=True, frames_shown=1) == []

    def test_each_missing_fact_by_name(self):
        assert _app_owing(protocol_loaded=False, frames_shown=0) == [
            'the saved protocol load',
            'a displayed frame',
        ]

    def test_the_frame_is_owed_until_one_is_shown(self):
        assert _app_owing(protocol_loaded=True, frames_shown=0) == ['a displayed frame']

    def test_no_frame_is_owed_by_a_scope_with_no_camera(self):
        assert _app_owing(protocol_loaded=True, frames_shown=0, camera_present=False) == []

    def test_the_protocol_load_is_not_owed_while_a_question_is_showing(self):
        """The load is hung on the objective question; a walk of that question
        answers it, so a popup up is bring-up waiting on the walk (docs, 2026-10-05:
        an LS850T with the objective unconfirmed, a shot as step 1; at 65232950 the
        walk stopped after 30 s still owed the load)."""
        assert _app_owing(protocol_loaded=False, frames_shown=1, popups=[object()]) == []
        assert _app_owing(protocol_loaded=False, frames_shown=1) == ['the saved protocol load']


class TestTheDisplayCountsWhatItShows:
    """Both live-view blits land on the widget's texture; each is a frame shown."""

    def test_each_live_view_blit_counts_a_frame(self):
        for name in ('create_and_set_texture', 'create_and_set_bullseye_texture'):
            fn = find_def('ui/scope_display.py', name)
            assert fn is not None, name
            src = ast.unparse(fn)
            assert 'self.frames_shown += 1' in src, (
                f'{name} sets the texture without counting the frame, so a walk '
                'waiting on a displayed frame never sees this one'
            )
            assert src.index('self.texture = ') < src.index('self.frames_shown += 1'), (
                f'{name} counts the frame before the texture is set'
            )


def _nested_def(fn: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f'{name} is defined {len(found)} times in _fit'
    return found[0]


def _first_line_naming(fn: ast.FunctionDef, text: str) -> int:
    for statement in fn.body:
        if text in ast.unparse(statement):
            return statement.lineno
    raise AssertionError(f'{fn.name} has no statement naming {text!r}')


def test_the_walk_is_started_at_the_startup_motions_end():
    """The walk is built and started where the startup motion's end runs,
    after ``ctx.ready`` is set and before the start counts as finished, and
    is handed what bring-up owes after the motion."""
    fit = find_def('lumaviewpro.py', '_fit', class_name='LumaViewProApp')
    assert fit is not None
    start_the_walk = _nested_def(fit, 'start_the_walk')
    assert 'bring_up_owes=self._bring_up_owes' in ast.unparse(start_the_walk), (
        'the walk waits on something other than what bring-up owes'
    )
    fit_calls = direct_call_names(fit)
    assert 'start_the_walk' not in fit_calls and 'Walk' not in fit_calls, (
        'the walk is started beside the startup motion, so its bound runs through the home'
    )
    end = _nested_def(fit, 'after_the_startup_motion')
    ready = _first_line_naming(end, 'complete_initialization(')
    walk = _first_line_naming(end, 'start_the_walk(')
    finished = _first_line_naming(end, 'crash_guard.started = True')
    assert ready < walk < finished, (
        'the walk starts before ctx.ready is set, or after the start counts as '
        'finished (a raise in it would then be reported, not a start that failed)'
    )
