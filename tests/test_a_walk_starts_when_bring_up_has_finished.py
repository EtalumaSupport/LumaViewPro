# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A sim walk's step 1 runs once bring-up has finished, by its tracked facts.

``ctx.ready`` is set by a Clock timer 0.3 s after ``on_start``; nothing it
names has finished. The saved protocol loads later, after the objective
question resolves, and the display shows its first camera frame whenever the
display thread delivers one. A walk started on the timer pressed a control
mid-layout ("the touch did not reach it") and shot an all-black window (docs
track, 2026-10-05). The app hands the walk what bring-up still owes; the walk
starts on nothing owed and stops, saying what was owed, when that never comes.
"""

from __future__ import annotations

import ast

from tests.ast_seams import find_def


def _app_owing(
    *, ready: bool, protocol_loaded: bool, frames_shown: int, popups: list | None = None
) -> list[str]:
    import sys
    import types

    import modules.app_context as _app_ctx

    class _App:
        _persisted_protocol_loaded = protocol_loaded
        _bring_up_owes = __import__(
            'lumaviewpro', fromlist=['LumaViewProApp']
        ).LumaViewProApp._bring_up_owes

    original = _app_ctx.ctx
    # The driver imports Kivy's test touch, which the suite's Kivy mock does
    # not carry; the predicate asks it one thing, which popups are open, so
    # a stand-in answers that for the call and is gone afterwards.
    driver = types.ModuleType('ui.sim_walk')
    driver.open_popups = lambda: list(popups or [])
    real_driver = sys.modules.get('ui.sim_walk')
    sys.modules['ui.sim_walk'] = driver
    try:
        _app_ctx.ctx = type(
            'C',
            (),
            {'ready': ready, 'scope_display': type('D', (), {'frames_shown': frames_shown})()},
        )()
        import lumaviewpro

        lumaviewpro.ctx = _app_ctx.ctx
        return _App()._bring_up_owes()
    finally:
        _app_ctx.ctx = original
        if real_driver is None:
            del sys.modules['ui.sim_walk']
        else:
            sys.modules['ui.sim_walk'] = real_driver


class TestWhatBringUpOwes:
    def test_nothing_once_every_fact_holds(self):
        assert _app_owing(ready=True, protocol_loaded=True, frames_shown=1) == []

    def test_each_missing_fact_by_name(self):
        assert _app_owing(ready=False, protocol_loaded=False, frames_shown=0) == [
            'initialization',
            'the saved protocol load',
            'a displayed frame',
        ]

    def test_the_frame_is_owed_until_one_is_shown(self):
        assert _app_owing(ready=True, protocol_loaded=True, frames_shown=0) == ['a displayed frame']

    def test_the_protocol_load_is_not_owed_while_a_question_is_showing(self):
        """The load is hung on the objective question; a walk of that question
        answers it, so a popup up is bring-up waiting on the walk (docs, 2026-10-05:
        an LS850T with the objective unconfirmed, a shot as step 1; at 65232950 the
        walk stopped after 30 s still owed the load)."""
        assert (
            _app_owing(ready=True, protocol_loaded=False, frames_shown=1, popups=[object()]) == []
        )
        assert _app_owing(ready=True, protocol_loaded=False, frames_shown=1) == [
            'the saved protocol load'
        ]


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


def test_the_walk_is_handed_what_bring_up_owes_not_the_timer_flag():
    fn = find_def('lumaviewpro.py', 'on_start')
    assert fn is not None
    src = ast.unparse(fn)
    assert 'bring_up_owes=self._bring_up_owes' in src, (
        'the walk waits on something other than what bring-up owes'
    )
    assert 'ready=lambda: ctx.ready' not in src, 'the walk is started on the 0.3 s timer flag again'
