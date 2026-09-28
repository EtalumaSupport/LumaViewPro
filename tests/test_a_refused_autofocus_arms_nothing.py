# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A click that does not start a run must leave nothing armed behind it.

The standalone autofocus button arms a 15-second stuck-AF bound: if the
sweep stops progressing, the timer force-aborts it rather than leaving
the lockout up until someone notices. The bound is real and stays.

What matters is WHEN it is armed. Armed before ``prepare()``, it
outlives every exit between the arm and a run actually starting -- the
engine's refusal, a raise out of the protocol builder, anything the
starter's blanket handler catches. The timer then sits live for 15
seconds with no run of its own, and its predicate cannot tell the
difference: it asks whether the button's run handle is live, reading the
handle when it fires -- and by then the button holds the handle of the
next standalone autofocus the user started inside that window, which is
the run it force-aborts. A click that was refused would be reaching forward to
kill the click that was not.

So the bound is armed by the run it bounds: by the button's redraw, the
first time it sees that run live, and once per run. A press that started
nothing never shows a live run, so it arms nothing whichever way it
ended; disarming on each refusal path would fix the exits by hand.
"""

from __future__ import annotations

from tests.test_the_autofocus_button_runs_through_run_autofocus import (  # noqa: F401
    _live,
    held,
    pressed,
)
from modules.exceptions import ProtocolRunRefusedError
from modules.run_outcome import PendingRunOutcome


REFUSAL = ProtocolRunRefusedError(
    reason='already_running',
    title='Run In Progress',
    message='A protocol run is using the microscope.',
)


class TestARefusedAutofocusClick:
    def test_it_arms_no_safety_timer(self, pressed):
        pressed.member.run_autofocus.side_effect = REFUSAL

        pressed.button.run_autofocus_from_ui()

        assert pressed.member.run_autofocus.called, (
            'the click never reached the engine -- the test is not exercising the refusal'
        )
        assert pressed.button.armed == [], (
            'a refused click starts no run, so it must leave no stuck-AF bound '
            "behind it: the next real autofocus is what that timer's predicate "
            f'would match. Timer calls: {pressed.button.armed}'
        )

    def test_a_started_run_still_arms_one(self, pressed):
        _live(pressed.engine, pressed.handle)

        pressed.button.run_autofocus_from_ui()

        assert pressed.button.armed == [pressed.handle], (
            'the stuck-AF bound is the reason this timer exists; a committed '
            f'run must still get one. Timer calls: {pressed.button.armed}'
        )

    def test_every_later_redraw_of_the_same_run_arms_nothing_more(self, pressed):
        _live(pressed.engine, pressed.handle)
        pressed.button.run_autofocus_from_ui()

        pressed.button.draw_autofocus_button()
        pressed.button.draw_autofocus_button()

        assert pressed.button.armed == [pressed.handle], (
            'one bound per run: a re-armed timer restarts the 15 s, and a run '
            'redrawn on every edge would never be bounded at all'
        )

    def test_the_next_run_gets_its_own(self, pressed):
        first, second = pressed.handle, PendingRunOutcome()
        _live(pressed.engine, first)
        pressed.button.run_autofocus_from_ui()
        _live(pressed.engine)
        pressed.button.draw_autofocus_button()

        pressed.member.run_autofocus.return_value = second
        _live(pressed.engine, second)
        pressed.button.run_autofocus_from_ui()

        assert pressed.button.armed == [first, second]
