"""A corrected value the app writes back must not masquerade as the typed one.

A text handler that clips what the user typed writes the correction back into
the box and declares it. If a record carrying that corrected value then reaches
the debounced logger -- which is keyed by record name and cancels any pending
line for that name -- it would REPLACE the typed record, and the bundle would
read:

    TEXT_INPUT GAIN_BF 24
    TEXT_INPUT GAIN_BF_APPLIED 24

asserting the user typed the value it simultaneously reports as a correction,
while the 99 they actually typed is gone. The declaration marks that record as
the app's own; it is consumed by the deferred emit if nothing echoes it, so it
cannot swallow a later deliberate retype.

These tests drive the emitter directly with a fake Clock, so they exercise the
real suppression logic rather than asserting on source text.
"""

import pytest

from ui import ui_helpers


class _FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def clock(monkeypatch):
    """A Clock that records what was scheduled and lets the test fire it."""
    scheduled = []

    class _Clock:
        @staticmethod
        def schedule_once(fn, delay):
            timer = _FakeTimer()
            scheduled.append((fn, timer))
            return timer

    monkeypatch.setattr(ui_helpers, 'Clock', _Clock)
    monkeypatch.setattr(ui_helpers, '_text_input_debounce_timers', {})
    monkeypatch.setattr(ui_helpers.gui_logger, '_write_backs', {})
    return scheduled


@pytest.fixture
def emitted(monkeypatch):
    """Capture what actually reaches the gui_interactions vocabulary."""
    lines = []
    monkeypatch.setattr(
        ui_helpers.gui_logger, 'text_input', lambda name, value: lines.append((name, value))
    )
    return lines


def _fire(scheduled):
    """Run the callbacks a real Clock would run -- cancelled timers do not fire."""
    for fn, timer in list(scheduled):
        if not timer.cancelled:
            fn(0)
    scheduled.clear()


def test_an_echoed_correction_keeps_the_typed_value(clock, emitted):
    """Log, correct, declare, then a record carrying the correction arrives."""
    # the handler reads 99, records it, clips to 24, writes 24 into the box
    ui_helpers.text_input_debounced('GAIN_BF', '99')
    ui_helpers.text_input_debounced('GAIN_BF_APPLIED', 24)
    ui_helpers.gui_logger.note_write_back('GAIN_BF', 24)

    # the echo -- a record for the value the app just wrote
    ui_helpers.text_input_debounced('GAIN_BF', '24')

    _fire(clock)

    assert ('GAIN_BF', '99') in emitted, (
        'the typed value was replaced by the correction; the bundle would claim '
        f'the user typed what the app wrote. Got {emitted}'
    )
    assert ('GAIN_BF', '24') not in emitted, f'the echo was recorded as typed: {emitted}'
    assert ('GAIN_BF_APPLIED', 24) in emitted, 'the correction itself must still be reported'


def test_only_one_echo_is_absorbed(clock, emitted):
    """A user who genuinely retypes the corrected value still gets a record."""
    ui_helpers.gui_logger.note_write_back('GAIN_BF', 24)
    ui_helpers.text_input_debounced('GAIN_BF', '24')  # the echo -- dropped
    ui_helpers.text_input_debounced('GAIN_BF', '24')  # deliberate retype -- kept

    _fire(clock)

    assert ('GAIN_BF', '24') in emitted, (
        'the suppression outlived its one echo and swallowed a real user entry'
    )


def test_a_different_value_is_never_suppressed(clock, emitted):
    """A write-back must not mask an unrelated later entry."""
    ui_helpers.gui_logger.note_write_back('GAIN_BF', 24)
    ui_helpers.text_input_debounced('GAIN_BF', '7')

    _fire(clock)

    assert ('GAIN_BF', '7') in emitted, f'an unrelated value was suppressed: {emitted}'


def test_an_unconsumed_write_back_does_not_outlive_its_line(clock, emitted):
    """One entry is one handler call, so the echo never arrives.

    The declaration must not survive to swallow a later real entry of the same
    value -- which is what would happen if it were only cleared on being matched.
    """
    ui_helpers.text_input_debounced('GAIN_BF', '99')
    ui_helpers.gui_logger.note_write_back('GAIN_BF', 24)
    _fire(clock)

    ui_helpers.text_input_debounced('GAIN_BF', '24')  # much later, genuinely typed
    _fire(clock)

    assert emitted.count(('GAIN_BF', '24')) == 1, (
        f'a stale write-back declaration swallowed a later real entry: {emitted}'
    )


def test_per_keystroke_fields_still_collapse_to_the_settled_value(clock, emitted):
    """The collapse is still the function's shape: last value wins in a burst.

    No caller sends a per-keystroke burst today, but a field that logged on
    every keystroke would rely on typing 123 recording 123 -- not 1. This is
    why the suppression could not simply keep the first value seen in a window.
    """
    for partial in ('1', '12', '123'):
        ui_helpers.text_input_debounced('PROTOCOL_PERIOD', partial)

    _fire(clock)

    assert ('PROTOCOL_PERIOD', '123') in emitted
    assert ('PROTOCOL_PERIOD', '1') not in emitted
