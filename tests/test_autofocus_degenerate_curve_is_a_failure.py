# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A flat or all-zero focus curve is a FAILURE, and reports like one.

`AutofocusRunner.run()`'s docstring has always said it returns "None when
the AF curve was degenerate". The code did not: the degenerate exit
assigned the PRE-AUTOFOCUS Z into the result and set the completion
event, so the curve that found nothing reported a float and claimed to
have completed.

That was load-bearing rather than merely wrong. The protocol step runner
reads the result to place a z-stack group and move the stage, so the
mislabeled value was the only thing putting the stage back after a failed
sweep -- which is why it could not be deleted on its own.

The consequences it produced, worst first:

* Nothing moves the stage between an autofocus and its capture. The
  degenerate exit left the stage wherever the sweep ended, up to one
  `AF_range` from the intended plane, and the step captured and SAVED
  there. The operator was told the autofocus failed; nothing said the
  image that followed was off-plane.
* The reported focus was a position no autofocus had chosen.
* `complete()` answered True for a run that completed nothing.

Now: the stage returns to where it started, the result is None, and
`complete()` is False. The channel LED keeps being held for the capture
that follows, which is deliberate and separately covered -- the capture
does still run, so darkening and re-lighting it would be a blink for
nothing.

What is NOT changed here: the detector itself. `:725` tests
`scores.max() == 0 or scores.isna().all()`, so a genuinely FLAT non-zero
curve is not detected at all, despite the comment and the popup both
saying "flat". Widening it is separate work.
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import pytest

_mock_settings_init = MagicMock()
_mock_settings_init.settings = {'BF': {'autofocus': False}, 'Green': {'autofocus': False}}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from modules.exceptions import AutofocusAborted
from tests.af_drives import AF_CENTER_Z, af_runner_and_scope, drive_af


def _flat_curve(monkeypatch):
    """Every frame scores zero -- the degenerate case the exit detects."""
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 0.0)


def _z_moves(scope):
    """Every absolute Z move issued through the non-dispatching body."""
    return [
        call.args[1]
        for call in scope.motion._move_absolute_impl.call_args_list
        if call.args and call.args[0] == 'Z'
    ]


class TestDegenerateCurveReportsFailure:
    def test_the_result_is_none(self, monkeypatch):
        """D1: the value the docstring has always promised."""
        _flat_curve(monkeypatch)
        runner, _scope = af_runner_and_scope()

        result = drive_af(runner)

        assert result is None, f'a degenerate curve must report no result; got {result}'
        assert runner.best_focus_position() is None, (
            'and the stored result must agree with the returned one'
        )

    def test_complete_is_false(self, monkeypatch):
        """D2: nothing completed, so complete() must not say it did.

        This is also what re-points the Z readout: the step runner
        suppresses its own Z push only while complete() is True.
        """
        _flat_curve(monkeypatch)
        runner, _scope = af_runner_and_scope()

        drive_af(runner)

        assert runner.complete() is False, (
            'a degenerate curve completed nothing; complete() must be False'
        )

    def test_the_stage_returns_to_the_pre_autofocus_z(self, monkeypatch):
        """D3: the whole point. A failed autofocus is a no-op in position.

        Before this, the stage was left at the sweep end and a protocol
        step captured there.
        """
        _flat_curve(monkeypatch)
        runner, scope = af_runner_and_scope()

        drive_af(runner)

        moves = _z_moves(scope)
        assert moves, 'the sweep must have moved the stage at all'
        assert moves[-1] == AF_CENTER_Z, (
            'the last Z move on a degenerate curve must be the restore to the '
            f'pre-autofocus position {AF_CENTER_Z}; got {moves[-1]} (all: {moves})'
        )

    def test_the_channel_led_is_still_held_for_the_capture(self, monkeypatch):
        """D5: the capture after a degenerate curve still runs, so the
        LED must not be darkened and re-lit.

        The rejected mechanism for this fix -- flipping
        completed_successfully -- would have turned this off, because that
        same flag gates the hold. It is the regression that mechanism
        would have caused, and it is why the flag was left alone.
        """
        _flat_curve(monkeypatch)
        runner, scope = af_runner_and_scope()

        drive_af(runner, keep_led_on=True, led_color='Green')

        lease = scope.illumination.acquire_led_lease.return_value
        assert lease.apply.called, 'the AF-end LED transition must be applied'
        ctx = lease.apply.call_args_list[-1].args[1]
        assert ctx.keep_led_on is True, (
            'a degenerate curve must still hold the channel for the capture '
            f'that follows; got keep_led_on={ctx.keep_led_on}'
        )

    def test_an_abort_racing_a_degenerate_curve_reports_the_abort(self, monkeypatch):
        """D7: a stated behaviour change, tested rather than discovered.

        The degenerate exit no longer sets the completion event, so the
        abort check at the top of the unwind now fires where it used to
        be suppressed. The caller asked to stop; reporting the stop is
        right.
        """
        _flat_curve(monkeypatch)
        runner, _scope = af_runner_and_scope()
        abort = threading.Event()

        real_iterate = runner._iterate

        def _iterate_then_abort():
            real_iterate()
            abort.set()

        monkeypatch.setattr(runner, '_iterate', _iterate_then_abort)

        with pytest.raises(AutofocusAborted):
            drive_af(runner, abort_event=abort)

    def test_the_pre_autofocus_snapshot_field_is_gone(self):
        """D8: one value, one store.

        The restore used to read `_saved_z_position`, a second snapshot of
        the same quantity taken fourteen lines after `_params['center']`,
        inside a try whose except swallowed the failure to a debug line
        and assigned None -- which silently switched the restore OFF. The
        field is deleted rather than guarded.
        """
        runner, _scope = af_runner_and_scope()

        assert not hasattr(runner, '_saved_z_position'), (
            'the duplicate pre-AF Z snapshot must be gone; the restore reads '
            "_params['center'], which cannot be missing"
        )


class TestTheSuccessPathIsUnchanged:
    def test_success_still_reports_and_leaves_the_stage_at_the_result(self, monkeypatch):
        """D6: the fix must not touch the path that works.

        A positive score drives the two-pass success exit, whose last
        move is to the found position -- NOT to the pre-autofocus Z.
        """
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()

        result = drive_af(runner)

        assert result == AF_CENTER_Z, f'the success path must report its result; got {result}'
        assert runner.complete() is True, 'a completed autofocus must say so'
        moves = _z_moves(scope)
        assert moves[-1] == AF_CENTER_Z, (
            f'success must leave the stage at the found position; got {moves[-1]}'
        )
