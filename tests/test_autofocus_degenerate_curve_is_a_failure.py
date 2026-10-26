# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A flat or all-zero focus curve is a FAILURE, and reports like one.

The degenerate exit once assigned the PRE-AUTOFOCUS Z into the result and
set the completion event, so a curve that found nothing reported a float
and claimed to have completed; a step then captured wherever the sweep
ended. Later it reported its own failure and returned None, a second exit
beside ``run()``'s failure path, and a single-autofocus run read that None
as a run that completed.

Now the flat curve raises ``AutofocusFailedError('flat_focus_curve')`` and
leaves through ``run()``'s one failure path, like every other failure: it
is reported once, the stage returns to where the sweep started,
``complete()`` is False, and the channel LED is handed back rather than
held, since the capture that follows is not the failed sweep's to light.

What is NOT changed here: the detector itself. It tests
``scores.max() == 0 or scores.isna().all()``, so a genuinely FLAT non-zero
curve is not detected at all, despite the words saying "flat". Widening it
is separate work.
"""

from __future__ import annotations

import pytest

from modules.exceptions import AutofocusFailedError
from modules.notification_center import Severity
from tests.af_drives import AF_CENTER_Z, af_lease, af_runner_and_scope, drive_af
from tests.stock_channels import LED1


def _flat_curve(monkeypatch):
    """Every frame scores zero -- the degenerate case the exit detects."""
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 0.0)


def _z_moves(scope):
    """Every absolute Z move autofocus issued, started or waited, in order."""
    return [
        call.args[0] for call in scope.motion.mock_calls if call[0] in ('start_move_z', 'move_z')
    ]


def _drive_flat(runner, **overrides):
    """Drive a sweep over a flat curve; it raises the flat-curve failure."""
    with pytest.raises(AutofocusFailedError) as raised:
        drive_af(runner, **overrides)
    assert raised.value.reason == 'flat_focus_curve'


class TestDegenerateCurveReportsFailure:
    def test_it_raises_and_is_reported_once(self, monkeypatch, centre_posts):
        """The failure is raised, told once as 'Autofocus Failed' in the
        flat curve's words, and no focus is stored."""
        _flat_curve(monkeypatch)
        runner, _scope = af_runner_and_scope()

        _drive_flat(runner)

        errors = [(n.title, n.message) for n in centre_posts if n.severity is Severity.ERROR]
        assert errors == [('Autofocus Failed', str(AutofocusFailedError('flat_focus_curve')))], (
            f'a flat curve must be told once, as itself; got {errors}'
        )
        assert runner.best_focus_position() is None, 'a flat curve chose no focus'

    def test_complete_is_false(self, monkeypatch):
        """Nothing completed, so complete() must not say it did.

        This is also what re-points the Z readout: the step runner
        suppresses its own Z push only while complete() is True.
        """
        _flat_curve(monkeypatch)
        runner, _scope = af_runner_and_scope()

        _drive_flat(runner)

        assert runner.complete() is False, (
            'a degenerate curve completed nothing; complete() must be False'
        )

    def test_the_stage_returns_to_the_pre_autofocus_z(self, monkeypatch):
        """The whole point: a failed autofocus is a no-op in position, so
        a protocol step never captures at the sweep's end."""
        _flat_curve(monkeypatch)
        runner, scope = af_runner_and_scope()

        _drive_flat(runner)

        moves = _z_moves(scope)
        assert moves, 'the sweep must have moved the stage at all'
        assert moves[-1] == AF_CENTER_Z, (
            'the last Z move on a degenerate curve must be the restore to the '
            f'pre-autofocus position {AF_CENTER_Z}; got {moves[-1]} (all: {moves})'
        )

    def test_the_channel_led_is_handed_back_not_held(self, monkeypatch):
        """A failure, like every failure, restores the pre-autofocus LED
        state even when the step asked to keep the channel lit: the
        capture that follows is not the failed sweep's to light for, and
        re-lights its own channel."""
        _flat_curve(monkeypatch)
        runner, scope = af_runner_and_scope()

        _drive_flat(runner, keep_led_on=True, led_color=LED1)

        lease = af_lease(scope)
        assert lease.apply.called, 'the AF-end LED transition must be applied'
        ctx = lease.apply.call_args_list[-1].args[1]
        assert ctx.keep_led_on is False, (
            f'a degenerate curve must not hold the channel; got keep_led_on={ctx.keep_led_on}'
        )

    def test_the_pre_autofocus_snapshot_field_is_gone(self):
        """One value, one store.

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
