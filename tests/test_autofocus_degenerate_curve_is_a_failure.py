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

from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import AutofocusFailedError
from modules.lumascope_api.illumination import LedState
from modules.notification_center import Severity
from tests.af_drives import (
    FOCAL_Z_UM,
    PARKED_Z_UM,
    af_runner_on,
    settled_z,
    sweep,
    z_asked,
)


def _lit(scope):
    return [r.name for r in scope.illumination.get_led_states() if r.state is LedState.LIT]


def _sweep_flat(runner, scope, **overrides):
    """Sweep a field with nothing in it; every frame scores zero, and it raises the flat-curve failure."""
    with pytest.raises(AutofocusFailedError) as raised:
        sweep(runner, scope, **overrides)
    assert raised.value.reason == 'flat_focus_curve'


@pytest.fixture
def flat(sim_scope):
    """A runner over a white field, and its scope."""
    return af_runner_on(sim_scope, pattern='white'), sim_scope


class TestDegenerateCurveReportsFailure:
    def test_it_raises_and_is_reported_once(self, flat, centre_posts):
        """The failure is raised, told once as 'Autofocus Failed' in the
        flat curve's words, and no focus is stored."""
        runner, scope = flat

        _sweep_flat(runner, scope)

        errors = [(n.title, n.message) for n in centre_posts if n.severity is Severity.ERROR]
        assert errors == [('Autofocus Failed', str(AutofocusFailedError('flat_focus_curve')))], (
            f'a flat curve must be told once, as itself; got {errors}'
        )
        assert runner.best_focus_position() is None, 'a flat curve chose no focus'

    def test_complete_is_false(self, flat):
        """Nothing completed, so complete() must not say it did.

        This is also what re-points the Z readout: the step runner
        suppresses its own Z push only while complete() is True.
        """
        runner, scope = flat

        _sweep_flat(runner, scope)

        assert runner.complete() is False, (
            'a degenerate curve completed nothing; complete() must be False'
        )

    def test_the_stage_returns_to_the_pre_autofocus_z(self, flat, monkeypatch):
        """The whole point: a failed autofocus is a no-op in position, so
        a protocol step never captures at the sweep's end."""
        runner, scope = flat
        asked = z_asked(scope, monkeypatch)

        _sweep_flat(runner, scope)

        assert asked, 'the sweep must have moved the stage at all'
        assert asked[-1] == PARKED_Z_UM, (
            'the last Z move on a degenerate curve must be the restore to the '
            f'pre-autofocus position {PARKED_Z_UM}; got {asked[-1]} (all: {asked})'
        )
        assert settled_z(scope) == pytest.approx(PARKED_Z_UM, abs=0.01)

    def test_the_channel_led_is_handed_back_not_held(self, flat):
        """A failure, like every failure, restores the pre-autofocus LED
        state even when the step asked to keep the channel lit: the
        capture that follows is not the failed sweep's to light for, and
        re-lights its own channel."""
        runner, scope = flat
        assert _lit(scope) == [], 'precondition: nothing lit before the sweep'

        _sweep_flat(runner, scope, keep_led_on=True)

        assert _lit(scope) == [], (
            f'a degenerate curve must not hold the channel; lit: {_lit(scope)}'
        )

    def test_the_pre_autofocus_snapshot_field_is_gone(self, sim_scope):
        """One value, one store.

        The restore used to read `_saved_z_position`, a second snapshot of
        the same quantity taken fourteen lines after `_params['center']`,
        inside a try whose except swallowed the failure to a debug line
        and assigned None -- which silently switched the restore OFF. The
        field is deleted rather than guarded.
        """
        runner = AutofocusRunner(scope=sim_scope)

        assert not hasattr(runner, '_saved_z_position'), (
            'the duplicate pre-AF Z snapshot must be gone; the restore reads '
            "_params['center'], which cannot be missing"
        )


class TestTheSuccessPathIsUnchanged:
    def test_success_still_reports_and_leaves_the_stage_at_the_result(self, sim_scope, monkeypatch):
        """D6: the fix must not touch the path that works.

        A sweep over the focus target drives the success exit, whose last
        move is to the found position -- NOT to the pre-autofocus Z. It
        holds the channel when asked to, as the failure does not.
        """
        runner = af_runner_on(sim_scope)
        asked = z_asked(sim_scope, monkeypatch)

        result = sweep(runner, sim_scope, keep_led_on=True)

        assert result == pytest.approx(FOCAL_Z_UM, abs=10.0), (
            f'the success path must report the focus it found; got {result}'
        )
        assert runner.complete() is True, 'a completed autofocus must say so'
        assert asked[-1] == result, (
            f'success must leave the stage at the found position; got {asked}'
        )
        assert _lit(sim_scope) == ['BF'], (
            f'a success holds the channel asked for; lit: {_lit(sim_scope)}'
        )
