# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus with no usable focus score is a FAILURE, and reports like one.

The degenerate exit once assigned the PRE-AUTOFOCUS Z into the result and
set the completion event, so a curve that found nothing reported a float
and claimed to have completed; a step then captured wherever the sweep
ended. Later it reported its own failure and returned None, a second exit
beside ``run()``'s failure path, and a single-autofocus run read that None
as a run that completed.

A sweep whose every frame scores zero has no usable score (one finite and
above zero) and raises ``AutofocusFailedError('no_focus_scores')``. It
leaves through ``run()``'s one failure path, like every other failure: it
is reported once, the stage returns to where the sweep started,
``complete()`` is False, and the channel LED is handed back rather than
held, since the capture that follows is not the failed sweep's to light.
A curve with usable scores is never refused for its shape
(``tests/test_autofocus_focuses_where_its_final_pass_peaks.py``).
"""

from __future__ import annotations

import pytest

from _lumascope.modules.autofocus_runner import AutofocusRunner
from _lumascope.modules.exceptions import AutofocusFailedError
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.notification_center import Severity
from tests.af_drives import (
    FOCAL_Z_UM,
    PARKED_Z_UM,
    af_runner_on,
    settled_z,
    sweep,
    z_asked,
)


def _lit(scope):
    return [r.name for r in scope.illumination.led_states if r.state is LedState.LIT]


def _sweep_blank(runner, scope, **overrides):
    """Sweep a field with nothing in it; every frame scores zero, and it raises the no-usable-score failure."""
    with pytest.raises(AutofocusFailedError) as raised:
        sweep(runner, scope, **overrides)
    assert raised.value.reason == 'no_focus_scores'


@pytest.fixture
def blank(sim_scope):
    """A runner over a white field, and its scope."""
    return af_runner_on(sim_scope, pattern='white'), sim_scope


class TestNoUsableScoreReportsFailure:
    def test_it_raises_and_is_reported_once(self, blank, centre_posts):
        """The failure is raised, told once as 'Autofocus Failed' in the
        no-usable-score words, and no focus is stored."""
        runner, scope = blank

        _sweep_blank(runner, scope)

        errors = [(n.title, n.message) for n in centre_posts if n.severity is Severity.ERROR]
        assert errors == [('Autofocus Failed', str(AutofocusFailedError('no_focus_scores')))], (
            f'a sweep with no usable score must be told once, as itself; got {errors}'
        )
        assert runner.best_focus_position() is None, 'a sweep with no usable score chose no focus'

    def test_complete_is_false(self, blank):
        """Nothing completed, so complete() must not say it did.

        This is also what re-points the Z readout: the step runner
        suppresses its own Z push only while complete() is True.
        """
        runner, scope = blank

        _sweep_blank(runner, scope)

        assert runner.complete() is False, (
            'a degenerate curve completed nothing; complete() must be False'
        )

    def test_the_stage_returns_to_the_pre_autofocus_z(self, blank, monkeypatch):
        """The whole point: a failed autofocus is a no-op in position, so
        a protocol step never captures at the sweep's end."""
        runner, scope = blank
        asked = z_asked(scope, monkeypatch)

        _sweep_blank(runner, scope)

        assert asked, 'the sweep must have moved the stage at all'
        assert asked[-1] == PARKED_Z_UM, (
            'the last Z move on a degenerate curve must be the restore to the '
            f'pre-autofocus position {PARKED_Z_UM}; got {asked[-1]} (all: {asked})'
        )
        assert settled_z(scope) == pytest.approx(PARKED_Z_UM, abs=0.01)

    def test_a_sweep_centred_away_from_the_stage_returns_it_where_it_stood(
        self, blank, monkeypatch
    ):
        """A z-stack's sweep starts at its first slice, centred on its reference
        slice; a failure puts the stage back at the slice, not at the centre."""
        runner, scope = blank
        asked = z_asked(scope, monkeypatch)

        _sweep_blank(runner, scope, center_um=PARKED_Z_UM + 100.0)

        assert asked[-1] == PARKED_Z_UM, (
            f'a failed sweep centred at {PARKED_Z_UM + 100.0} must restore the Z '
            f'it found, {PARKED_Z_UM}; got {asked[-1]} (all: {asked})'
        )

    def test_the_channel_led_is_handed_back_not_held(self, blank):
        """A failure, like every failure, restores the pre-autofocus LED
        state even when the step asked to keep the channel lit: the
        capture that follows is not the failed sweep's to light for, and
        re-lights its own channel."""
        runner, scope = blank
        assert _lit(scope) == [], 'precondition: nothing lit before the sweep'

        _sweep_blank(runner, scope, keep_led_on=True)

        assert _lit(scope) == [], (
            f'a degenerate curve must not hold the channel; lit: {_lit(scope)}'
        )

    def test_the_pre_autofocus_snapshot_field_is_gone(self, sim_scope):
        """One value, one store.

        The restore used to read `_saved_z_position`, a second snapshot of
        the pre-autofocus Z taken fourteen lines after the sweep's
        parameters, inside a try whose except swallowed the failure to a
        debug line and assigned None -- which silently switched the restore
        OFF. The field is deleted rather than guarded.
        """
        runner = AutofocusRunner(scope=sim_scope)

        assert not hasattr(runner, '_saved_z_position'), (
            'the duplicate pre-AF Z snapshot must be gone; the restore reads '
            "_params['found_z'], which cannot be missing"
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
