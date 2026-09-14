# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A failed autofocus SETUP leaves the runner able to run again.

`run()` claims the runner by setting `_af_in_progress`, and the try /
finally bracket below is what releases it. Between the two sat the run's
setup: the objective load, the parameter calculation, the AF START log
line and the two hardware snapshots -- five statements, every one of
which can raise.

Any of them raising latched the flag for the life of the process.
`reset()` refuses on that same flag, so it could not clear it, and every
later `run()` raised `RuntimeError('Autofocus already in progress')`
while `scope.imaging.is_focusing` answered True the whole time. One bad
objective config, one camera that did not answer, and autofocus was dead
until the app restarted.

The setup now runs inside the bracket, so nothing raise-capable remains
between claiming the run and the block that guarantees the claim is
released. Three consequences are asserted here rather than discovered:
the flag clears, the original exception reaches the caller unmasked by
anything in the finally, and the unwind does to the stage and the camera
exactly what the state reached so far justifies -- no more.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

_mock_settings_init = MagicMock()
_mock_settings_init.settings = {'BF': {'autofocus': False}, 'Green': {'autofocus': False}}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from tests.af_drives import AF_CENTER_Z, af_runner_and_scope, drive_af


class SetupError(Exception):
    """Distinct from anything the runner raises on its own, so a test that
    sees it knows the caller got the ORIGINAL failure."""


def _raise(*_args, **_kwargs):
    raise SetupError('the setup statement under test')


# The five raise-capable setup statements, in source order, each paired
# with the seam that makes it fail and with whether the scan parameters
# exist by the time it runs. The parameters are what the unwind's Z
# restore reads, so they decide what the unwind is entitled to do.
SETUP_STATEMENTS = [
    ('objective load', 'objective', False),
    ('parameter calculation', 'position_read', False),
    ('LED snapshot', 'led_snapshot', True),
    ('camera snapshot', 'camera_snapshot', True),
]

# The fifth setup statement, the AF START log line, is covered
# STRUCTURALLY below rather than by fault injection: `_af_log.info` is
# called again inside the finally, so a seam that breaks it breaks the
# unwind as well and would be measuring two things at once.
SETUP_CALLS_IN_SOURCE_ORDER = [
    'get_objective_info',
    '_calculate_params',
    'info',
    'save_led_state',
    'save_camera_state',
]


def _break(runner, scope, seam, monkeypatch):
    """Make one setup statement raise, leaving the other four intact."""
    if seam == 'objective':
        runner._objective_loader.get_objective_info.side_effect = _raise
    elif seam == 'position_read':
        # The scan centre is the first thing _calculate_params reads.
        scope.motion.get_current_position.side_effect = _raise
    elif seam == 'led_snapshot':
        scope.illumination.save_led_state.side_effect = _raise
    elif seam == 'camera_snapshot':
        scope.imaging.save_camera_state.side_effect = _raise
    else:
        raise AssertionError(f'unknown seam {seam}')


def _z_moves(scope):
    return [
        call.args[1]
        for call in scope.motion._move_absolute_impl.call_args_list
        if call.args and call.args[0] == 'Z'
    ]


@pytest.mark.parametrize(
    'label,seam,params_exist', SETUP_STATEMENTS, ids=[s[0] for s in SETUP_STATEMENTS]
)
class TestSetupFailureReleasesTheRunner:
    """D9: the window is the defect, so every statement in it is a case."""

    def test_the_runner_is_left_free_and_the_caller_gets_the_original_error(
        self, label, seam, params_exist, monkeypatch
    ):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 0.0)
        runner, scope = af_runner_and_scope()
        _break(runner, scope, seam, monkeypatch)

        with pytest.raises(SetupError):
            drive_af(runner)

        assert runner._af_in_progress.is_set() is False, (
            f'a raise from the {label} must not latch the in-progress flag'
        )
        assert scope.imaging.is_focusing is False, (
            f'a raise from the {label} must not leave is_focusing standing True'
        )
        assert runner.in_progress() is False, (
            f'a raise from the {label} must leave the public probe clear'
        )

    def test_a_later_run_can_still_start(self, label, seam, params_exist, monkeypatch):
        """The symptom the window actually produced: autofocus dead for the
        session. A healthy run after the failure must start normally."""
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        _break(runner, scope, seam, monkeypatch)

        with pytest.raises(SetupError):
            drive_af(runner)

        # Heal every seam, then run for real.
        runner._objective_loader.get_objective_info.side_effect = None
        scope.motion.get_current_position.side_effect = None
        scope.illumination.save_led_state.side_effect = None
        scope.imaging.save_camera_state.side_effect = None
        monkeypatch.undo()
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)

        assert drive_af(runner) == AF_CENTER_Z, (
            f'autofocus must still be usable after a raise from the {label}'
        )


class TestTheUnwindDoesOnlyWhatTheStateJustifies:
    """D10: `_params` is not a proxy for "nothing happened".

    Of the five setup statements, two run before the scan parameters
    exist and three run after. The unwind's Z restore reads those
    parameters, so it is skipped for the first two and FIRES for the last
    three. Firing is a real hardware write on a failure path: it is a
    no-op in VALUE -- the sweep's first move never ran, so the stage is
    still at the scan centre -- and it is asserted here rather than
    discovered on a bench.
    """

    @pytest.mark.parametrize('label,seam', [(s[0], s[1]) for s in SETUP_STATEMENTS if not s[2]])
    def test_before_the_parameters_exist_the_stage_is_not_touched(self, label, seam, monkeypatch):
        runner, scope = af_runner_and_scope()
        _break(runner, scope, seam, monkeypatch)

        with pytest.raises(SetupError):
            drive_af(runner)

        assert _z_moves(scope) == [], (
            f'a raise from the {label} happens before the scan centre is known, '
            'so the unwind has no position to restore to and must not move'
        )

    @pytest.mark.parametrize('label,seam', [(s[0], s[1]) for s in SETUP_STATEMENTS if s[2]])
    def test_after_the_parameters_exist_the_restore_fires_as_a_no_op_move(
        self, label, seam, monkeypatch
    ):
        runner, scope = af_runner_and_scope()
        _break(runner, scope, seam, monkeypatch)

        with pytest.raises(SetupError):
            drive_af(runner)

        assert _z_moves(scope) == [AF_CENTER_Z], (
            f'a raise from the {label} unwinds with the scan centre known, so the '
            'restore issues exactly one move -- to where the stage already is'
        )


class TestTheCameraArmIsPutBackWhenItWasTakenAndNotOtherwise:
    """D11: the invariant the auto-gain bracket test now states outright.

    The camera snapshot records a live-view auto-gain arm before the lock
    consumes it, and the restore in the finally is what puts it back. So
    the restore is owed exactly when the snapshot was taken -- including
    when the lock itself is what failed -- and owed nothing before that,
    because nothing has been consumed yet.
    """

    def test_a_failure_at_the_lock_still_restores_the_snapshot(self, monkeypatch):
        runner, scope = af_runner_and_scope()
        scope.imaging._lock_auto_gain_impl.side_effect = _raise

        with pytest.raises(SetupError):
            drive_af(runner)

        assert scope.imaging.restore_camera_state.called, (
            'the lock consumes a live-view arm; a raise there must still reach '
            'the restore that puts it back'
        )

    def test_a_failure_before_the_snapshot_restores_nothing(self, monkeypatch):
        runner, scope = af_runner_and_scope()
        _break(runner, scope, 'objective', monkeypatch)

        with pytest.raises(SetupError):
            drive_af(runner)

        assert not scope.imaging.restore_camera_state.called, (
            'nothing was snapshotted or consumed yet, so there is no camera state to write back'
        )


def test_every_setup_statement_sits_inside_the_bracket():
    """The window closes structurally, not statement by statement.

    Fault injection can only reach four of the five setup statements, and
    the defect was never about any one of them: it was that the region
    existed at all. This asserts the region is empty -- all five setup
    calls are inside the try whose finally releases the claim, and in the
    order the rest of the run depends on.
    """
    import ast

    from tests import ast_seams

    run = ast_seams.find_def('modules/autofocus_runner.py', 'run', class_name='AutofocusRunner')
    assert run is not None
    brackets = [n for n in run.body if isinstance(n, ast.Try) and n.finalbody]
    assert len(brackets) == 1, 'run() must have exactly one top-level try/finally bracket'
    bracket = brackets[0]

    def _attrs_called(nodes):
        return [
            n.func.attr
            for stmt in nodes
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]

    before_the_bracket = _attrs_called(run.body[: run.body.index(bracket)])
    for call in SETUP_CALLS_IN_SOURCE_ORDER:
        assert call not in before_the_bracket, (
            f'{call}() can raise, so above the bracket it latches the '
            'in-progress flag for the life of the process; it belongs inside'
        )

    inside = _attrs_called(bracket.body)
    positions = [inside.index(call) for call in SETUP_CALLS_IN_SOURCE_ORDER]
    assert positions == sorted(positions), (
        f'the setup calls must keep their source order inside the bracket; got {positions}'
    )


@pytest.mark.xfail(
    strict=True,
    reason='KNOWN OPEN: the unwind chain itself is unguarded -- see the class docstring',
)
def test_a_failure_during_the_unwind_does_not_latch_the_runner():
    """The same defect as the setup window, on the other side of the try,
    and NOT closed here.

    Moving the setup inside the bracket makes the finally the thing that
    guarantees the claim is released. But the finally's own restore chain
    is only partly guarded: the precision restore and the Z restore each
    sit in a try/except, while the LED transition, the camera restore and
    the diagnostic log line -- which reads gain and exposure off the
    camera inside its f-string -- do not. Any of them raising skips
    `_af_in_progress.clear()` at the end of the block, which is the exact
    session-killing symptom the setup move just closed: every later
    autofocus raises 'Autofocus already in progress' until the app
    restarts.

    This is reachable by a camera or USB failure during the restore, on a
    run that otherwise SUCCEEDED, and it predates this change -- verified
    against the pre-change tip, where all three seams behave identically.

    Marked strict so closing it turns this test red rather than letting
    it be closed silently.
    """
    runner, scope = af_runner_and_scope()
    scope.imaging.restore_camera_state.side_effect = SetupError('camera went away mid-restore')

    with pytest.raises(SetupError):
        drive_af(runner)

    assert runner._af_in_progress.is_set() is False, (
        'a raise in the unwind must not latch the runner for the session'
    )
