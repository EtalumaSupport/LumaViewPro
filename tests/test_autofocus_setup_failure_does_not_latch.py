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
anything in the finally (raised as itself, as every API failure is, or
chained as the cause of the one failure the run reports and raises), and
the unwind does to the stage and the camera
exactly what the state reached so far justifies -- no more.
"""

from __future__ import annotations

import pytest

from modules.exceptions import (
    AutofocusAborted,
    CameraReadFailedError,
    CaptureError,
    CatalogueNameRefusedError,
)
from modules.lumascope_api.camera import CameraAutoMode
from modules.lumascope_api.illumination import LedLease, LedState, LedTransition
from tests.af_drives import (
    FOCAL_Z_UM,
    PARKED_Z_UM,
    af_runner_on,
    park_z,
    raises_the_failure_from,
    sweep,
    z_asked,
)
from tests.protocol_drives import held_run_claim
from tests.test_auto_gain_lock import AG_SETTINGS_TRANSMITTED


class SetupError(Exception):
    """Distinct from anything the runner raises on its own, so a test that
    sees it knows the caller got the ORIGINAL failure."""


@pytest.fixture
def scope(sim_scope):
    """The simulated scope; ``runner`` makes it one a sweep can run on."""
    return sim_scope


@pytest.fixture
def runner(scope):
    return af_runner_on(scope)


def _unknown_objective(scope, sweep_kwargs):
    sweep_kwargs['objective_id'] = 'not an objective'


def _z_at_the_floor(scope, sweep_kwargs):
    # The first window, centre +- AF_range, reaches below the floor.
    park_z(scope, scope.motion.get_axis_limits('Z').min_um)


def _camera_setting_unreadable(scope, sweep_kwargs):
    scope.simulation.refuse_camera_read('frame_size', 'failed')


# The raise-capable setup statements, in source order, each with a real
# failure of it, the failure it raises, and whether the scan parameters
# exist by the time it runs. The parameters are what the unwind's Z
# restore reads, so they decide what the unwind is entitled to do. The LED
# snapshot raises nothing (``save_led_state``: Raises: Nothing), so it has
# no case; the parameter calculation's raise is the first window past travel.
SETUP_FAILURES = [
    (
        'objective load',
        _unknown_objective,
        CatalogueNameRefusedError,
        'objective_not_in_catalogue',
        False,
    ),
    ('parameter calculation', _z_at_the_floor, CaptureError, 'out_of_travel', False),
    (
        'camera snapshot',
        _camera_setting_unreadable,
        CameraReadFailedError,
        'camera_read_failed',
        True,
    ),
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


def _fail_setup(scope, runner, breaker, raised_type, reason):
    """Sweep with one setup statement failing; it raises that statement's failure, as itself."""
    sweep_kwargs = {}
    breaker(scope, sweep_kwargs)
    with pytest.raises(raised_type) as raised:
        sweep(runner, scope, **sweep_kwargs)
    assert raised.value.reason == reason


def _live_view_arm(scope):
    """A standing live-view auto-gain arm: the one thing only the camera restore puts back."""
    scope.camera.set_auto_gain(True, dict(AG_SETTINGS_TRANSMITTED), resume_after_capture=True)
    assert scope.camera.auto_gain is CameraAutoMode.CONTINUOUS


def _while_sweeping(scope, monkeypatch, then):
    """Run ``then`` once, at the sweep's first scored frame: after the setup's writes."""
    import modules.autofocus_functions as autofocus_functions

    score = autofocus_functions.focus_function
    done = []

    def scoring(image):
        if not done:
            done.append(True)
            then()
        return score(image)

    monkeypatch.setattr('modules.autofocus_functions.focus_function', scoring)


def _camera_restores(scope, monkeypatch) -> list:
    """Every camera-state restore asked of the scope: the exception it
    raised, or None when it returned."""
    asked = []
    real = scope.camera.restore_camera_state

    def spy(snapshot):
        try:
            result = real(snapshot)
        except Exception as raised:
            asked.append(raised)
            raise
        asked.append(None)
        return result

    monkeypatch.setattr(scope.camera, 'restore_camera_state', spy)
    return asked


def _focused(result) -> bool:
    return result == pytest.approx(FOCAL_Z_UM, abs=10.0)


@pytest.mark.parametrize(
    'label,breaker,raised_type,reason,params_exist',
    SETUP_FAILURES,
    ids=[s[0] for s in SETUP_FAILURES],
)
class TestSetupFailureReleasesTheRunner:
    """D9: the window is the defect, so every statement in it is a case."""

    def test_the_runner_is_left_free_and_the_caller_gets_the_original_error(
        self, scope, runner, label, breaker, raised_type, reason, params_exist
    ):
        _fail_setup(scope, runner, breaker, raised_type, reason)

        assert scope.imaging.is_focusing is False, (
            f'a raise from the {label} must not leave is_focusing standing True'
        )
        assert runner.in_progress() is False, (
            f'a raise from the {label} must leave the public probe clear'
        )

    def test_a_later_run_can_still_start(
        self, scope, runner, label, breaker, raised_type, reason, params_exist
    ):
        """The symptom the window actually produced: autofocus dead for the
        session. A healthy run after the failure must start normally."""
        _fail_setup(scope, runner, breaker, raised_type, reason)
        park_z(scope, PARKED_Z_UM)

        assert _focused(sweep(runner, scope)), (
            f'autofocus must still be usable after a raise from the {label}'
        )


class TestTheUnwindDoesOnlyWhatTheStateJustifies:
    """D10: `_params` is not a proxy for "nothing happened".

    Of the setup statements, the first two run before the scan parameters
    exist and the camera snapshot after. The unwind's Z restore reads those
    parameters, so it is skipped for the first two and FIRES for the last.
    Firing is a real hardware write on a failure path: it is a no-op in
    VALUE -- the sweep's first move never ran, so the stage is still at the
    scan centre -- and it is asserted here rather than discovered on a
    bench.
    """

    # parametrize-justified: each case breaks a different statement on a fresh runner
    @pytest.mark.parametrize(
        'label,breaker,raised_type,reason',
        [s[:4] for s in SETUP_FAILURES if not s[4]],
        ids=[s[0] for s in SETUP_FAILURES if not s[4]],
    )
    def test_before_the_parameters_exist_the_stage_is_not_touched(
        self, scope, runner, monkeypatch, label, breaker, raised_type, reason
    ):
        sweep_kwargs = {}
        breaker(scope, sweep_kwargs)
        asked = z_asked(scope, monkeypatch)

        with pytest.raises(raised_type):
            sweep(runner, scope, **sweep_kwargs)

        assert asked == [], (
            f'a raise from the {label} happens before the scan centre is known, '
            'so the unwind has no position to restore to and must not move'
        )

    def test_after_the_parameters_exist_the_restore_fires_as_a_no_op_move(
        self, scope, runner, monkeypatch
    ):
        asked = z_asked(scope, monkeypatch)

        _fail_setup(
            scope, runner, _camera_setting_unreadable, CameraReadFailedError, 'camera_read_failed'
        )

        assert asked == [PARKED_Z_UM], (
            'a raise from the camera snapshot unwinds with the scan centre known, so '
            'the restore issues exactly one move -- to where the stage already is'
        )


def test_a_failure_before_the_snapshot_restores_no_camera_state(scope, runner, monkeypatch):
    """D11: the restore is owed exactly when the snapshot was taken.

    The camera snapshot records a live-view auto-gain arm before the lock
    consumes it, and the restore in the finally is what puts it back; before
    the snapshot nothing has been consumed, so there is nothing to write back.
    """
    restores = _camera_restores(scope, monkeypatch)

    _fail_setup(
        scope, runner, _unknown_objective, CatalogueNameRefusedError, 'objective_not_in_catalogue'
    )

    assert restores == [], (
        'nothing was snapshotted or consumed yet, so no camera state is written back'
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


def _camera_restore_refused(scope):
    # The restore reads the frame size to compare with the snapshot's: a
    # read the camera fails raises from the restore.
    scope.simulation.refuse_camera_read('frame_size', 'failed')


def _the_restore_raised(restores) -> bool:
    return bool(restores) and isinstance(restores[-1], CameraReadFailedError)


def _af_end_light_refused(monkeypatch):
    """The AF lease's end transition raises, once; its start transition runs.

    Breaking ``apply`` wholesale breaks AF_ENTER first, which runs inside
    the try during setup -- that measures the setup window, not the
    unwind. The fast simulated LED board takes no refused-write fault (the
    FX2's ``refuse_led_write`` is that fault), so it is raised at the lease.
    """
    real = LedLease.apply
    raised = []

    def apply(lease, transition, ctx):
        if transition is LedTransition.AF_TO_CAPTURE and not raised:
            raised.append(transition)
            raise SetupError('usb')
        return real(lease, transition, ctx)

    monkeypatch.setattr(LedLease, 'apply', apply)


def test_a_failure_during_the_unwind_does_not_latch_the_runner(scope, runner, monkeypatch):
    """D12 + D15: a cleanup failure costs the run neither its claim nor its
    answer.

    This test asserted the opposite one commit ago, deliberately. The
    release moved into its own nested finally first, which stopped the
    runner being latched but still let the cleanup's exception stand in
    for the run's outcome. Guarding the restore steps finishes the job:
    the run that found a focus reports it, and the camera failure is in
    the log where a cleanup failure belongs.
    """
    restores = _camera_restores(scope, monkeypatch)
    _while_sweeping(scope, monkeypatch, lambda: _camera_restore_refused(scope))

    result = sweep(runner, scope)

    assert _the_restore_raised(restores), f'precondition: the camera restore failed; {restores}'
    assert _focused(result), (
        "a cleanup failure must not become the run's answer; the sweep found "
        f'a focus and must report it, got {result}'
    )
    assert runner.in_progress() is False, (
        'a raise in the unwind must not latch the runner for the session'
    )


UNWIND_SEAMS = ['LED transition', 'camera restore']


def _break_unwind(scope, seam, monkeypatch):
    """Break one UNWIND step, and only in the unwind."""
    if seam == 'LED transition':
        _af_end_light_refused(monkeypatch)
    else:
        restores = _camera_restores(scope, monkeypatch)
        _while_sweeping(scope, monkeypatch, lambda: _camera_restore_refused(scope))
        return restores


def _the_af_lease_is_released(scope, run_lease) -> bool:
    """The run's lease is the top one again: the AF child under it was released."""
    return scope.illumination.led_lease_purpose == run_lease.purpose


@pytest.mark.parametrize('seam', UNWIND_SEAMS)
class TestEveryUnwindFailureStillReleasesTheClaim:
    """D13: three things latched, not one, and all three must clear.

    The in-progress flag is the one that refuses the next run outright.
    The public `is_focusing` mirror, left True, suppresses every live
    camera apply the UI makes. And the LED lease, never released, makes
    illumination authority permanently unclaimable -- because its
    liveness probe is the very flag that stayed set.
    """

    def test_the_run_succeeded_and_the_runner_is_left_free(self, scope, runner, monkeypatch, seam):
        restores = _break_unwind(scope, seam, monkeypatch)
        run_lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())

        assert _focused(sweep(runner, scope, led_lease=run_lease)), (
            f'a {seam} failure is cleanup, not the run; the run must still '
            'report the focus it found'
        )
        assert restores is None or _the_restore_raised(restores), (
            f'precondition: the camera restore failed; {restores}'
        )

        assert runner.in_progress() is False, (
            f'a raise from the {seam} must not latch the in-progress probe'
        )
        assert scope.imaging.is_focusing is False, (
            f'a raise from the {seam} must not leave is_focusing standing True, '
            'which suppresses every live camera apply the UI makes'
        )
        assert _the_af_lease_is_released(scope, run_lease), (
            f'a raise from the {seam} must still release the LED lease; an '
            'un-released lease makes illumination authority unclaimable; '
            f'top lease: {scope.illumination.led_lease_purpose}'
        )
        run_lease.release()

    @pytest.mark.slow
    def test_a_later_run_can_still_start(self, scope, runner, monkeypatch, seam):
        _break_unwind(scope, seam, monkeypatch)
        sweep(runner, scope)
        park_z(scope, PARKED_Z_UM)

        assert _focused(sweep(runner, scope)), (
            f'autofocus must still be usable after a raise from the {seam}'
        )


def test_the_release_is_structurally_unreachable_past():
    """D14: the guarantee is the shape, not the three seams tested above.

    Guarding the statements that can raise today would leave the next
    restore statement to be written wrong again. The release lives in its
    own finally, so nothing placed in the restore chain can skip it.
    """
    import ast

    from tests import ast_seams

    run = ast_seams.find_def('modules/autofocus_runner.py', 'run', class_name='AutofocusRunner')
    assert run is not None
    outer = [n for n in run.body if isinstance(n, ast.Try) and n.finalbody]
    assert len(outer) == 1, 'run() must have exactly one top-level try/finally bracket'

    inner = [n for n in outer[0].finalbody if isinstance(n, ast.Try) and n.finalbody]
    assert len(inner) == 1, (
        "the bracket's finally must nest a try/finally, so the release cannot "
        'be skipped by a raise in the restore chain'
    )

    released = [
        n.func.attr
        for stmt in inner[0].finalbody
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert 'clear' in released, 'the in-progress flag must clear in the inner finally'
    assert 'release' in released, 'the LED lease must be released in the inner finally'

    restore_attrs = [
        n.func.attr
        for stmt in inner[0].body
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert 'restore_camera_state' in restore_attrs, (
        'the restore chain belongs in the inner try, not beside the release'
    )


class RunError(Exception):
    """Raised from the run BODY, so a test can tell the run's own failure
    apart from the cleanup's."""


class TestTheRunsOwnOutcomeSurvivesACleanupFailure:
    """D16: a cleanup failure never stands in for the run's answer.

    A raise inside a finally REPLACES the exception in flight. That is how
    a camera that vanished during cleanup used to erase the reason the run
    actually failed -- including an abort the caller had asked for, which
    is the common path, not an exotic one.
    """

    def test_the_runs_own_exception_reaches_the_caller(self, scope, runner, monkeypatch):
        restores = _camera_restores(scope, monkeypatch)

        def fail_the_run():
            _camera_restore_refused(scope)
            raise RunError('boom')

        _while_sweeping(scope, monkeypatch, fail_the_run)

        with raises_the_failure_from(RunError):
            sweep(runner, scope)
        assert _the_restore_raised(restores), f'precondition: the camera restore failed; {restores}'

    def test_an_abort_is_still_reported_as_an_abort(self, scope, runner, monkeypatch):
        """The common case: the caller stopped the run, and the cleanup
        then failed. The caller must hear about their own abort."""
        import threading

        abort = threading.Event()
        restores = _camera_restores(scope, monkeypatch)

        def stop_the_sweep():
            # The caller stops the sweep during its first pass, before it
            # chose any focus.
            _camera_restore_refused(scope)
            abort.set()

        _while_sweeping(scope, monkeypatch, stop_the_sweep)
        run_lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())

        with pytest.raises(AutofocusAborted):
            sweep(runner, scope, abort_event=abort, led_lease=run_lease)

        assert _the_restore_raised(restores), f'precondition: the camera restore failed; {restores}'
        assert runner.in_progress() is False
        assert _the_af_lease_is_released(scope, run_lease)
        run_lease.release()


def test_an_led_failure_still_re_arms_the_live_view(scope, runner, monkeypatch):
    """D17: the harm that made this change worth building.

    The auto-gain lock CONSUMES a live-view arm and only the camera
    restore puts it back. The LED transition runs first, so before this
    change an LED failure cost the user their live-view auto gain with no
    message -- the camera restore was never reached at all.
    """
    _live_view_arm(scope)
    _af_end_light_refused(monkeypatch)

    assert _focused(sweep(runner, scope))

    assert scope.camera.auto_gain is CameraAutoMode.CONTINUOUS, (
        'the camera restore is the ONLY thing that re-arms a live-view auto-gain '
        'arm; an LED failure ahead of it must not cost the user that arm'
    )


def test_a_clean_exit_still_runs_every_restore_step(scope, runner, monkeypatch):
    """D18: the preservation lock.

    Guards that swallow are indistinguishable from steps that never ran,
    unless something asserts the steps DO run when nothing is broken.
    """
    precision = []
    real = scope.motion.set_precision_mode

    def spy(axis, enabled):
        precision.append((axis, enabled))
        return real(axis, enabled)

    monkeypatch.setattr(scope.motion, 'set_precision_mode', spy)
    _live_view_arm(scope)

    assert _focused(sweep(runner, scope, keep_led_on=True))

    assert precision[-1] == ('Z', True), f'the precision restore must still run; {precision}'
    lit = [r.name for r in scope.illumination.get_led_states() if r.state is LedState.LIT]
    assert lit == ['BF'], f'the AF-end LED transition must still run; lit: {lit}'
    assert scope.camera.auto_gain is CameraAutoMode.CONTINUOUS, 'the camera restore must still run'


# The one restore statement that is deliberately NOT guarded, and the proof
# that it needs no guard: its only calls are the two camera getters, which
# route through _live_validated_read -- a try that catches every driver
# exception and answers from the last-known-good cache -- plus a locked
# attribute read. Guarding it would be dead code. Any OTHER unguarded
# call-bearing statement in the chain is the N+1 defect and fails the build.
_UNGUARDABLE_BY_PROOF = '[AF DIAG] Clearing _af_in_progress'


def test_no_restore_step_can_be_added_unguarded():
    """D19: the N+1 lock.

    Per-step guards fix the steps that exist today and leave the seventh
    one, whenever someone adds it, to be written unguarded -- where it
    would again replace the run's outcome with the cleanup's. This makes
    that a build failure instead of a customer's failure.

    The release block is exempt by construction: it is the nested
    try/finally the restore chain lives inside, pinned separately by the
    structural test above.
    """
    import ast

    from tests import ast_seams

    run = ast_seams.find_def('modules/autofocus_runner.py', 'run', class_name='AutofocusRunner')
    assert run is not None
    outer = [n for n in run.body if isinstance(n, ast.Try) and n.finalbody]
    assert len(outer) == 1
    inner = [n for n in outer[0].finalbody if isinstance(n, ast.Try) and n.finalbody]
    assert len(inner) == 1, 'the release must still be the inner finally'

    unguarded = []
    exempted = 0
    for stmt in inner[0].body:
        if isinstance(stmt, ast.Try):
            continue
        if any(isinstance(node, ast.Try) for node in ast.walk(stmt)):
            continue
        if not any(isinstance(node, ast.Call) for node in ast.walk(stmt)):
            continue
        unparsed = ast.unparse(stmt)
        if _UNGUARDABLE_BY_PROOF in unparsed:
            exempted += 1
            continue
        unguarded.append(unparsed.splitlines()[0][:70])

    assert not unguarded, (
        'every call-bearing statement in the restore chain must sit inside a '
        "try/except, or a cleanup failure becomes the run's answer. Add the "
        'guard, or add an exemption here with the proof that the statement '
        f'cannot raise. Unguarded: {unguarded}'
    )
    # The exemption is only safe while the statement it names is still there
    # and still unguarded. If it is renamed, deleted or wrapped, the exemption
    # stops matching -- and would silently excuse nothing, or something else
    # later. Asserted from the same walk rather than by reading the source.
    assert exempted == 1, (
        f'the exemption names {_UNGUARDABLE_BY_PROOF!r}, which must match exactly '
        f'one unguarded statement in the restore chain; matched {exempted}. '
        'Update the exemption and its proof together.'
    )
