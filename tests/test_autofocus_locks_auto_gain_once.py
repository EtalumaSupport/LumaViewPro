# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus sweep locks a standing auto-gain arm once, at its own
camera-state bracket, and resumes it on every exit.

Bug shape: a capture under a continuous auto-gain arm locks the arm and,
for a live-view arm, re-arms afterwards. A sweep of forty positions would
pay that lock and the twenty-frame re-arm settle at every position, and
the step's gain and exposure the sweep is asked to scan at would be
overridden by the still-running loop. The runner already saves the camera
state before the sweep and restores it in a finally; the lock rides the
same bracket, so there is one lock, one resume, and no exit that skips the
resume.
"""

from __future__ import annotations

import ast
import contextlib
import types

import pytest

from modules.exceptions import AutofocusAborted

from tests import ast_seams
from tests.test_auto_gain_lock import AG_SETTINGS_TRANSMITTED, _arm, _build

AF_RUNNER = 'modules/autofocus_runner.py'


def _calls(node: ast.AST, attr: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


def _the_bracket() -> ast.Try:
    run = ast_seams.find_def(AF_RUNNER, 'run', class_name='AutofocusRunner')
    assert run is not None
    tries = [n for n in run.body if isinstance(n, ast.Try) and n.finalbody]
    assert len(tries) == 1, 'run() must have exactly one top-level try/finally bracket'
    return tries[0]


def _index_of(bracket: ast.Try, attr: str) -> int:
    """Position in the bracket's body of the statement calling attr."""
    hits = [i for i, stmt in enumerate(bracket.body) if _calls(stmt, attr)]
    assert len(hits) == 1, f'expected exactly one {attr} call in the bracket; got {hits}'
    return hits[0]


def test_the_lock_is_inside_the_bracket_and_the_camera_snapshot_precedes_it():
    """Two invariants, both of which the file depends on.

    The lock must be INSIDE the try, so every later exit reaches the
    resume in the finally. It used to be the try's FIRST statement, and
    that position encoded the invariant by accident; the run's setup --
    the objective load, the parameter calculation and the two hardware
    snapshots -- has since moved inside the bracket too, because above it
    any of them could raise and latch the in-progress flag for the life
    of the process.

    The snapshot must PRECEDE the lock: it records a live-view auto-gain
    arm before the lock consumes it, and the restore in the finally is
    what puts that arm back. Moving the snapshot after the lock would
    break the re-arm, which is the ordering the position used to protect
    and this now states outright.
    """
    bracket = _the_bracket()
    lock_at = _index_of(bracket, 'lock_auto_gain')
    snapshot_at = _index_of(bracket, 'save_camera_state')
    assert isinstance(bracket.body[lock_at], ast.Assign), (
        'the lock must be bound to a name so the finally can resume it'
    )
    assert snapshot_at < lock_at, (
        'the pre-AF camera snapshot must be taken BEFORE the auto-gain lock '
        f'consumes the arm; snapshot at body[{snapshot_at}], lock at body[{lock_at}]'
    )


def test_restore_re_arms_on_every_exit_of_the_bracket(monkeypatch):
    """The re-arm rides the camera-state restore in the bracket's finally,
    so an abort exit puts a live-view arm back the same way a completion
    does. The snapshot is taken at the top of the try and always carries
    its tag, so once it has been taken the restore branch cannot be
    skipped on any exit. A preservation guard on the exit path; the
    re-arm itself is proven on the API in test_auto_gain_lock."""
    from modules.lumascope_api.imaging import AutoGainLock, _AutoGainArm
    from tests.af_drives import af_runner_and_scope, drive_af

    arm = _AutoGainArm(dict(AG_SETTINGS_TRANSMITTED), True)
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 0.0)
    runner, scope = af_runner_and_scope()
    scope.imaging.save_camera_state.return_value = {
        'tag': 'autofocus',
        'gain_db': 1.0,
        'exposure_ms': 10.0,
        'auto_gain_arm': arm,
    }
    scope.imaging.lock_auto_gain.return_value = AutoGainLock(state=None)
    with contextlib.suppress(AutofocusAborted):
        drive_af(runner)
    restored = scope.imaging.restore_camera_state.call_args.args[0]
    assert restored['auto_gain_arm'] is arm
    assert scope.imaging._resume_auto_gain_impl.call_count == 0


def test_lock_and_resume_pair_on_the_api():
    """One lock before N captures and one resume after: no capture inside
    the bracket locks on its own, and the live-view arm is back at the end."""
    imaging, cam = _build(ae_lands_on_ms=8.0)
    _arm(imaging, AG_SETTINGS_TRANSMITTED, resume_after_capture=True)
    lock = imaging._lock_auto_gain_impl()
    assert lock.state == lock.state.CONVERGED
    assert cam._auto_gain_enabled is False
    for _ in range(3):
        assert imaging._capture_and_wait_impl(timeout_s=1.0) is not None
        assert 'auto_gain' not in (imaging.last_capture_info or {}), (
            'a capture inside the bracket locked on its own'
        )
        assert cam._auto_gain_enabled is False
    imaging._resume_auto_gain_impl(lock)
    assert cam._auto_gain_enabled is True
    assert imaging._auto_gain_arm is not None


class _RecordingImaging:
    """Records the setter writes the sweep's target step makes."""

    def __init__(self):
        self.writes = []

    def set_gain_db(self, gain_db):
        self.writes.append(('gain', gain_db))

    def set_exposure_ms(self, exposure_ms):
        self.writes.append(('exposure', exposure_ms))


def _runner_with_step_targets(gain_db, exposure_ms):
    from modules.autofocus_runner import AutofocusRunner

    runner = AutofocusRunner.__new__(AutofocusRunner)
    runner._scope = types.SimpleNamespace(imaging=_RecordingImaging())
    runner._camera_gain = gain_db
    runner._camera_exposure = exposure_ms
    return runner


def test_sweep_scans_at_the_locks_values():
    """A sweep under a live-view arm scans at the exposure and gain the
    lock just read from the camera; the step's stored values are stale by
    construction under an arm (the slider poll reads a cache the arm
    invalidates), and writing them over the lock scanned a dark field --
    the first bench sweep ran at 1.0 dB / 2 ms on a scene the loop had
    settled at 6.8 dB / 50 ms. With no arm, or a FAILED lock carrying no
    values, the step's values are written as before."""
    from modules.lumascope_api.imaging import AutoGainConvergence, AutoGainLock

    runner = _runner_with_step_targets(1.0, 2.0)
    runner._apply_sweep_camera_targets(
        AutoGainLock(AutoGainConvergence.MAXED, exposure_ms=50.0, gain_db=6.72)
    )
    assert runner._scope.imaging.writes == []
    assert (runner._camera_gain, runner._camera_exposure) == (6.72, 50.0)

    runner = _runner_with_step_targets(1.0, 2.0)
    runner._apply_sweep_camera_targets(AutoGainLock(state=None))
    assert runner._scope.imaging.writes == [('gain', 1.0), ('exposure', 2.0)]
    assert (runner._camera_gain, runner._camera_exposure) == (1.0, 2.0)

    runner = _runner_with_step_targets(1.0, 2.0)
    runner._apply_sweep_camera_targets(AutoGainLock(AutoGainConvergence.FAILED))
    assert runner._scope.imaging.writes == [('gain', 1.0), ('exposure', 2.0)]


def test_half_populated_lock_is_unconstructible():
    """A lock carries both achieved values or neither: a lock with one of
    them would let the sweep keep the lock's gain and the snapshot's
    exposure, a mixed camera state nothing asked for."""
    from modules.lumascope_api.imaging import AutoGainConvergence, AutoGainLock

    with pytest.raises(ValueError):
        AutoGainLock(AutoGainConvergence.CONVERGED, exposure_ms=5.0)
    with pytest.raises(ValueError):
        AutoGainLock(AutoGainConvergence.CONVERGED, gain_db=5.0)
