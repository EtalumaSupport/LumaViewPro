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


def test_lock_is_the_first_statement_inside_the_bracket():
    bracket = _the_bracket()
    first = bracket.body[0]
    assert isinstance(first, ast.Assign) and _calls(first, '_lock_auto_gain_impl'), (
        'the auto-gain lock must be the first statement inside the try, so '
        'every later exit reaches the resume in the finally'
    )


def test_restore_re_arms_on_every_exit_of_the_bracket(monkeypatch):
    """The re-arm rides the camera-state restore in the bracket's finally,
    so an abort exit puts a live-view arm back the same way a completion
    does. The producer's snapshot is always truthy (the save runs above
    the try and always carries its tag), so the restore branch cannot be
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
    scope.imaging._lock_auto_gain_impl.return_value = AutoGainLock(state=None)
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

    def _set_gain_db_impl(self, gain_db):
        self.writes.append(('gain', gain_db))

    def _set_exposure_ms_impl(self, exposure_ms):
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
