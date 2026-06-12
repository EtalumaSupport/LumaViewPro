# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: AutofocusRunner mirrors AF lifecycle to scope.imaging.is_focusing.

Bug
---
ImagingAPI exposes is_focusing (property + setter) that reads/writes
self._focusing_event, but NO caller ever set or read it.
AutofocusRunner kept the real "AF in flight" state on its private
_af_in_progress (cleared LAST in the finally block, after camera/LED/Z
restore). External callers asking scope.imaging.is_focusing got False
during a live autofocus run. Rule-35 semantic-duplicate audit
2026-05-19, finding 4.

The fix mirrors _af_in_progress, not _is_focusing_event, because
_is_focusing_event clears mid-flight (before camera/LED/Z restore)
while _af_in_progress clears at the END of the finally block -- the
public surface must stay True until the scope is genuinely safe to
use again.

Tests drive the real run() via tests/af_drives.py and observe the
public flag from inside and after the run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.af_drives import af_runner_and_scope, drive_af


class _FocusFlagRecorder:
    """scope.imaging stand-in that records every is_focusing write
    together with whether the runner's private _af_in_progress gate was
    still set at that moment. All other attribute traffic delegates to
    a MagicMock."""

    def __init__(self):
        object.__setattr__(self, 'inner', MagicMock())
        object.__setattr__(self, 'writes', [])
        object.__setattr__(self, 'runner_ref', [])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)

    def __setattr__(self, name, value):
        if name == 'is_focusing':
            runner_ref = object.__getattribute__(self, 'runner_ref')
            in_progress = runner_ref[0]._af_in_progress.is_set() if runner_ref else None
            object.__getattribute__(self, 'writes').append((value, in_progress))
        setattr(object.__getattribute__(self, 'inner'), name, value)


def _drive_with_recorder(monkeypatch, focus_fn):
    monkeypatch.setattr('modules.autofocus_functions.focus_function', focus_fn)
    runner, scope = af_runner_and_scope()
    recorder = _FocusFlagRecorder()
    recorder.inner.save_camera_state.return_value = {'gain_db': 1.0, 'exposure_ms': 10.0}
    recorder.inner.capture_and_wait.return_value = scope.imaging.capture_and_wait.return_value
    scope.imaging = recorder
    recorder.runner_ref.append(runner)
    drive_af(runner)
    return scope, recorder


def test_is_focusing_true_while_af_runs(monkeypatch):
    """During the scan loop, the public flag reads True."""
    seen = []

    runner_scope = []

    def scoring(image):
        seen.append(runner_scope[0].imaging.is_focusing)
        return 7.0

    monkeypatch.setattr('modules.autofocus_functions.focus_function', scoring)
    runner, scope = af_runner_and_scope()
    runner_scope.append(scope)
    drive_af(runner)
    assert seen and all(value is True for value in seen), (
        'scope.imaging.is_focusing must read True while the AF loop is '
        f'scoring frames; observed {seen}'
    )


def test_is_focusing_false_after_run_completes(monkeypatch):
    """After run() returns, the public flag reads False."""
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
    runner, scope = af_runner_and_scope()
    drive_af(runner)
    assert scope.imaging.is_focusing is False, (
        'scope.imaging.is_focusing must be False once run() has returned'
    )


def test_false_flip_waits_for_af_in_progress_clear(monkeypatch):
    """The False write lands only after the private _af_in_progress gate
    has cleared -- i.e. after camera/LED/Z restoration finished -- so
    callers polling the public surface cannot race ahead of restore."""
    _scope, recorder = _drive_with_recorder(monkeypatch, lambda image: 7.0)
    true_writes = [w for w in recorder.writes if w[0] is True]
    false_writes = [w for w in recorder.writes if w[0] is False]
    assert true_writes and true_writes[0][1] is True, (
        'the True mirror must be written while _af_in_progress is set; '
        f'writes: {recorder.writes}'
    )
    assert false_writes and false_writes[-1][1] is False, (
        'the False mirror must be written only after _af_in_progress '
        f'cleared (post-restore); writes: {recorder.writes}'
    )
