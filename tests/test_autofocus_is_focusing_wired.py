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

Tests drive a real sweep on the simulated scope (tests/af_drives.py) and
observe the public flag from inside and after the run.
"""

from __future__ import annotations

import modules.autofocus_functions as autofocus_functions
from modules.lumascope_api.illumination import LedState
from modules.lumascope_api.imaging import ImagingAPI
from tests.af_drives import af_runner_on, sweep


def _lit(scope):
    return [r.name for r in scope.illumination.get_led_states() if r.state is LedState.LIT]


def test_is_focusing_true_while_af_runs(sim_scope, monkeypatch):
    """During the scan loop, the public flag reads True."""
    seen = []
    score = autofocus_functions.autofocus_score

    def scoring(image):
        seen.append(sim_scope.imaging.is_focusing)
        return score(image)

    runner = af_runner_on(sim_scope)
    monkeypatch.setattr('modules.autofocus_functions.autofocus_score', scoring)
    sweep(runner, sim_scope)
    assert seen and all(value is True for value in seen), (
        'scope.imaging.is_focusing must read True while the AF loop is '
        f'scoring frames; observed {seen}'
    )


def test_is_focusing_false_after_run_completes(sim_scope):
    """After run() returns, the public flag reads False."""
    runner = af_runner_on(sim_scope)
    sweep(runner, sim_scope)
    assert sim_scope.imaging.is_focusing is False, (
        'scope.imaging.is_focusing must be False once run() has returned'
    )


def test_false_flip_waits_for_the_restore(sim_scope, monkeypatch):
    """The False write lands only after the runner's in-progress probe has
    cleared -- i.e. after camera/LED/Z restoration finished -- so callers
    polling the public surface cannot race ahead of restore."""
    runner = af_runner_on(sim_scope)
    writes = []
    flag = ImagingAPI.is_focusing

    def recording(imaging, value):
        writes.append((value, runner.in_progress(), _lit(sim_scope)))
        flag.fset(imaging, value)

    monkeypatch.setattr(ImagingAPI, 'is_focusing', property(flag.fget, recording))

    sweep(runner, sim_scope)

    true_writes = [w for w in writes if w[0] is True]
    false_writes = [w for w in writes if w[0] is False]
    assert true_writes and true_writes[0][1] is True, (
        f'the True mirror must be written while the runner is in progress; writes: {writes}'
    )
    assert false_writes and all(w[1:] == (False, []) for w in false_writes), (
        'every False mirror write must come only after the in-progress probe '
        f'cleared and the sweep light was put out (post-restore); writes: {writes}'
    )
