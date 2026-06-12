# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: post-AF camera keeps the run's committed gain/exposure targets.

Bug
---
The AF runner saved the raw camera state at AF start and re-applied it
whole at AF end. When the user committed a new exposure by clicking away
from the text box directly onto the AF button, AF correctly ran at the
new value, but the end-of-run restore reverted the camera to the stale
pre-AF snapshot -- the widget and settings store said one value while
the camera silently ran another.

Fix
---
AutofocusRunner._camera_state_to_restore() strips the fields the run was
given explicit targets for (the committed layer/step values) from the
snapshot before restore, so those values survive AF. Fields AF never
explicitly set still fall back to the snapshot.
"""

from __future__ import annotations

from modules.autofocus_runner import AutofocusRunner


def _runner(saved, gain, exposure) -> AutofocusRunner:
    runner = AutofocusRunner.__new__(AutofocusRunner)
    runner._saved_camera_state = saved
    runner._camera_gain = gain
    runner._camera_exposure = exposure
    return runner


def test_explicit_targets_survive_af():
    """Targeted gain+exposure must NOT revert to the pre-AF snapshot."""
    runner = _runner(
        {'tag': 'autofocus', 'gain_db': 0.0, 'exposure_ms': 10.0},
        gain=0.0,
        exposure=1.0,
    )
    assert runner._camera_state_to_restore() == {'tag': 'autofocus'}


def test_untargeted_field_falls_back_to_snapshot():
    """A field AF never set (gain here) still restores from the snapshot."""
    runner = _runner(
        {'tag': 'autofocus', 'gain_db': 5.0, 'exposure_ms': 10.0},
        gain=None,
        exposure=1.0,
    )
    assert runner._camera_state_to_restore() == {
        'tag': 'autofocus',
        'gain_db': 5.0,
    }


def test_no_targets_restores_full_snapshot():
    snapshot = {'tag': 'autofocus', 'gain_db': 5.0, 'exposure_ms': 10.0}
    runner = _runner(snapshot, gain=None, exposure=None)
    assert runner._camera_state_to_restore() == snapshot


def test_missing_snapshot_is_safe():
    runner = _runner(None, gain=0.0, exposure=1.0)
    assert runner._camera_state_to_restore() == {}
