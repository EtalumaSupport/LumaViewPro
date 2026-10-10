# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stage map that fails to draw is reported once, not only logged.

The labware and step-marker drawings run in clock callbacks that nothing
waits on. Each caught its own failure and logged it at ERROR, so the person
saw an empty map and no word why. They now run through run_unasked, the
GUI's one catch for such a callback: the failure is handed to the reporter,
unasked, under the drawing's own label, and the application keeps running.
"""

from __future__ import annotations

import pytest

from modules.notification_center import notifications
from ui.stage import Stage


@pytest.mark.parametrize(
    ('callback', 'builder', 'label'),
    [
        ('_draw_labware_fbo_scheduled', 'create_labware_fbo', 'UI:STAGE_LABWARE_DRAW'),
        ('_draw_steps_fbo_scheduled', 'create_step_locations_fbo', 'UI:STAGE_STEPS_DRAW'),
    ],
)
def test_a_failed_drawing_is_reported_unasked(monkeypatch, callback, builder, label):
    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda e, **k: reported.append((e, k)))
    failure = RuntimeError('framebuffer incomplete')

    def _fails():
        raise failure

    stage = Stage.__new__(Stage)
    setattr(stage, builder, _fails)

    getattr(stage, callback)(0, 0, 520, 346, 0)

    assert len(reported) == 1, f'the failure must be reported once; got {reported}'
    error, kwargs = reported[0]
    assert error is failure
    assert kwargs['solicited'] is False
    assert kwargs['category'] == label
