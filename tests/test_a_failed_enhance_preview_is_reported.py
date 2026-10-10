# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An Enhance output the display cannot hold is reported, not only logged.

The just-made output is shown from a clock callback nothing waits on. The
display caught its own failure and logged a WARNING, so a frame deeper than
its declared depth, or a shape the display cannot draw, was never seen. The
callback now runs the hold through run_unasked: the failure is reported
once, unasked, and the application keeps running.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import modules.app_context as _app_ctx
import ui.post_processing as post_processing
from modules.notification_center import notifications


def test_a_failed_hold_is_reported_unasked(monkeypatch):
    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda e, **k: reported.append((e, k)))
    failure = ValueError('a shape the display cannot draw')

    def _hold(image, bits):
        raise failure

    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(scope_display=SimpleNamespace(hold_derived_image=_hold))
    )
    monkeypatch.setattr(post_processing.Clock, 'schedule_once', lambda cb, _t: cb(0))

    post_processing.QuickEnhanceControls._queue_derived_image(
        SimpleNamespace(), np.zeros((4, 4), dtype=np.uint8), 8
    )

    assert len(reported) == 1, reported
    error, kwargs = reported[0]
    assert error is failure
    assert kwargs['solicited'] is False
    assert kwargs['category'] == 'UI:ENHANCE_PREVIEW'
