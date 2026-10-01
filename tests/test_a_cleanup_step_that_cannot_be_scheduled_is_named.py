# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A cleanup step handed to the UI clock that cannot even be scheduled is
one of the run's cleanup failures, by name.

The two UI-side steps fail in two places: inside the scheduled call, which
reports itself, and in the scheduling, which cleanup collects with the
rest. The second is what the run's outcome names to its caller.
"""

from modules.notification_center import notifications
from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_cleanup import run_cleanup

from tests.test_audit_fixes import _run_cleanup_kwargs


def test_both_ui_steps_are_named_when_their_scheduling_fails(monkeypatch):
    def _cannot_schedule(*args, **kwargs):
        raise RuntimeError('the UI clock is gone')

    monkeypatch.setattr('modules.protocol_cleanup._schedule_cleanup_ui', _cannot_schedule)
    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda ex, *a, **k: reported.append(ex))
    recorded = []
    kwargs = _run_cleanup_kwargs(
        callbacks=ProtocolCallbacks(
            restore_layer_shader=lambda: None,
            sync_layer_widgets=lambda: None,
        ),
    )
    kwargs['record_cleanup_failures'] = recorded.append

    run_cleanup(**kwargs)

    assert recorded == [('Restore layer shader', 'Sync layer panel')], recorded
    assert len(reported) == 1, reported
