# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame listener that removes itself from inside its callback stops the simulated pump.

The budget enforcer removes a failing or slow handler from the camera's own
callback thread. When it was the last listener, the simulated camera tried
to join the pump thread it was running on, raised "cannot join current
thread", and the removal was logged as a driver that failed to unregister --
when the driver had already let it go.
"""

from __future__ import annotations

import threading

import modules.lumascope_api.imaging as imaging_mod


def test_a_last_listener_auto_removed_on_the_pump_thread_unregisters_cleanly(monkeypatch):
    from tests.scope_fakes import build_scope

    scope = build_scope(simulate=True)
    try:
        warnings = []
        monkeypatch.setattr(
            imaging_mod.logger, 'warning', lambda msg, *a, **k: warnings.append(msg)
        )
        monkeypatch.setattr(imaging_mod.logger, 'exception', lambda *a, **k: None)
        monkeypatch.setattr(imaging_mod.notifications, 'report_outcome', lambda *a, **k: None)
        removed = threading.Event()
        original_remove = scope.imaging._remove_wrapper

        def observing_remove(wrapper):
            original_remove(wrapper)
            removed.set()

        monkeypatch.setattr(scope.imaging, '_remove_wrapper', observing_remove)

        def bad(*_a):
            raise ValueError('plugin bug')

        scope.imaging.set_exposure_ms(1.0)
        scope.imaging.add_frame_listener(bad, name='bad_plugin')
        scope.imaging.start_streaming()
        assert removed.wait(timeout=5.0)

        assert [w for w in warnings if 'did not unregister' in w] == []
        assert scope._camera_driver._registered_frame_callbacks == []
    finally:
        scope.imaging.stop_streaming()
        scope.disconnect()
