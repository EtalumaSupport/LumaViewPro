# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame listener the camera refuses is raised to whoever registered it.

``add_frame_listener`` used to show a popup and return normally when the
camera driver refused a listener, so every registrant's own handling was
dead: a protocol video step's unwind never ran, a manual recording started
with no frames, and a plugin was listed as loaded with a handler that never
ran. Each registrant now hears the refusal and ends it its own way.

Beside it, the rest of the listener surface: a handler that raises on every
frame is bounded like a slow one (it logged a traceback per frame, 80 in
2 s on the simulator), a removal is reported once as a warning, and a
removal stops new calls whatever the driver does.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import modules.lumascope_api.imaging as imaging_mod
import modules.protocol_recording as protocol_recording
from modules import notification_center
from modules.exceptions import (
    CaptureError,
    FrameHandlerRemovedError,
    FrameListenerNotRegisteredError,
)
from modules.lumascope_api.imaging import HANDLER_BUDGET_MS, HANDLER_DROP_K, _BudgetedHandler
from modules.notification_center import Severity
from tests.shown_outcomes import capture_shown
from tests.test_manual_recording_controller import make_controller
from tests.test_video_camera_lost_outcome import _make_recorder
from tests.scope_fakes import bind_settings_like_a_session
from tests.protocol_drives import run_identity


def _refuse(*_a, **_kw):
    raise RuntimeError('scripted driver refusal')


@pytest.fixture
def shown(monkeypatch):
    """What the one reporter shows, with imaging's own name bound to it."""
    shown = capture_shown(monkeypatch)
    monkeypatch.setattr(imaging_mod, 'notifications', notification_center.notifications)
    return shown


@pytest.fixture
def sim_scope():
    from tests.scope_fakes import build_scope

    scope = build_scope(simulate=True)
    scope.runtime_state.set_turreted(False)
    bind_settings_like_a_session(scope, objective_id='4x Oly')
    yield scope
    scope.imaging.stop_streaming()
    scope.disconnect()


class TestTheRegistrantHearsTheRefusal:
    def test_a_refused_registration_raises_posts_nothing_and_can_be_retried(
        self, sim_scope, shown, monkeypatch
    ):
        def handler(*_a):
            pass

        with monkeypatch.context() as m:
            m.setattr(sim_scope._camera_driver, 'register_frame_callback', _refuse)
            with pytest.raises(FrameListenerNotRegisteredError) as raised:
                sim_scope.imaging.add_frame_listener(handler, name='probe')

        assert isinstance(raised.value.__cause__, RuntimeError)
        assert shown == [], 'the registrant reports it; the API shows nothing itself'
        assert handler not in sim_scope.imaging._frame_listener_wrappers

        sim_scope.imaging.add_frame_listener(handler, name='probe')
        assert handler in sim_scope.imaging._frame_listener_wrappers

    def test_a_plugin_whose_listener_is_refused_is_not_listed_as_loaded(
        self, sim_scope, monkeypatch
    ):
        from modules.plugins import PluginRegistry, PluginSpec

        registry = PluginRegistry()
        registry.live_processing.bind_scope(sim_scope)
        spec = PluginSpec(
            name='refused_plugin',
            version='1.0.0',
            requires_lvp_version='>=4.0.0',
            description='A plugin whose listener the camera refuses.',
        )

        def handler(*_a):
            pass

        with monkeypatch.context() as m:
            m.setattr(sim_scope._camera_driver, 'register_frame_callback', _refuse)
            with pytest.raises(FrameListenerNotRegisteredError):
                registry.live_processing.register(spec, handler)

        assert 'refused_plugin' not in registry.live_processing.names()
        # Not refused as a duplicate once the camera takes it.
        registry.live_processing.register(spec, handler)
        assert 'refused_plugin' in registry.live_processing.names()
        registry.live_processing.unregister('refused_plugin')

    @pytest.mark.parametrize('video_as_frames', [True, False])
    def test_a_record_press_is_told_the_recording_did_not_start_and_nothing_is_left(
        self, tmp_path, video_as_frames
    ):
        controller, scope, _ = make_controller(tmp_path, video_as_frames=video_as_frames)
        scope.imaging.add_frame_listener = MagicMock(
            side_effect=FrameListenerNotRegisteredError('manual_recording')
        )

        with pytest.raises(CaptureError) as raised:
            controller.start()

        assert raised.value.reason == 'recording_not_started'
        assert 'did not start' in str(raised.value)
        assert isinstance(raised.value.__cause__, FrameListenerNotRegisteredError)
        assert not controller.is_recording and not controller.is_draining
        manual = tmp_path / 'Manual'
        leftovers = sorted(p.name for p in manual.iterdir()) if manual.exists() else []
        assert leftovers == [], f'a start that never happened left {leftovers}'

    def test_a_video_step_whose_listener_is_refused_ends_with_no_frames(
        self, tmp_path, monkeypatch
    ):
        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, active_cached=True)
        recorder._scope.imaging.add_frame_listener.side_effect = FrameListenerNotRegisteredError(
            'protocol_video:clip'
        )
        reported = []
        monkeypatch.setattr(
            protocol_recording.notifications,
            'report_outcome',
            lambda exc, **kw: reported.append((exc, kw)),
        )

        with patch.object(recorder, '_prologue', return_value=None):
            outcome = recorder.run_blocking()

        assert outcome == protocol_recording.NO_FRAMES, (
            'a refused listener is a step with no frames, taking its strike -- not a run crash'
        )
        assert recorder._engine is None
        assert [type(exc) for exc, _ in reported] == [FrameListenerNotRegisteredError]
        assert reported[0][1]['solicited'] is False


class TestAFailingHandlerIsBounded:
    def test_a_handler_raising_k_frames_running_is_removed_with_one_traceback(self):
        imaging = MagicMock()
        calls = []

        def raises(*_a):
            calls.append(1)
            raise ValueError('plugin bug')

        w = _BudgetedHandler(imaging, raises, name='bad_plugin')
        with (
            patch.object(imaging_mod, 'logger') as log,
            patch.object(imaging_mod, 'notifications') as notify,
        ):
            for _ in range(HANDLER_DROP_K + 5):
                w(None, None, None)

        assert len(calls) == HANDLER_DROP_K, 'no call reaches the handler once it is removed'
        assert log.exception.call_count == 1, 'the first failure is logged; the rest counted'
        imaging._remove_wrapper.assert_called_once_with(w)
        outcome = notify.report_outcome.call_args[0][0]
        assert isinstance(outcome, FrameHandlerRemovedError)
        assert outcome.reason == 'raised'

    def test_a_handler_that_recovers_before_k_is_kept(self):
        imaging = MagicMock()
        results = iter([ValueError('x')] * (HANDLER_DROP_K - 1) + [None] * 3)

        def flaky(*_a):
            result = next(results)
            if result is not None:
                raise result

        w = _BudgetedHandler(imaging, flaky, name='flaky')
        with patch.object(imaging_mod, 'logger'), patch.object(imaging_mod, 'notifications'):
            for _ in range(HANDLER_DROP_K + 2):
                w(None, None, None)

        imaging._remove_wrapper.assert_not_called()
        assert w._consecutive_raised == 0

    def test_on_a_streaming_camera_a_raising_handler_logs_once_and_is_removed(
        self, sim_scope, shown, monkeypatch
    ):
        logged = []
        monkeypatch.setattr(
            imaging_mod.logger, 'exception', lambda msg, *a, **k: logged.append(msg)
        )
        removed = threading.Event()
        original_remove = sim_scope.imaging._remove_wrapper

        def observing_remove(wrapper):
            original_remove(wrapper)
            removed.set()

        monkeypatch.setattr(sim_scope.imaging, '_remove_wrapper', observing_remove)

        def bad(*_a):
            raise ValueError('plugin bug')

        sim_scope.imaging.set_exposure_ms(1.0)
        sim_scope.imaging.add_frame_listener(bad, name='bad_plugin')
        sim_scope.imaging.start_streaming()
        assert removed.wait(timeout=5.0), 'a handler failing every frame must be removed'
        time.sleep(0.2)

        assert len(logged) == 1, f'one traceback, not one per frame: {len(logged)}'
        assert bad not in sim_scope.imaging._frame_listener_wrappers
        assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Plugin Removed')]


class TestARemovalIsReportedAndStopsCalls:
    def test_an_over_budget_removal_is_one_warning_and_no_traceback(self, shown, monkeypatch):
        errors = []
        monkeypatch.setattr(
            notification_center._outcome_logger,
            'error',
            lambda *a, **k: errors.append(a),
        )
        imaging = MagicMock()
        calls = []

        def slow(*_a):
            calls.append(1)
            time.sleep((HANDLER_BUDGET_MS + 5) / 1000.0)

        w = _BudgetedHandler(imaging, slow, name='slow_plugin')
        for _ in range(HANDLER_DROP_K + 2):
            w(None, None, None)

        assert len(calls) == HANDLER_DROP_K
        assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Plugin Removed')]
        assert 'slow_plugin' in shown[0].message
        assert errors == [], 'a removal is a refusal: no ERROR, no traceback'

    def test_a_removal_the_driver_fails_to_unregister_still_stops_calls(
        self, sim_scope, monkeypatch
    ):
        calls = []

        def handler(*_a):
            calls.append(1)

        sim_scope.imaging.add_frame_listener(handler, name='leaky')
        wrapper = sim_scope.imaging._frame_listener_wrappers[handler]
        monkeypatch.setattr(sim_scope._camera_driver, 'unregister_frame_callback', _refuse)
        with patch.object(imaging_mod, 'logger') as log:
            sim_scope.imaging.remove_frame_listener(handler)

        # The driver still holds the wrapper; a frame it delivers does not
        # reach the handler.
        wrapper(None, None, None)
        assert calls == []
        log.warning.assert_called_once()
        log.exception.assert_not_called()


class TestTheUnwindAndRemovalEdges:
    """Each pins one hunk that no test above catches when reverted."""

    def test_an_auto_remove_stops_calls_even_when_the_driver_unregister_raises(
        self, sim_scope, monkeypatch
    ):
        calls = []

        def bad(*_a):
            calls.append(1)
            raise ValueError('plugin bug')

        sim_scope.imaging.add_frame_listener(bad, name='leaky_bad')
        wrapper = sim_scope.imaging._frame_listener_wrappers[bad]
        monkeypatch.setattr(sim_scope._camera_driver, 'unregister_frame_callback', _refuse)
        with (
            patch.object(imaging_mod, 'logger') as log,
            patch.object(imaging_mod, 'notifications'),
        ):
            for _ in range(HANDLER_DROP_K + 3):
                wrapper(None, None, None)

        assert len(calls) == HANDLER_DROP_K
        assert bad not in sim_scope.imaging._frame_listener_wrappers
        # The first failure's traceback only; the unregister failure is a WARNING.
        assert log.exception.call_count == 1
        log.warning.assert_called_once()

    def test_a_removal_with_no_driver_still_stops_calls_and_forgets_the_listener(self, sim_scope):
        calls = []

        def handler(*_a):
            calls.append(1)

        sim_scope.imaging.add_frame_listener(handler, name='orphan')
        wrapper = sim_scope.imaging._frame_listener_wrappers[handler]
        driver = sim_scope._camera_driver
        sim_scope._camera_driver = None
        try:
            sim_scope.imaging.remove_frame_listener(handler)
        finally:
            sim_scope._camera_driver = driver
        wrapper(None, None, None)
        assert calls == []
        assert handler not in sim_scope.imaging._frame_listener_wrappers
        driver.unregister_frame_callback(wrapper)

    def test_a_start_refused_at_the_engine_leaves_no_empty_frames_folder(self, tmp_path):
        from modules.exceptions import RecordingRefusedError

        controller, _, _ = make_controller(tmp_path, video_as_frames=True)
        controller._claim.try_claim('protocol', run=run_identity())
        with pytest.raises(RecordingRefusedError):
            controller.start()
        manual = tmp_path / 'Manual'
        leftovers = sorted(p.name for p in manual.iterdir()) if manual.exists() else []
        assert leftovers == []

    def test_a_frames_folder_with_something_in_it_is_kept(self, tmp_path):
        controller, scope, _ = make_controller(tmp_path, video_as_frames=True)

        def write_then_refuse(*_a, **_kw):
            folder = max((tmp_path / 'Manual').glob('Video_*'))
            (folder / 'frame_0001.tiff').write_bytes(b'x')
            raise FrameListenerNotRegisteredError('manual_recording')

        scope.imaging.add_frame_listener = write_then_refuse
        with pytest.raises(CaptureError):
            controller.start()
        kept = list((tmp_path / 'Manual').glob('Video_*/frame_0001.tiff'))
        assert len(kept) == 1, 'a folder holding anything is kept'

    def test_a_refused_video_step_resets_the_title(self, tmp_path, monkeypatch):
        clock = {'t': 1000.0}
        recorder = _make_recorder(tmp_path, clock, active_cached=True)
        recorder._scope.imaging.add_frame_listener.side_effect = FrameListenerNotRegisteredError(
            'protocol_video:clip'
        )
        reset = MagicMock()
        recorder._callbacks = {'reset_title': reset}
        monkeypatch.setattr(protocol_recording, '_schedule_ui', lambda fn, *a, **k: fn(0))
        monkeypatch.setattr(
            protocol_recording.notifications, 'report_outcome', lambda *a, **k: None
        )

        with patch.object(recorder, '_prologue', return_value=None):
            assert recorder.run_blocking() == protocol_recording.NO_FRAMES
        reset.assert_called_once()
