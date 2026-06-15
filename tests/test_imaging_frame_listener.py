# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Phase 4d.5c unit tests + 4d.5f integration tests for the live_processing
frame-listener infrastructure.

- 4d.5c unit tests: _BudgetedHandler directly with mock callables.
- 4d.5f integration tests: end-to-end fan-out through SimulatedCamera
  (driver pump fires real callbacks at exposure rate), plus the
  ctx.plugins.live_processing namespace fan-out path.
"""

import threading
import time
from unittest.mock import MagicMock, patch

from modules.lumascope_api.imaging import (
    _BudgetedHandler,
    HANDLER_BUDGET_MS,
    HANDLER_DROP_K,
)


def _make_imaging_stub():
    """Minimal ImagingAPI stand-in exposing only _remove_wrapper.

    Tests that drive the wrapper directly don't need a real ImagingAPI;
    they only need the auto-remove hook to be observable.
    """
    m = MagicMock()
    m._remove_wrapper = MagicMock()
    return m


def test_in_budget_handler_does_not_count_toward_drop():
    """Fast handler runs every frame; counter stays at 0."""
    imaging = _make_imaging_stub()
    handler = MagicMock()
    w = _BudgetedHandler(imaging, handler, name='fast')
    for _ in range(10):
        w(None, None, None)
    assert handler.call_count == 10
    assert w._consecutive_over == 0
    imaging._remove_wrapper.assert_not_called()


def test_over_budget_increments_consecutive_counter():
    """Slow handler increments the over-budget counter each call."""
    imaging = _make_imaging_stub()

    def slow(*args):
        time.sleep((HANDLER_BUDGET_MS + 5) / 1000.0)

    w = _BudgetedHandler(imaging, slow, name='slow')
    w(None, None, None)
    w(None, None, None)
    assert w._consecutive_over == 2
    imaging._remove_wrapper.assert_not_called()


def test_one_in_budget_call_resets_counter():
    """A fast call resets the consecutive counter after over-budget hits."""
    imaging = _make_imaging_stub()
    delays_ms = [HANDLER_BUDGET_MS + 5, HANDLER_BUDGET_MS + 5, 0]
    iter_d = iter(delays_ms)

    def variable(*args):
        d = next(iter_d)
        if d > 0:
            time.sleep(d / 1000.0)

    w = _BudgetedHandler(imaging, variable, name='variable')
    w(None, None, None)
    w(None, None, None)
    w(None, None, None)
    assert w._consecutive_over == 0
    imaging._remove_wrapper.assert_not_called()


def test_drop_at_K_consecutive_over_budget():
    """K consecutive over-budget hits triggers auto-remove + notification."""
    imaging = _make_imaging_stub()

    def slow(*args):
        time.sleep((HANDLER_BUDGET_MS + 5) / 1000.0)

    w = _BudgetedHandler(imaging, slow, name='slow-plugin')
    with patch('modules.lumascope_api.imaging.notifications') as mock_notify:
        for _ in range(HANDLER_DROP_K):
            w(None, None, None)
        imaging._remove_wrapper.assert_called_once_with(w)
        mock_notify.warning.assert_called_once()
        # Notification title + body name the plugin so L1 knows who to debug.
        title_args = mock_notify.warning.call_args[0]
        assert 'slow-plugin' in title_args[1]
    assert w._removed is True


def test_handler_exception_does_not_count_toward_budget():
    """A handler that raises is logged but doesn't count as over-budget.

    Rationale: a handler that crashes is a different failure class
    from a handler that's slow. We don't want a one-off transient
    exception (e.g. numpy ValueError on a malformed frame) to slowly
    accumulate the counter to K and silently drop the plugin.
    """
    imaging = _make_imaging_stub()

    def raises(*args):
        raise RuntimeError('boom')

    w = _BudgetedHandler(imaging, raises, name='raises')
    for _ in range(5):
        w(None, None, None)
    assert w._consecutive_over == 0
    imaging._remove_wrapper.assert_not_called()


def test_removed_wrapper_short_circuits():
    """After auto-removal, further calls don't reach the handler."""
    imaging = _make_imaging_stub()
    handler = MagicMock()
    w = _BudgetedHandler(imaging, handler, name='dead')
    w._removed = True
    w(None, None, None)
    handler.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 4d.5f integration tests: end-to-end fan-out via SimulatedCamera
# ---------------------------------------------------------------------------
#
# These drive the real SimulatedCamera pump (background thread firing
# callbacks at the configured exposure rate) instead of unit-testing
# the wrapper directly. They cover the four shapes per
# WAVE7_PHASE_4D5_PLAN section 6: basic fan-out, drop policy under a
# real pump, concurrent register/unregister thread-safety, and the
# ctx.plugins.live_processing namespace forwarding.


def _make_simulated_scope():
    """Construct a Lumascope wired to SimulatedCamera with a fast pump.

    Default sim exposure is 10 ms; we drop to 1 ms so the pump fires
    ~1000 times/sec, which keeps wall-clock test latency in the
    tens-of-ms range even when waiting for K=30 callbacks.
    """
    from modules.lumascope_api._lumascope import Lumascope

    scope = Lumascope(simulate=True)
    scope.imaging.set_exposure_time(1.0)
    return scope


def _start_sim_pump(scope):
    """Spin up the SimulatedCamera callback pump."""
    scope._camera_driver.start_grabbing()


def _stop_sim_pump(scope):
    """Stop the SimulatedCamera pump (releases the background thread)."""
    scope._camera_driver.stop_grabbing()


def test_integration_basic_fanout_via_simulated_camera():
    """Register one listener; the SimulatedCamera pump fires it
    repeatedly; assert the listener received frames and the frame
    payload shape (image, timestamp, chunks)."""
    scope = _make_simulated_scope()
    received = []
    done = threading.Event()

    def listener(image, ts, chunks):
        received.append((image, ts, chunks))
        if len(received) >= 5:
            done.set()

    scope.imaging.add_frame_listener(listener, name='basic_fanout')
    _start_sim_pump(scope)
    try:
        assert done.wait(timeout=2.0), (
            f'listener did not receive 5 frames within 2s; got {len(received)}'
        )
    finally:
        scope.imaging.remove_frame_listener(listener)
        _stop_sim_pump(scope)

    # Sim has no chunk surface; chunks should always be None.
    assert all(r[2] is None for r in received)
    # Payload shape: image is a numpy array, ts is a datetime.
    import datetime
    import numpy as np

    assert isinstance(received[0][0], np.ndarray)
    assert isinstance(received[0][1], datetime.datetime)


def test_integration_drop_policy_via_simulated_camera():
    """A listener that sleeps past the budget every call gets auto-removed
    by the wrapper after HANDLER_DROP_K consecutive over-budget hits, even
    when fired by the real SimulatedCamera pump thread."""
    scope = _make_simulated_scope()
    call_count = [0]
    removed = threading.Event()
    slow_ms = HANDLER_BUDGET_MS + 5

    def slow_listener(image, ts, chunks):
        call_count[0] += 1
        time.sleep(slow_ms / 1000.0)

    # Monkey-patch _remove_wrapper so we can observe the drop event.
    original_remove = scope.imaging._remove_wrapper

    def observing_remove(wrapper):
        original_remove(wrapper)
        removed.set()

    scope.imaging._remove_wrapper = observing_remove  # type: ignore[method-assign]

    scope.imaging.add_frame_listener(slow_listener, name='slow_integration')
    _start_sim_pump(scope)
    try:
        # K=30 over-budget hits at ~slow_ms each = ~slow_ms * K wall-clock
        # plus pump scheduling overhead. 5s budget is generous.
        assert removed.wait(timeout=5.0), (
            f'slow listener was not auto-removed within 5s; '
            f'call_count={call_count[0]} (expected at least {HANDLER_DROP_K})'
        )
        assert call_count[0] >= HANDLER_DROP_K
    finally:
        _stop_sim_pump(scope)

    # After auto-remove, the listener is no longer in the wrappers dict.
    assert slow_listener not in scope.imaging._frame_listener_wrappers


def test_integration_concurrent_register_unregister():
    """Two threads racing on register/unregister of distinct handlers
    don't corrupt the wrapper dict or the driver's callback list."""
    scope = _make_simulated_scope()
    handlers = [(lambda i, t, c: None) for _ in range(50)]
    errors = []

    def worker(start_idx, step):
        try:
            for i in range(start_idx, len(handlers), step):
                scope.imaging.add_frame_listener(handlers[i], name=f'h_{i}')
            for i in range(start_idx, len(handlers), step):
                scope.imaging.remove_frame_listener(handlers[i])
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(0, 2))
    t2 = threading.Thread(target=worker, args=(1, 2))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert not errors, f'concurrent register/unregister raised: {errors}'
    # All listeners registered + unregistered cleanly.
    assert scope.imaging._frame_listener_wrappers == {}


def test_integration_plugin_namespace_fanout_via_simulated_camera():
    """ctx.plugins.live_processing.register(spec, handler) routes through
    to scope.imaging.add_frame_listener and the SimulatedCamera pump
    fires the handler. Tests the end-to-end registration -> driver fan-out
    path that intern-led live_processing plugins will exercise."""
    from modules.plugins import PluginRegistry, PluginSpec

    scope = _make_simulated_scope()
    registry = PluginRegistry()
    registry.live_processing.bind_scope(scope)

    spec = PluginSpec(
        name='ns_fanout_demo',
        version='1.0.0',
        requires_lvp_version='>=4.0.0',
        description='Live-processing namespace fan-out integration test.',
    )
    received = []
    done = threading.Event()

    def handler(image, ts, chunks):
        received.append((image, ts, chunks))
        if len(received) >= 3:
            done.set()

    registry.live_processing.register(spec, handler)
    _start_sim_pump(scope)
    try:
        assert done.wait(timeout=2.0), (
            f'plugin handler did not receive 3 frames; got {len(received)}'
        )
    finally:
        registry.live_processing.unregister('ns_fanout_demo')
        _stop_sim_pump(scope)

    # After unregister, the handler is gone from the underlying wrapper
    # registry (one canonical source-of-truth per Rule 35).
    assert handler not in scope.imaging._frame_listener_wrappers


def test_add_frame_listener_notifies_user_on_driver_registration_failure(monkeypatch):
    """When the driver rejects register_frame_callback, add_frame_listener
    must surface the failure via notifications.warning (Rule 14).

    Pre-fix, the except handler logged + rolled back the dict entry but
    fired no user-facing notification -- a plugin's frame handler would
    silently never receive frames, with no signal to the user that the
    registration failed. AUDIT_SILENT_FAIL_AST_2026-05-23 flagged this
    as the one confirmed Class B Rule 14 violation in the listener
    cluster.
    """
    from modules.lumascope_api import imaging as imaging_mod

    scope = _make_simulated_scope()

    # Force the driver-side registration to fail.
    def boom(*_a, **_kw):
        raise RuntimeError('synthetic driver rejection for test')

    monkeypatch.setattr(scope._camera_driver, 'register_frame_callback', boom)

    captured = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            captured.append((category, title, message))

        def info(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def critical(self, *_a, **_kw):
            pass

    monkeypatch.setattr(imaging_mod, 'notifications', _RecordingNotifier())

    def handler(_image, _ts, _chunks):
        pass

    scope.imaging.add_frame_listener(handler, name='rejected_listener')

    assert len(captured) == 1, (
        f'add_frame_listener must fire exactly one notifications.warning '
        f'when the driver rejects registration. Captured: {captured}'
    )
    category, title, _message = captured[0]
    assert category == 'Frame Listener', (
        f"Notification category must be 'Frame Listener'; got {category!r}"
    )
    assert 'rejected_listener' in title, (
        f'Notification title must name the listener so the user can correlate '
        f'with their registration call; got title={title!r}'
    )
    # The dict rollback must still happen -- a future register attempt
    # for the same handler must not see the stale wrapper entry.
    assert handler not in scope.imaging._frame_listener_wrappers, (
        'Failed registration must roll back the dict so a retry can fire.'
    )
