# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Phase 4d.5c unit tests -- _BudgetedHandler budget + drop policy.

Tests the wrapper class directly with mock callables. End-to-end
fan-out tests through SimulatedCamera live in Phase 4d.5f per
WAVE7_PHASE_4D5_PLAN section 6 test infrastructure.
"""
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
        raise RuntimeError("boom")
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
