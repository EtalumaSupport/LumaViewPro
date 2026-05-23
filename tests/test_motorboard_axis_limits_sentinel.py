"""Regression tests for MotorBoard.get_axis_limits() sentinel return.

Bug shape (#665): ``MotorBoard.get_axis_limits('T')`` logged at ERROR
on every call, including the documented "T has no configured limits"
case. The error fired multiple times per protocol scan because the
caller in sequenced_capture_runner.py iterates every axis returned by
``scope.capabilities.axes`` (which includes T) and wraps each call
with ``except Exception: pass``. The ERROR had already surfaced into
the user-visible log by the time the caller swallowed the exception.

Fix shape: align driver implementations with the protocol declaration
at ``drivers/protocols.py:77`` (``dict | None``). Return ``None`` for
the "no limits defined" case instead of raising. Keep HardwareError
for the "unsupported axis" case (genuine programmer error vs
configuration variant). Caller in sequenced_capture_runner.py
switched from try/except to a None check.

Tests below lock the new contract for the real and simulated drivers
and the API wrapper.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Real driver: MotorBoard
# ---------------------------------------------------------------------------


class TestMotorBoardReturnsSentinelForNoLimits:
    """drivers/motorboard.py::get_axis_limits returns None when an axis
    is supported but has no ``limits`` key in its config. Programmer-
    error case (unsupported axis at all) still raises HardwareError."""

    @pytest.fixture
    def motorboard_with_t_no_limits(self):
        from drivers.motorboard import MotorBoard
        from unittest.mock import patch

        # Real instantiation needs a serial connection — patch out the
        # __init__ and inject a minimal axes_config. The driver method
        # under test reads only self.axes_config.
        with patch.object(MotorBoard, '__init__', lambda self, *a, **kw: None):
            mb = MotorBoard()
            mb.axes_config = {
                'X': {'limits': {'min': 0, 'max': 100000}, 'move_func': lambda x: x},
                'Y': {'limits': {'min': 0, 'max': 100000}, 'move_func': lambda x: x},
                'Z': {'limits': {'min': 0, 'max': 14000}, 'move_func': lambda x: x},
                'T': {'move_func': lambda x: x},  # no 'limits' key
            }
            return mb

    def test_t_axis_returns_none_not_raises(self, motorboard_with_t_no_limits):
        # Was: raised HardwareError + logged ERROR. Now: returns None.
        result = motorboard_with_t_no_limits.get_axis_limits('T')
        assert result is None, (
            'T axis (no configured limits) must return None, not raise. '
            'ERROR-level logging on every protocol scan was a Rule 20 '
            "violation: 'no limits' is documented, expected behavior."
        )

    def test_xyz_axis_still_returns_limits_dict(self, motorboard_with_t_no_limits):
        for axis in ('X', 'Y', 'Z'):
            limits = motorboard_with_t_no_limits.get_axis_limits(axis)
            assert limits is not None, f'{axis} must return its configured limits'
            assert 'min' in limits and 'max' in limits

    def test_unsupported_axis_still_raises(self, motorboard_with_t_no_limits):
        from drivers.exceptions import HardwareError

        # Programmer error (axis not in config at all) is distinct from
        # configuration variant ("no limits") and must still raise.
        with pytest.raises(HardwareError):
            motorboard_with_t_no_limits.get_axis_limits('Q')


# ---------------------------------------------------------------------------
# Simulated driver: SimulatedMotorBoard
# ---------------------------------------------------------------------------


class TestSimulatedMotorBoardReturnsSentinelForNoLimits:
    """drivers/simulated_motorboard.py mirrors the real driver's
    contract (Rule 11: simulators behave like the real hardware path)."""

    def test_t_axis_returns_none_not_raises(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard

        board = SimulatedMotorBoard()
        result = board.get_axis_limits('T')
        assert result is None, (
            "SimulatedMotorBoard.get_axis_limits('T') must return None "
            'to match the real MotorBoard contract.'
        )

    def test_xyz_still_returns_limits_dict(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard

        board = SimulatedMotorBoard()
        for axis in ('X', 'Y', 'Z'):
            limits = board.get_axis_limits(axis)
            assert limits is not None
            assert 'min' in limits and 'max' in limits

    def test_unsupported_axis_still_raises(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard

        board = SimulatedMotorBoard()
        with pytest.raises(Exception):
            board.get_axis_limits('Q')


# ---------------------------------------------------------------------------
# Structural check: ERROR-level log line is gone from motorboard.py
# ---------------------------------------------------------------------------


class TestNoErrorLogForExpectedNoLimitsCase:
    """Lock that the misleading ERROR log line at the 'no limits'
    branch is gone. A future refactor that re-adds an ERROR log there
    would fire this test."""

    SRC = 'drivers/motorboard.py'

    def _get_axis_limits_source(self) -> str:
        path = pathlib.Path(__file__).resolve().parent.parent / self.SRC
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_axis_limits':
                text = ast.get_source_segment(source, node)
                assert text is not None
                return text
        raise AssertionError('get_axis_limits not found in motorboard.py')

    def test_no_limits_branch_does_not_log_error(self):
        body = self._get_axis_limits_source()
        # The unsupported-axis branch still legitimately logs at ERROR
        # (genuine programmer error). The 'no limits' branch must not.
        assert 'No limits defined' not in body, (
            'drivers/motorboard.py::get_axis_limits must not log '
            "'No limits defined' at ERROR — that case is now a "
            'sentinel return, not an error condition.'
        )


# ---------------------------------------------------------------------------
# Caller in sequenced_capture_runner: None-check pattern
# ---------------------------------------------------------------------------


class TestSequencedCaptureRunnerHandlesNoneFromGetAxisLimits:
    """Lock the caller's None-handling pattern. The try/except wrapping
    was effectively dead-letter after the driver-side fix; structurally
    the caller now branches on None instead of catching exceptions."""

    SRC = 'modules/sequenced_capture_runner.py'

    def test_caller_branches_on_none_return(self):
        path = pathlib.Path(__file__).resolve().parent.parent / self.SRC
        source = path.read_text()
        # Locate the run-validation block that calls get_axis_limits
        # and assert the None-check exists in surrounding context.
        idx = source.find('get_axis_limits(axis)')
        assert idx != -1, (
            'sequenced_capture_runner.py must call self._scope.motion.get_axis_limits(axis).'
        )
        # Look at the 200 chars after the call site
        window = source[idx : idx + 400]
        assert 'is not None' in window, (
            'After calling get_axis_limits, the caller must branch on '
            'is-not-None to skip axes that have no configured limits. '
            'Catching a broad Exception is now insufficient: the driver '
            "no longer raises for the expected 'no limits' case."
        )
