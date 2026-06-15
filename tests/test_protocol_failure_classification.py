# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the fatal-vs-transient classification at the
protocol run-loop level.

Before this fix:
- ``scan_loop`` caught every exception and fired a
  ``notifications.error("Protocol scan stopped", ...)`` popup, then
  cleared ``_scan_in_progress`` and returned normally.
- The outer ``_run_loop_inner`` saw the clean return, incremented
  ``_scan_count``, waited the protocol period, and re-ran the scan.
  Same failure -> same popup -> same retry, every period.

Symptom: bench session showed identical "Protocol scan stopped"
notifications ~5 min apart on the same protocol -- the periodic
scan scheduler kept restarting on a fault that wasn't transient.

After this fix:
- ``scan_loop`` no longer catches exceptions; they propagate to
  ``_run_loop_inner``'s outer except.
- The outer except classifies via ``scope.are_all_connected()``:
    * disconnected -> fatal, fire "Protocol Aborted" + cleanup + break
    * connected   -> transient, log warning only, do not increment
      ``_scan_count``, do not break; outer loop's next iteration
      waits the protocol period and retries the same scan number.
- The "Protocol scan stopped" notification text is retired entirely.
  Transients are silent; fatals route through one "Hardware
  disconnected" notification path.

These tests pin the source-level shape so a future cleanup that
re-introduces the broad except or the spurious notification text
fires the regression.
"""

from __future__ import annotations

import ast
import pathlib


def _read(path: str) -> str:
    return (pathlib.Path(__file__).resolve().parent.parent / path).read_text()


def _function_source(source: str, func_name: str) -> str:
    """Return the raw source text of a named method/function."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise AssertionError(f'could not extract source for {func_name!r}')
            return text
    raise AssertionError(f'function {func_name!r} not found in source')


class TestScanLoopPropagatesExceptions:
    """``scan_loop`` must NOT swallow exceptions with a broad except.
    The outer run-loop is the single point of failure classification."""

    def test_scan_loop_has_no_broad_except_block(self):
        body = _function_source(_read('modules/protocol_step_runner.py'), 'scan_loop')
        # The old broad-except pattern fired a notification and broke
        # the loop. Either of these substrings appearing in scan_loop
        # is a regression.
        assert 'Protocol scan stopped' not in body, (
            "scan_loop must not fire 'Protocol scan stopped' "
            'notification -- that classification lives in the outer '
            'run_loop_inner via are_all_connected().'
        )
        assert 'notifications.error' not in body, (
            'scan_loop must not call notifications.error directly. '
            'Let exceptions propagate to the outer handler.'
        )

    def test_scan_loop_does_not_catch_broad_exception(self):
        """AST-check: no ``except Exception`` (or bare ``except``) inside
        the scan_loop function body."""
        src = _read('modules/protocol_step_runner.py')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'scan_loop':
                for sub in ast.walk(node):
                    if isinstance(sub, ast.ExceptHandler):
                        # Allow narrow except like AutofocusAborted, but
                        # fail if it's a bare except or `except Exception`.
                        if sub.type is None:
                            raise AssertionError(
                                'scan_loop contains a bare `except:` -- let exceptions propagate.'
                            )
                        # Check for `except Exception` (broad)
                        type_src = ast.get_source_segment(src, sub.type)
                        if type_src and type_src.strip() == 'Exception':
                            raise AssertionError(
                                f'scan_loop has `except {type_src}:` -- '
                                f'broad excepts at this layer fire the '
                                f'wrong notification and turn transient '
                                f'faults into halting popups.'
                            )
                return
        raise AssertionError('scan_loop function not found')


class TestRunLoopInnerClassifiesByConnection:
    """The outer ``_run_loop_inner`` exception handler must classify
    failures by hardware connection state: disconnected = fatal (abort +
    notify), still-connected = transient (silent retry on next period,
    bounded by the consecutive-failure ceiling)."""

    def _drive_failing_run_loop(self, monkeypatch, *, connected):
        """run_loop on a runner whose every scan raises; classification
        is steered by the mocked are_all_connected."""
        from unittest.mock import MagicMock

        from modules.notification_center import notifications
        from tests.protocol_drives import protocol_step, run_loop_ready_runner

        captured = []
        monkeypatch.setattr(notifications, 'error', lambda *a, **k: captured.append(a))
        runner = run_loop_ready_runner(protocol_step())
        runner._protocol.step.side_effect = RuntimeError('serial dropped mid-step')
        runner._scope.are_all_connected = MagicMock(return_value=connected)
        runner._run_loop_executor.run_loop()
        return runner, captured

    def test_disconnect_aborts_with_classified_notification(self, monkeypatch):
        from modules.protocol_state_machine import ProtocolState

        runner, captured = self._drive_failing_run_loop(monkeypatch, connected=False)
        assert len(captured) == 1 and captured[0][1] == 'Protocol Aborted', (
            f'a disconnect must surface exactly one abort popup; got {captured}'
        )
        assert 'Hardware disconnected' in captured[0][2], (
            f'the popup must name the disconnect; got {captured[0]}'
        )
        assert runner.protocol_state == ProtocolState.ERROR, (
            'a disconnect mid-scan must land the run in ERROR'
        )
        assert runner._protocol.step.call_count == 1, (
            'a fatal failure must abort, not retry the scan'
        )
        assert runner._cleanup.called

    def test_transient_failure_retries_then_escalates(self, monkeypatch):
        runner, captured = self._drive_failing_run_loop(monkeypatch, connected=True)
        assert runner._protocol.step.call_count == 3, (
            'transient (still-connected) failures must retry on the next '
            f'period up to the ceiling; got {runner._protocol.step.call_count} attempts'
        )
        assert runner._scan_count == 0, 'failed scans must not count as completed'
        assert len(captured) == 1, (
            f'transients are silent until the consecutive-failure ceiling; got {captured}'
        )
        assert '3 times' in captured[0][2] and 'in a row' in captured[0][2], (
            f'the ceiling popup must name the repeated failure; got {captured[0]}'
        )

    def test_retired_scan_stopped_notification_absent(self):
        """The per-failure 'Protocol scan stopped' popup is retired --
        transients are silent; fatals use the disconnect shape."""
        assert 'Protocol scan stopped' not in _read('modules/protocol_run_loop.py'), (
            "'Protocol scan stopped' notification text is retired. "
            "Transient failures don't notify; disconnects use the "
            "'Hardware disconnected' / 'Protocol Aborted' shape."
        )

    def test_outer_except_does_not_fire_generic_protocol_error(self):
        """The retired generic 'Protocol Error' notification (which
        used to fire for every exception, fatal or transient) must be
        gone from the source. All notifications now route through the
        classified disconnect path."""
        src = _read('modules/protocol_run_loop.py')
        # The exact retired call shape was:
        #   notifications.error("Protocol", "Protocol Error", str(ex))
        assert '"Protocol Error"' not in src, (
            "Retired 'Protocol Error' notification title -- transients "
            "are silent; fatals use 'Protocol Aborted' / 'Hardware "
            "disconnected'."
        )


class TestScanLoopBehaviorPreserved:
    """Sanity: scan_loop still runs the iteration body + 60s GC
    maintenance after the refactor."""

    def test_scan_loop_still_calls_scan_iterate(self):
        from tests.protocol_drives import protocol_step, scan_ready_runner

        runner = scan_ready_runner(protocol_step())
        runner._step_executor.scan_loop()
        assert runner._protocol.step.called, (
            'scan_loop must drive scan_iterate (which fetches the step row)'
        )
        assert not runner._scan_in_progress.is_set(), 'the single-step scan must run to completion'

    def test_scan_loop_still_does_periodic_gc(self, monkeypatch):
        """With >60s elapsing between maintenance checks (faked clock),
        scan_loop must run its GC sweep."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.protocol_drives import protocol_step, scan_ready_runner

        runner = scan_ready_runner(protocol_step())
        ticks = {'now': 0.0}

        def fake_monotonic():
            ticks['now'] += 61.0
            return ticks['now']

        monkeypatch.setattr(
            'modules.protocol_step_runner.time',
            SimpleNamespace(monotonic=fake_monotonic, sleep=lambda s: None),
        )
        gc_recorder = MagicMock()
        gc_recorder.collect.return_value = 0
        monkeypatch.setattr('modules.protocol_step_runner.gc', gc_recorder)
        runner._step_executor.scan_loop()
        assert gc_recorder.collect.called, 'scan_loop must run the periodic GC sweep on long scans'
