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


class TestRunLoopInnerClassifiesByConnection:
    """The outer ``_run_loop_inner`` exception handler must classify
    failures by hardware connection state: disconnected = fatal (abort +
    notify), still-connected = transient (silent retry on next period,
    bounded by the consecutive-failure ceiling)."""

    def _drive_failing_run_loop(self, centre_posts, *, connected):
        """run_loop on a runner whose every scan raises; classification
        is steered by the mocked are_all_connected."""
        from unittest.mock import MagicMock

        from modules.notification_center import Severity
        from tests.protocol_drives import protocol_step, run_loop_ready_runner

        # Both channels, because the two classifications notify through
        # different ones and "exactly one popup" has to count them all: a
        # fatal abort goes through the fatal-abort funnel, which posts at
        # critical severity, while the consecutive-failure ceiling notifies
        # itself at error. Capturing one channel would let a second popup on
        # the other slip past unseen.
        runner = run_loop_ready_runner(protocol_step())
        runner._protocol.step.side_effect = RuntimeError('serial dropped mid-step')
        runner._scope.are_all_connected = MagicMock(return_value=connected)
        runner._run_loop_executor.run_loop(runner._last_run())
        captured = [
            (n.category, n.title, n.message)
            for n in centre_posts
            if n.severity in (Severity.ERROR, Severity.CRITICAL)
        ]
        return runner, captured

    def test_disconnect_aborts_with_classified_notification(self, centre_posts):
        from modules.protocol_state_machine import ProtocolState

        runner, captured = self._drive_failing_run_loop(centre_posts, connected=False)
        # The abort and its popup both come from the fatal-abort funnel, which
        # this harness holds as a mock -- so the observable here is the one
        # call into it, carrying the cause. The popup the funnel then posts is
        # pinned by the funnel's own ordering test.
        aborts = runner._image_writer._abort_run_fatal.call_args_list
        assert len(aborts) == 1, (
            f'a disconnect must abort the run exactly once; got {len(aborts)} calls'
        )
        reason, _domain, title, message = aborts[0].args
        assert (reason, title) == ('hardware_disconnected', 'Protocol Aborted'), (
            f'the abort must name the disconnect as its cause; got {aborts[0].args}'
        )
        assert 'Hardware disconnected' in message, (
            f'the message the user reads must name the disconnect; got {message!r}'
        )
        assert captured == [], (
            f'the funnel posts the disconnect popup; a site posting its own too '
            f'would show the user two dialogs for one fault. Got {captured}'
        )
        assert runner._state == ProtocolState.ERROR, (
            'a disconnect mid-scan must land the run in ERROR'
        )
        assert runner._protocol.step.call_count == 1, (
            'a fatal failure must abort, not retry the scan'
        )
        assert runner._cleanup.called

    def test_transient_failure_retries_then_escalates(self, centre_posts):
        runner, captured = self._drive_failing_run_loop(centre_posts, connected=True)
        assert runner._protocol.step.call_count == 3, (
            'transient (still-connected) failures must retry on the next '
            f'period up to the ceiling; got {runner._protocol.step.call_count} attempts'
        )
        assert runner._scan_count == 0, 'failed scans must not count as completed'
        assert len(captured) == 1, (
            f'transients are silent until the consecutive-failure ceiling; got {captured}'
        )
        assert runner._image_writer._abort_run_fatal.call_args_list == [], (
            'the strike ceiling stops a run the instrument could not complete; '
            'it must not force-darken the sample the way a fault does'
        )
        assert '3 times' in captured[0][2] and 'in a row' in captured[0][2], (
            f'the ceiling popup must name the repeated failure; got {captured[0]}'
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
