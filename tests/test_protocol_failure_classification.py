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
    failures by hardware connection state."""

    def test_outer_except_calls_are_all_connected(self):
        body = _function_source(_read('modules/protocol_run_loop.py'), '_run_loop_inner')
        assert 'are_all_connected' in body, (
            '_run_loop_inner outer except must call are_all_connected() '
            'to classify exceptions as fatal (disconnected) vs '
            'transient (still connected).'
        )

    def test_fatal_branch_fires_hardware_disconnected_notification(self):
        """The fatal branch must fire a 'Hardware disconnected' (or
        equivalent) notification rather than the retired 'Protocol
        scan stopped' text."""
        src = _read('modules/protocol_run_loop.py')
        # The exact wording can evolve, but the disconnect-shape
        # notification must be the only error popup the outer handler
        # fires. "Protocol scan stopped" was the retired text.
        assert 'Protocol scan stopped' not in src, (
            "'Protocol scan stopped' notification text is retired. "
            "Transient failures don't notify; disconnects use the "
            "'Hardware disconnected' / 'Protocol Aborted' shape."
        )
        body = _function_source(src, '_run_loop_inner')
        assert 'Hardware disconnect' in body or 'Protocol Aborted' in body, (
            '_run_loop_inner fatal branch must surface a hardware-'
            'disconnect-shape notification to the user.'
        )

    def test_transient_branch_keeps_running(self):
        """The transient branch must NOT break the outer while loop
        or call cleanup -- it logs a warning and lets the next
        iteration retry the scan after the protocol period elapses."""
        body = _function_source(_read('modules/protocol_run_loop.py'), '_run_loop_inner')
        # The transient branch is identified by the warning that
        # mentions retry semantics. Don't pin to exact wording but
        # require the retry intent to be present.
        assert 'retry' in body.lower() or 'transient' in body.lower(), (
            '_run_loop_inner outer except must log the transient case '
            'explicitly so future readers understand the no-break '
            'no-increment semantics.'
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
        body = _function_source(_read('modules/protocol_step_runner.py'), 'scan_loop')
        assert 'self.scan_iterate()' in body, (
            'scan_loop must still call scan_iterate per iteration.'
        )

    def test_scan_loop_still_does_periodic_gc(self):
        body = _function_source(_read('modules/protocol_step_runner.py'), 'scan_loop')
        assert 'gc.collect()' in body, 'scan_loop must still run the periodic GC sweep.'
