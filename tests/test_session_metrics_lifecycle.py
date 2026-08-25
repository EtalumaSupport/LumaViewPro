# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The session owns the metrics-logger lifecycle.

The MetricsLogger lives ON the scope, but its scheduler is a host
choice (Kivy Clock vs threading timers) -- so the session holds the
injected scheduler and owns start/stop/restart. The payoff is the
reconnect path: set_scope stops the OLD scope's logger (its
system/watchdog ticks survive a disconnect and would double-tick
beside the new logger) and starts the NEW scope's with the same
scheduler and the same settings-derived cadence. Sessions without a
scheduler (headless factories) never start metrics -- the pre-existing
contract, preserved.

MetricsLogger.start is NOT idempotent (a second start silently
overwrites its schedule handles, orphaning forever-ticking Clock
events), so start_metrics refuses a double start loudly; the running
fact lives in one place (the session flag) with start/stop as its only
writers. These members are host-serialized (main-thread-only in the
GUI); a threaded host must serialize them itself.

Also here: the ctx.metrics_logger absence sweep. The retired
AppContext mirror has no runtime guard (AppContext is a plain
dataclass; nothing reds if future code recreates the field), so the
sweep is the creep-back guard -- and it must include lumaviewpro.py
explicitly, because the package walker only sees modules/ and ui/
while the only realistic creep-back site is exactly lumaviewpro.py.
"""

import ast
from unittest.mock import MagicMock

import pytest

import tests.ast_seams as ast_seams
from modules.scope_session import ScopeSession
from tests.scope_fakes import spec_scope


_SCHEDULER = object()  # identity sentinel: restart must reuse THIS instance


def _make_session(**kwargs):
    defaults = {
        'settings': {'profiling': {'metrics_interval_s': 42}},
        'scope': spec_scope(),
        'io_executor': MagicMock(),
        'camera_executor': MagicMock(),
        'metrics_scheduler': _SCHEDULER,
    }
    defaults.update(kwargs)
    return ScopeSession(**defaults)


class TestStartStop:
    def test_start_metrics_uses_injected_scheduler_and_interval_override(self):
        session = _make_session()
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(
            _SCHEDULER, system_metrics_interval_s=42.0
        )

    def test_start_metrics_without_override_uses_logger_defaults(self):
        session = _make_session(settings={})
        session.start_metrics()
        session.scope.metrics_logger.start.assert_called_once_with(_SCHEDULER)

    def test_start_metrics_without_scheduler_is_a_no_op(self):
        session = _make_session(metrics_scheduler=None)
        session.start_metrics()
        session.scope.metrics_logger.start.assert_not_called()

    def test_double_start_refuses_loudly(self):
        # MetricsLogger.start overwrites its handles on a second call,
        # orphaning the first set as untracked forever-ticking events;
        # the refusal makes that state unreachable.
        session = _make_session()
        session.start_metrics()
        with pytest.raises(RuntimeError):
            session.start_metrics()

    def test_start_tolerates_absent_metrics_logger(self):
        # MetricsLogger construction can fail-to-warning; a scope with
        # metrics_logger None must not crash the session lifecycle.
        session = _make_session(scope=spec_scope(metrics_logger=None))
        session.start_metrics()

    def test_stop_metrics_stops_and_is_idempotent(self):
        session = _make_session()
        session.start_metrics()
        session.stop_metrics()
        session.scope.metrics_logger.stop.assert_called_once()
        session.stop_metrics()  # second stop: no raise, no second call
        session.scope.metrics_logger.stop.assert_called_once()

    def test_shutdown_stops_running_metrics(self):
        session = _make_session()
        session.start_metrics()
        session.shutdown()
        session.scope.metrics_logger.stop.assert_called_once()


class TestSetScopeRestart:
    def test_running_metrics_move_to_the_new_scope(self):
        session = _make_session()
        session.start_metrics()
        old_logger = session.scope.metrics_logger
        new = spec_scope()

        session.set_scope(new)

        old_logger.stop.assert_called_once()
        new.metrics_logger.start.assert_called_once_with(_SCHEDULER, system_metrics_interval_s=42.0)

    def test_stopped_metrics_stay_stopped_across_set_scope(self):
        session = _make_session()
        old_logger = session.scope.metrics_logger
        new = spec_scope()

        session.set_scope(new)

        old_logger.stop.assert_not_called()
        new.metrics_logger.start.assert_not_called()


class TestCtxMirrorStaysRetired:
    def test_no_production_read_of_ctx_metrics_logger(self):
        """ctx.metrics_logger was a Rule-2 mirror of scope.metrics_logger
        that went stale at every reconnect; it is retired and must not
        creep back. lumaviewpro.py is scanned explicitly -- the package
        walker cannot see top-level files, and the retired readers
        lived exactly there."""
        offenders = []

        def scan(rel_path, tree):
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Attribute) and node.attr == 'metrics_logger'):
                    continue
                value = node.value
                is_ctx = isinstance(value, ast.Name) and value.id == 'ctx'
                is_ctx_attr = isinstance(value, ast.Attribute) and value.attr == 'ctx'
                if is_ctx or is_ctx_attr:
                    offenders.append(f'{rel_path}:{node.lineno}')

        for rel_path, tree in ast_seams.iter_package_modules(('modules', 'ui')):
            scan(rel_path, tree)
        scan('lumaviewpro.py', ast_seams.parse_module('lumaviewpro.py'))

        assert not offenders, (
            'ctx.metrics_logger is a retired mirror of scope.metrics_logger '
            '(it went stale at reconnect); read the scope via the session '
            f'instead. Reads found: {offenders}'
        )
