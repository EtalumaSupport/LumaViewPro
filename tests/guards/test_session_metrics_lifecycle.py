# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The session owns the metrics-logger lifecycle.

The MetricsLogger lives ON the scope, but its scheduler is a host
choice (Kivy Clock vs threading timers) -- so the session holds the
injected scheduler and owns start/stop. Sessions without a
scheduler (headless factories) never start metrics -- the pre-existing
contract, preserved.

MetricsLogger.start is NOT idempotent (a second start silently
overwrites its schedule handles, orphaning forever-ticking Clock
events), so enable_metrics refuses a double start loudly; the running
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
import logging
from unittest.mock import MagicMock, patch

import pytest

import tests.ast_seams as ast_seams
from modules.exceptions import AlreadyStartedError, RefusalCause
from modules.scope_session import ScopeSession
from tests.scope_fakes import session_on_a_simulated_scope


# Identity sentinel: restart must reuse THIS instance. A mock (not a bare
# object) because the session shuts its scheduler down at teardown -- the
# injected instance must satisfy the Scheduler protocol.
_SCHEDULER = MagicMock()


def _make_session(**kwargs):
    defaults = {
        'settings': {'profiling': {'metrics_interval_s': 42}},
        'scheduler': _SCHEDULER,
    }
    defaults.update(kwargs)
    # The session builds its own metrics logger; a stand-in shows what the
    # session asks of it.
    with patch('modules.scope_session.MetricsLogger'):
        return session_on_a_simulated_scope(**defaults)


class TestStartStop:
    def test_enable_metrics_uses_injected_scheduler_and_interval_override(self):
        session = _make_session()
        session.enable_metrics()
        session.metrics_logger.start.assert_called_once_with(
            _SCHEDULER, display=None, system_metrics_interval_s=42.0
        )

    def test_enable_metrics_without_override_uses_logger_defaults(self):
        session = _make_session(settings={'profiling': {'metrics_interval_s': None}})
        session.enable_metrics()
        session.metrics_logger.start.assert_called_once_with(_SCHEDULER, display=None)

    def test_default_session_owns_a_real_scheduler_and_metrics_stay_opt_in(self):
        # No injected scheduler: the session builds and owns its own, and
        # metrics still start ONLY when enable_metrics is called -- the
        # scheduler existing must not start them.
        session = _make_session(scheduler=None)
        session.metrics_logger.start.assert_not_called()
        session.enable_metrics()
        session.metrics_logger.start.assert_called_once()
        (sched_arg,), _kwargs = session.metrics_logger.start.call_args
        assert sched_arg is session._scheduler

    def test_double_start_refuses_loudly(self):
        # MetricsLogger.start overwrites its handles on a second call,
        # orphaning the first set as untracked forever-ticking events;
        # the refusal makes that state unreachable.
        session = _make_session()
        session.enable_metrics()
        with pytest.raises(AlreadyStartedError) as refused:
            session.enable_metrics()
        assert refused.value.reason == 'metrics_running'
        # Refused for the scope's state: a stop lets it start again.
        assert refused.value.cause is RefusalCause.STATE
        assert isinstance(refused.value, RuntimeError), 'still what it was documented as'
        session.metrics_logger.start.assert_called_once()
        session.disable_metrics()
        session.enable_metrics()

    def test_disable_metrics_stops_and_is_idempotent(self):
        session = _make_session()
        session.enable_metrics()
        session.disable_metrics()
        session.metrics_logger.stop.assert_called_once()
        session.disable_metrics()  # second stop: no raise, no second call
        session.metrics_logger.stop.assert_called_once()

    def test_shutdown_stops_running_metrics(self):
        session = _make_session()
        session.enable_metrics()
        session.shutdown()
        session.metrics_logger.stop.assert_called_once()


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


def test_a_host_that_starts_metrics_logs_its_environment_first(caplog):
    """Every host that starts metrics -- the GUI and the REST server alike --
    gets the environment fingerprint once, ahead of any metric line, because
    the session's start logs it rather than one host's launch."""
    from tests.settings_fixtures import complete_settings

    caplog.set_level(logging.INFO, logger='LVP.metrics')
    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        caplog.clear()
        session.enable_metrics()
        lines = [r.getMessage() for r in caplog.records if r.name == 'LVP.metrics']
    finally:
        session.shutdown()

    fingerprints = [line for line in lines if line.startswith('[ENV METRICS]')]
    assert len(fingerprints) == 1, f'one fingerprint per metrics start: {lines}'
    assert lines[0].startswith('[ENV METRICS]'), f'a metric line came first: {lines}'
