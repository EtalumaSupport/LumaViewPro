# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The session a program builds starts the memory profiler and the handle trace.

Both are process-wide and src/_lumascope/lib/ reads no setting, so ``ScopeSession.create``
decides them from its prepared settings, for every program that brings a
scope up: the GUI, the headless server, a script. The session also takes the
profiler's two startup snapshots, ``post_homing`` after a home that
succeeded and ``cold_start_done`` on its own timer once its metrics start,
so every host's trail holds the same named states.

The profiler's own entry points are stood in for by design: tracemalloc and
its sampler thread are process-wide and never stop, so a test that started
them would leave them running for every later test. What the session hands
them is the behaviour pinned.
"""

import pytest

from _lumascope.lib import handle_trace, memory_profile
from _lumascope.modules.exceptions import HomingFaultError
from _lumascope.modules.scope_session import COLD_START_SNAPSHOT_DELAY_S, ScopeSession
from _lumascope.modules.scheduler import ThreadingTimerScheduler
from tests.settings_fixtures import complete_settings


@pytest.fixture
def started(monkeypatch):
    """Every ``memory_profile.start`` and ``handle_trace.enable`` call, in order."""
    calls = []
    monkeypatch.setattr(
        memory_profile,
        'start',
        lambda *, enabled, interval_s: calls.append(('memory_profile', enabled, interval_s)),
    )
    monkeypatch.setattr(handle_trace, 'enable', lambda: calls.append(('handle_trace',)))
    return calls


@pytest.fixture
def snapshots(monkeypatch):
    """The states the profiler is asked to snapshot, with the profiler taken as on."""
    states = []
    monkeypatch.setattr(memory_profile, 'is_enabled', lambda: True)
    monkeypatch.setattr(memory_profile, 'snapshot', states.append)
    return states


def _create(**settings):
    session = ScopeSession.create(
        complete_settings(**settings), simulate=True, warn_pre_release=False
    )
    session.shutdown()


def test_a_session_whose_settings_say_on_starts_the_profiler_with_their_interval(started):
    _create(memory_profile_enabled=True, memory_profile_interval_s=2.0)

    assert ('memory_profile', True, 2.0) in started


def test_a_session_whose_settings_say_off_starts_no_profiler(started):
    _create(memory_profile_enabled=False)

    assert [c for c in started if c[0] == 'memory_profile' and c[1]] == []


def test_a_session_whose_settings_say_on_turns_the_handle_trace_on(started):
    _create(profiling={'handle_trace_enabled': True})

    assert ('handle_trace',) in started


def test_a_session_whose_settings_say_off_leaves_the_handle_trace_off(started):
    _create(profiling={'handle_trace_enabled': False})

    assert ('handle_trace',) not in started


class _RecordingScheduler(ThreadingTimerScheduler):
    """The session's scheduler, holding each one-shot instead of arming it."""

    def __init__(self):
        super().__init__()
        self.once = []

    def schedule_once(self, callback, delay_s):
        self.once.append((callback, delay_s))
        return None


def test_the_sessions_metrics_take_the_settled_footprint_on_its_timer(sim_scope, snapshots):
    from tests.scope_fakes import real_executor_bundle

    scheduler = _RecordingScheduler()
    session = ScopeSession(
        settings=complete_settings(),
        scope=sim_scope,
        executor_bundle=real_executor_bundle(),
        scheduler=scheduler,
    )
    snapshot_delays = []
    try:
        session.enable_metrics()
        assert snapshots == [], 'taken on the timer, not by the call'
        # Every one-shot the metrics start asked for (the camera's first
        # temperature sample is one), fired by hand.
        for fire, delay_s in scheduler.once:
            taken = len(snapshots)
            fire()
            if len(snapshots) > taken:
                snapshot_delays.append(delay_s)
    finally:
        session.disable_metrics()
        scheduler.shutdown()

    assert snapshots == ['cold_start_done']
    assert snapshot_delays == [COLD_START_SNAPSHOT_DELAY_S]


@pytest.fixture
def session():
    s = ScopeSession.create(
        complete_settings(microscope='LS850T'), simulate=True, warn_pre_release=False
    )
    yield s
    s.shutdown()


def test_a_startup_home_that_succeeds_is_followed_by_the_post_homing_snapshot(session, snapshots):
    session.run_application_session()

    assert snapshots == ['post_homing']


def test_a_startup_home_that_fails_takes_no_snapshot(session, snapshots):
    def failed(target):
        raise HomingFaultError('ALL', 'error', ('X', 'Y', 'Z', 'T'))

    session.run_application_session(home_fn=failed)

    assert snapshots == []


def test_a_start_with_homing_disabled_takes_no_snapshot(session, snapshots):
    session.run_application_session(disable_homing=True)

    assert snapshots == []
