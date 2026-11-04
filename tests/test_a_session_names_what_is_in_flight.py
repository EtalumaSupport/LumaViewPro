# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session names what a gesture started that is still running, from each owner.

A sim walk's ``wait`` reads one list before its next step: what holds the
scope, each lane with a task running or queued, an axis moving, an
autofocus sweep. The GUI composes none of it (the guard
``test_the_gui_reads_run_state_from_the_session``); the Session asks the
owners. A real lane and a real home on the simulated scope here; the
moving-axis term is observed by the walk on the realistic tier, since the
fast tier's moves end before anything can read them.
"""

from __future__ import annotations

import threading

import pytest

from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.settings_fixtures import complete_settings

HELD_S = 30.0


@pytest.fixture
def session():
    s = ScopeSession.create(
        complete_settings(microscope='LS850'), simulate=True, warn_pre_release=False
    )
    assert s.io_executor.wait_for_idle(HELD_S), 'bring-up left the IO lane busy'
    yield s
    s.shutdown()


def test_nothing_in_flight_names_nothing(session):
    assert session.in_flight_named() == []


def test_a_lane_task_is_named_with_its_lane_while_it_runs(session):
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(HELD_S), 'the held task was never released'

    session.io_executor.put(IOTask(action=hold))
    assert entered.wait(HELD_S), 'the task never ran'
    try:
        still = session.in_flight_named()
    finally:
        release.set()
    assert len(still) == 1 and still[0].startswith('the IO lane (hold '), still
    assert session.io_executor.wait_for_idle(HELD_S)
    assert session.in_flight_named() == []


def test_a_home_is_named_as_holding_the_scope(session, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = session.scope._motion_driver.home

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(HELD_S), 'the held home was never released'
        return real(*args, **kwargs)

    monkeypatch.setattr(session.scope._motion_driver, 'home', held)
    home = threading.Thread(target=lambda: session.scope.motion.home('ALL'), daemon=True)
    home.start()
    assert entered.wait(HELD_S), 'the home never reached the driver'
    try:
        still = session.in_flight_named()
    finally:
        release.set()
    home.join(HELD_S)
    assert 'home' in still[0].lower(), still
    assert any(item.startswith('the IO lane (') for item in still), still


def test_a_plugin_run_is_named_from_its_press_to_its_end_with_no_claim_held(session):
    """A plugin run is named from the press, before any claim, until its worker leaves it.

    The press enters the count on one thread and the worker leaves it on
    its own, as the engineering tab does; nothing holds the scope between,
    so the name is the count's, not a claim's.
    """
    run = session.plugin_run_in_flight()
    run.__enter__()
    try:
        assert session.controls_lock.run_named is None
        assert session.in_flight_named() == ['an engineering plugin run']
    finally:
        worker = threading.Thread(target=run.__exit__, args=(None, None, None))
        worker.start()
        worker.join(HELD_S)
    assert session.in_flight_named() == []
