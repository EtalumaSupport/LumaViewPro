# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The computer is asked to stay awake exactly while the scope is in use.

The owner is ``KeepAwake`` (``src/_lumascope/modules/keep_awake.py``),
composed by the session over its activity claim: a run, a recording, a
home or a diagnostic holding the scope, or a client holding the control
session, holds the host's request; nothing holding it releases it. The
host is ``tests/host_power_stub.py``'s recording host, the one stand-in
for the operating system the suite installs (``conftest.install_mock_deps``);
``host.held`` is what the operating system would list.
"""

from __future__ import annotations

import threading

import pytest

from _lumascope.drivers.host_power import AwakeNotHeld
from _lumascope.modules.keep_awake import KeepAwake
from _lumascope.modules.scope_session import ScopeSession
from tests.host_power_stub import host
from tests.settings_fixtures import complete_settings

SETTLE_S = 60


@pytest.fixture(autouse=True)
def free_host():
    host.reset()
    yield
    host.reset()


@pytest.fixture
def heard():
    return []


@pytest.fixture
def session(tmp_path, heard):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)), simulate=True, outcome_listener=heard.append
    )
    yield session
    session.shutdown()


def _sample_at_every_edge(session):
    """What the host held at each run-state edge, beside whether the scope was in use.

    Registered after the session's own listener, so each sample reads the
    level that listener has just applied.
    """
    samples = []
    claim = session.activity_claim

    def sample():
        in_use = claim.holder is not None or claim.control is not None
        samples.append((in_use, host.held))

    session.add_run_state_listener(sample)
    return samples


def test_the_startup_home_and_a_protocol_run_hold_the_computer_awake_while_they_run(
    session, tmp_path
):
    samples = _sample_at_every_edge(session)
    session.run_application_session()
    assert samples and host.held == 0, samples
    session.set_channel_acquire('BF', 'image')
    protocol = session.new_protocol(duration_s=0, period_s=0)

    run = session.protocol_runner().start_single_scan(protocol, parent_dir=str(tmp_path / 'runs'))
    outcome = run.result(timeout=SETTLE_S)

    assert outcome.status == 'completed', outcome
    assert (True, 1) in samples, samples
    assert all(held == int(in_use) for in_use, held in samples), samples
    assert host.held == 0


def test_a_recording_holds_the_computer_awake_until_it_stops(session):
    operation = session.manual_recording.start()
    during = host.held
    operation.stop()
    operation.result(timeout=SETTLE_S)

    assert during == 1
    assert host.held == 0


def test_a_control_session_holds_the_computer_awake_until_it_ends(session):
    session_id = session.take_control('plate reader')
    during = host.held
    session.end_control(session_id)

    assert during == 1
    assert host.held == 0


def test_a_request_the_host_refuses_is_told_once_and_the_work_goes_on(session, heard):
    host.refuse_with = AwakeNotHeld('keep_awake_refused', 'the host said no')

    for _ in range(2):
        session.end_control(session.take_control('plate reader'))

    told = [n for n in heard if n.title == 'Computer may sleep']
    assert [(n.kind, n.reason) for n in told] == [('notice', 'keep_awake_refused')], told
    assert 'the host said no' in told[0].message


def test_a_hold_the_host_ends_by_itself_is_told(session, heard):
    session.take_control('plate reader')
    host.on_lost(AwakeNotHeld('keep_awake_refused', 'logind ended it'))

    told = [n for n in heard if n.title == 'Computer may sleep']
    assert [(n.kind, n.reason) for n in told] == [('notice', 'keep_awake_refused')], told
    assert 'logind ended it' in told[0].message


def test_the_close_releases_a_hold_a_client_still_has(session):
    session.take_control('plate reader')
    assert host.held == 1

    session.shutdown()

    assert host.held == 0


def test_a_release_racing_a_newer_take_leaves_the_newer_level_held():
    """Of two changes racing, the level read last is the one the host holds.

    A run's release reads "not in use" and is held up before applying it,
    while a recording's take makes the scope in use again; the take's sync
    must not be undone by the release landing after it.
    """
    # a stand-in by design: the reader is the catcher's collaborator, gated
    # so the release is held up between its read and its apply.
    level = [True]
    release_read = threading.Event()
    let_release_go = threading.Event()

    def in_use():
        value = level[0]
        if threading.current_thread().name == 'release':
            release_read.set()
            let_release_go.wait(SETTLE_S)
        return value

    keep_awake = KeepAwake(in_use=in_use)
    keep_awake.sync()
    assert host.held == 1

    level[0] = False
    releasing = threading.Thread(target=keep_awake.sync, name='release')
    releasing.start()
    assert release_read.wait(SETTLE_S)
    level[0] = True
    taking = threading.Thread(target=keep_awake.sync, name='take')
    taking.start()
    let_release_go.set()
    releasing.join(SETTLE_S)
    taking.join(SETTLE_S)

    assert host.held == 1
