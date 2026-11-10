# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The computer is asked to stay awake exactly while a session is open.

The owner is ``KeepAwake`` (``src/_lumascope/modules/keep_awake.py``),
composed by the session: built with the session, released by its close,
whatever the session does in between. The host is
``tests/host_power_stub.py``'s recording host, the one stand-in for the
operating system the suite installs (``conftest.install_mock_deps``);
``host.held`` is what the operating system would list.
"""

from __future__ import annotations

import logging

import pytest

from _lumascope.drivers.host_power import AwakeNotHeld
from _lumascope.modules.lumascope_api.bring_up import CONFIGURING
from _lumascope.modules.scope_session import ScopeSession
from tests.host_power_stub import HOLDS, host
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


def _create(tmp_path, heard, **kwargs):
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)),
        simulate=True,
        outcome_listener=heard.append,
        **kwargs,
    )


@pytest.fixture
def session(tmp_path, heard):
    # Shut down before free_host resets the host, which it set up first.
    session = _create(tmp_path, heard)
    yield session
    session.shutdown()


def _told(heard):
    return [n for n in heard if n.title == 'Computer may sleep']


def test_a_session_holds_the_computer_awake_with_nothing_running(session):
    assert host.held == 1

    session.run_application_session()

    assert host.held == 1


def test_the_hold_taken_is_logged_once_in_the_host_s_words(tmp_path, heard, caplog):
    # The log is the product here: a field bundle's only evidence the request
    # was held, and of what the platform was asked to hold.
    caplog.set_level(logging.INFO)
    session = _create(tmp_path, heard)
    try:
        session.run_application_session()
    finally:
        session.shutdown()

    logged = [r.getMessage() for r in caplog.records if HOLDS in r.getMessage()]
    assert len(logged) == 1, logged


def test_a_run_a_recording_and_a_control_session_end_without_releasing_it(session, tmp_path):
    session.run_application_session()
    session.set_channel_acquire('BF', 'image')
    protocol = session.new_protocol(duration_s=0, period_s=0)

    run = session.protocol_runner().start_single_scan(protocol, parent_dir=str(tmp_path / 'runs'))
    assert run.result(timeout=SETTLE_S).status == 'completed'
    after_the_run = host.held
    recording = session.manual_recording.start()
    recording.stop()
    recording.result(timeout=SETTLE_S)
    after_the_recording = host.held
    session.end_control(session.take_control('plate reader'))

    assert (after_the_run, after_the_recording, host.held) == (1, 1, 1)


def test_the_close_releases_it_and_a_second_close_takes_nothing(session):
    session.shutdown()
    after_the_close = host.held
    session.shutdown()

    assert (after_the_close, host.held) == (0, 0)


def test_a_create_that_raises_after_the_session_is_built_leaves_nothing_held(tmp_path, heard):
    def fail_at_configuring(percent, step):
        if step == CONFIGURING:
            raise RuntimeError('bring-up failed while configuring')

    with pytest.raises(RuntimeError, match='while configuring'):
        _create(tmp_path, heard, on_progress=fail_at_configuring)

    assert host.held == 0


def test_a_request_the_host_refuses_is_told_and_the_session_comes_up(tmp_path, heard):
    host.refuse_with = AwakeNotHeld('keep_awake_refused', 'the host said no')

    session = _create(tmp_path, heard)
    try:
        session.run_application_session()
    finally:
        session.shutdown()

    told = _told(heard)
    assert [(n.kind, n.reason) for n in told] == [('notice', 'keep_awake_refused')], told
    assert 'the host said no' in told[0].message


def test_a_hold_the_host_ends_by_itself_is_told(session, heard):
    host.on_lost(AwakeNotHeld('keep_awake_refused', 'logind ended it'))

    told = _told(heard)
    assert [(n.kind, n.reason) for n in told] == [('notice', 'keep_awake_refused')], told
    assert 'logind ended it' in told[0].message


def test_a_release_the_host_refuses_leaves_the_close_complete(tmp_path, heard):
    session = _create(tmp_path, heard)
    host.refuse_release_with = OSError('the host kept it')

    session.shutdown()

    assert session.live_work.closed
