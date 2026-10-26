# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A change to the settings store is announced once its write has let go of the store.

Every live write goes through ``_store_setting``, under ``settings_lock``,
and several members write two to five paths in one hold. The lock records
each path stored under a hold and, once the hold is released, announces
them together as one ``SettingsChanged`` to every ``add_settings_listener``
listener, on the writer's thread: never mid-batch, never under the lock,
never for a hold that stored nothing. A REST client hears it on the event
stream as ``settings``, so a display beside another writer -- the GUI
beside a client, or a client beside the GUI -- shows what is stored.
"""

from __future__ import annotations

import contextlib
import json
import threading

import httpx
import pytest
import uvicorn

from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


def test_a_write_of_several_paths_in_one_hold_is_announced_once_after_the_release(session):
    heard = []

    def listener(paths):
        # Told outside the lock: a listener may read the store.
        heard.append((paths, session.settings_lock.locked(), session.channel_acquire(LED0)))

    session.add_settings_listener(listener)
    try:
        # A layer set to acquire stops stimulating: two paths in one hold.
        session.set_channel_acquire(LED0, 'image')
        # A hold that stored nothing -- a reader's -- tells nothing.
        with session.settings_lock:
            pass
    finally:
        session.remove_settings_listener(listener)

    ((paths, locked, acquire),) = heard
    assert set(paths) == {f'{LED0}.acquire', f'{LED0}.stim_config.enabled'}
    assert (locked, acquire) == (False, 'image')


def test_a_removed_listener_is_told_nothing(session):
    heard = []
    session.add_settings_listener(heard.append)
    session.remove_settings_listener(heard.append)
    session.update_settings('jpg_quality', 80)

    assert heard == []


@contextlib.contextmanager
def _serving(session):
    server = uvicorn.Server(
        uvicorn.Config(build_app(session), host='127.0.0.1', port=0, log_level='warning')
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            assert thread.is_alive(), 'the server ended before it started'
            thread.join(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def _events_until(lines, name: str) -> list[dict]:
    seen: list[dict] = []
    event: dict = {}
    for line in lines:
        if not line:
            if event:
                seen.append(event)
                if event['event'] == name:
                    return seen
                event = {}
            continue
        if line.startswith(':'):
            continue
        field, _, value = line.partition(': ')
        event[field] = json.loads(value) if field == 'data' else value
    raise AssertionError(f'the stream ended before {name}')


def test_a_client_hears_a_change_another_writer_made(session):
    with _serving(session) as client, client.stream('GET', '/api/v1/events') as response:
        lines = response.iter_lines()
        _events_until(lines, 'status')
        # The GUI's write, or a script's: not this client's.
        session.update_settings('jpg_quality', 80)
        told = _events_until(lines, 'settings')[-1]

    assert told['data'] == {'paths': ['jpg_quality']}
