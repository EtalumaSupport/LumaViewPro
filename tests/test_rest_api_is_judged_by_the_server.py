# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``rest_api`` has one judge, the server's (``rest.server.build_server``), and fails closed.

A ``rest_api`` the server cannot use is refused with
``RestServerNotStartedError`` naming each value, before anything is bound
or started, from every host: the builder a program calls, the GUI's
``serve_beside``, and ``python -m rest``, which prints it and exits 1. The
load neither judges nor replaces the section, so a session comes up on it
and a save writes back what the user wrote.
"""

from __future__ import annotations

import json
import threading

import pytest

from modules.exceptions import RestServerNotStartedError
from modules.scope_session import ScopeSession
from rest.server import build_server, serve_beside
from tests.settings_fixtures import complete_settings, installation_of_its_own

# A section the server serves: this computer only, every client.
_SERVABLE = {'enabled': True, 'port': 8000, 'address': '127.0.0.1', 'allowed_clients': []}

# Each section the server cannot use, as its changes to a servable one, and
# the settings its refusal names.
_UNUSABLE = [
    ({'enabled': 'true'}, ['rest_api.enabled']),
    ({'enabled': None}, ['rest_api.enabled']),
    ({'port': 70000}, ['rest_api.port']),
    ({'port': 1023}, ['rest_api.port']),
    ({'port': 8000.5}, ['rest_api.port']),
    ({'port': True}, ['rest_api.port']),
    ({'port': '8000'}, ['rest_api.port']),
    ({'enabled': 1, 'port': -1}, ['rest_api.enabled', 'rest_api.port']),
    # Every network the computer is on, in each spelling.
    ({'address': '0.0.0.0'}, ['rest_api.address']),
    ({'address': '::'}, ['rest_api.address']),
    ({'address': '::ffff:0.0.0.0'}, ['rest_api.address']),
    # A name, not the one address served.
    ({'address': 'localhost'}, ['rest_api.address']),
    ({'address': None}, ['rest_api.address']),
    ({'address': 'fe80::1%en0'}, ['rest_api.address']),
    (
        {'address': '192.168.10.2', 'allowed_clients': ['192.168.10.3', 'lab-pc']},
        ['rest_api.allowed_clients', "'lab-pc'"],
    ),
    ({'address': '192.168.10.2', 'allowed_clients': [3]}, ['rest_api.allowed_clients']),
    # An address under a prefix is never widened to its network.
    (
        {'address': '192.168.10.2', 'allowed_clients': ['192.168.10.5/24']},
        ['rest_api.allowed_clients', "'192.168.10.5/24'"],
    ),
    # No other computer reaches a loopback address.
    ({'allowed_clients': ['192.168.10.3']}, ['rest_api.allowed_clients', 'loopback']),
    # A client of the other family could never be served.
    (
        {'address': 'fd00::2', 'allowed_clients': ['192.168.10.3']},
        ['rest_api.allowed_clients', 'IPv4'],
    ),
]


def _session_with(tmp_path, rest_api: dict) -> ScopeSession:
    live = tmp_path / 'live'
    live.mkdir(exist_ok=True)
    settings = complete_settings(live_folder=str(live))
    settings['rest_api'] = dict(rest_api)
    return ScopeSession.create(settings, simulate=True)


def _never_shut_down() -> None:
    raise AssertionError('a server that was not started closed the session')


def test_a_rest_api_the_server_cannot_use_is_refused_by_the_builder_and_nothing_starts(tmp_path):
    wrong = []
    for changes, named in _UNUSABLE:
        section = {**_SERVABLE, **changes}
        # The load leaves the section as handed: the session comes up on it.
        session = _session_with(tmp_path, section)
        try:
            assert session.get_setting('rest_api') == section
            threads_before = set(threading.enumerate())
            try:
                build_server(session, shut_down=_never_shut_down)
            except RestServerNotStartedError as e:
                said = str(e)
                if e.reason != 'rest_server_not_started' or not all(p in said for p in named):
                    wrong.append((section, e.reason, said))
            else:
                wrong.append((section, 'served'))
            if not set(threading.enumerate()) <= threads_before:
                wrong.append((section, 'a thread started'))
        finally:
            session.shutdown()
    assert wrong == []


def test_the_guis_start_refuses_it_where_the_start_is(tmp_path):
    session = _session_with(tmp_path, {**_SERVABLE, 'port': 70000})
    try:
        with pytest.raises(RestServerNotStartedError) as refused:
            serve_beside(session, shut_down=_never_shut_down)
    finally:
        session.shutdown()
    assert refused.value.reason == 'rest_server_not_started'
    assert 'rest_api.port' in str(refused.value)


def _installation_with(rest_api: dict):
    """An installation whose user file is the shipped one with *rest_api* written in."""
    root = installation_of_its_own()
    current = complete_settings()
    current.update(lvp_lock_port=0, rest_api=rest_api)
    (root / 'data' / 'current.json').write_text(json.dumps(current))
    return root


def test_the_headless_server_prints_it_and_exits_1(monkeypatch, capsys):
    import modules.path_utils
    from rest.__main__ import main

    root = _installation_with({**_SERVABLE, 'enabled': False, 'port': 70000})
    monkeypatch.setattr(modules.path_utils, 'get_source_root', lambda *_args: root)

    assert main(['--simulate']) == 1
    said = capsys.readouterr()
    assert 'rest_api.port' in said.err
    assert 'REST server:' not in said.out


def test_a_save_writes_back_the_rest_api_the_user_wrote():
    written = {**_SERVABLE, 'address': '0.0.0.0', 'allowed_clients': ['lab-pc']}
    root = _installation_with(written)
    settings_file = ScopeSession.user_settings_file(root)
    session = ScopeSession.create(
        ScopeSession.load_user_settings(str(root)),
        source_path=str(root),
        simulate=True,
        warn_pre_release=False,
        settings_file=settings_file,
    )
    try:
        session.update_settings('jpg_quality', 90)
    finally:
        session.shutdown()

    with open(settings_file, encoding='utf-8') as f:
        saved = json.load(f)
    assert saved['jpg_quality'] == 90
    assert saved['rest_api'] == written
