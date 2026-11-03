# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``rest_api.allowed_clients`` serves only the clients it names, on every route, whoever serves the app.

The guard is in ``rest.app.build_app``, so a program that serves the app
itself gets it as the builder's hosts do: a peer in no entry is refused
``403 client_not_allowed`` before any route sees the request, its answer
carrying the request's id. The peer is read from the connection alone: the
builder trusts no header naming another, whatever ``FORWARDED_ALLOW_IPS``
says. Empty, the list admits every client that reaches the address.
"""

from __future__ import annotations

import socket
import threading

import httpx
import pytest
from starlette.testclient import TestClient

from modules.scope_session import ScopeSession
from rest.app import PREFIX, build_app
from rest.problems import TYPE_PREFIX
from rest.server import build_server
from tests.settings_fixtures import complete_settings

# Not this computer's: build_app judges the section and binds nothing.
_SCOPE_PC = '192.168.10.2'
_ALLOWED = '192.168.10.3'
_REFUSED = TYPE_PREFIX + 'client_not_allowed'

# Every route kind a client reaches: the versions, a read, a write, the
# event stream, the live view, a file download, the description and the
# interactive reference.
_ROUTES = [
    ('GET', '/api'),
    ('GET', f'{PREFIX}/status'),
    ('POST', f'{PREFIX}/scope/illumination/leds_off'),
    ('GET', f'{PREFIX}/events'),
    ('GET', f'{PREFIX}/live'),
    ('GET', f'{PREFIX}/files/a.tiff'),
    ('GET', f'{PREFIX}/openapi.json'),
    ('GET', '/docs'),
]


def _session(tmp_path, rest_api: dict) -> ScopeSession:
    live = tmp_path / 'live'
    live.mkdir(exist_ok=True)
    settings = complete_settings(live_folder=str(live))
    settings['rest_api'] = {'enabled': True, 'port': 8000, **rest_api}
    return ScopeSession.create(settings, simulate=True)


@pytest.fixture
def guarded(tmp_path):
    session = _session(
        tmp_path, {'address': _SCOPE_PC, 'allowed_clients': [_ALLOWED, '10.20.0.0/16']}
    )
    yield build_app(session)
    session.shutdown()


def test_a_client_in_no_entry_is_refused_on_every_route(guarded):
    wrong = []
    with TestClient(guarded, client=('192.168.10.9', 50000)) as client:
        for method, path in _ROUTES:
            answer = client.request(method, path)
            body = answer.json()
            if (
                answer.status_code != 403
                or body['type'] != _REFUSED
                or '192.168.10.9' not in body['detail']
                or body['instance'] != f'urn:uuid:{answer.headers.get("X-Request-ID")}'
            ):
                wrong.append((method, path, answer.status_code, body))
    assert wrong == []


def test_a_client_named_or_inside_a_named_network_is_served(guarded):
    for peer in (_ALLOWED, '10.20.4.7'):
        with TestClient(guarded, client=(peer, 50000)) as client:
            assert client.get(f'{PREFIX}/status').status_code == 200, peer


def test_with_no_allowlist_every_client_is_served(tmp_path):
    session = _session(tmp_path, {'address': _SCOPE_PC, 'allowed_clients': []})
    try:
        with TestClient(build_app(session), client=('172.16.9.9', 50000)) as client:
            assert client.get(f'{PREFIX}/status').status_code == 200
    finally:
        session.shutdown()


def _this_computers_address() -> str:
    """An IPv4 address of this computer's other than loopback, the one it would send from."""
    # A datagram socket's connect sends nothing; it only picks the route.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(('192.0.2.1', 9))
        except OSError:
            pytest.skip('this computer has no network address but loopback')
        return probe.getsockname()[0]


def _free_port(address: str, family: socket.AddressFamily) -> int:
    with socket.create_server((address, 0), family=family) as sock:
        return sock.getsockname()[1]


def _serving(session: ScopeSession):
    server = build_server(session, shut_down=session.shutdown)
    thread = threading.Thread(target=server.run)
    thread.start()
    while not server.serving:
        assert thread.is_alive(), 'the server ended before it served'
        thread.join(0.01)
    return server, thread


def test_a_header_naming_an_allowed_client_does_not_admit_a_refused_peer(tmp_path, monkeypatch):
    """Served on this computer's own network address and asked from it, so the peer is that address."""
    # uvicorn's own fallback would trust a header from any peer.
    monkeypatch.setenv('FORWARDED_ALLOW_IPS', '*')
    here = _this_computers_address()
    session = _session(
        tmp_path,
        {
            'address': here,
            'port': _free_port(here, socket.AF_INET),
            'allowed_clients': ['10.255.255.1'],
        },
    )
    server, thread = _serving(session)
    try:
        answer = httpx.get(
            f'{server.reach.url}/status',
            headers={'X-Forwarded-For': '10.255.255.1', 'Forwarded': 'for=10.255.255.1'},
            timeout=10,
        )
    finally:
        server.close()
        thread.join()
    assert answer.status_code == 403
    assert answer.json()['type'] == _REFUSED


def test_a_client_at_the_one_allowed_address_is_served_and_the_reach_says_so(tmp_path):
    here = _this_computers_address()
    port = _free_port(here, socket.AF_INET)
    session = _session(tmp_path, {'address': here, 'port': port, 'allowed_clients': [here]})
    server, thread = _serving(session)
    try:
        reach = server.reach
        answer = httpx.get(f'{reach.url}/status', timeout=10)
    finally:
        server.close()
        thread.join()
    assert answer.status_code == 200
    assert (reach.url, str(reach)) == (
        f'http://{here}:{port}/api/v1',
        f'{here}:{port}, HTTP, 1 allowed client',
    )


def test_an_ipv6_address_is_served(tmp_path):
    port = _free_port('::1', socket.AF_INET6)
    session = _session(tmp_path, {'address': '::1', 'port': port, 'allowed_clients': []})
    server, thread = _serving(session)
    try:
        reach = server.reach
        answer = httpx.get(f'{reach.url}/status', timeout=10)
    finally:
        server.close()
        thread.join()
    assert answer.status_code == 200
    assert (reach.url, str(reach)) == (
        f'http://[::1]:{port}/api/v1',
        f'[::1]:{port}, HTTP, this computer only',
    )
