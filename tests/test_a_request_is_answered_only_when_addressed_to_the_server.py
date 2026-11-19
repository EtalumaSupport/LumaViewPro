# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The served REST server answers only a request addressed to it, and none a page from another site sent.

Before, any web page the user opened could drive the scope: a page posts
to ``127.0.0.1`` without reading the answer, and a site whose name it
pointed at the scope's address (DNS rebinding) could read the answers too.
``build_server`` serves the application ``bound``, so a request whose
``Host`` is not the server's address and port (or ``localhost`` at that
port, on loopback) is refused ``421 wrong_host``, and one whose ``Origin``
is another site's ``403 foreign_origin``, before any route sees it. Served
on the real stack, since the check is the bound server's alone: a test
client or a mount is not served at that address.
"""

from __future__ import annotations

import socket
import threading

import httpx
import pytest

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import PREFIX
from _lumascope.rest.server import build_server
from tests.settings_fixtures import complete_settings


def _free_port(address: str, family: socket.AddressFamily) -> int:
    # The judge refuses port 0, so the test asks the system for a free one.
    with socket.create_server((address, 0), family=family) as sock:
        return sock.getsockname()[1]


@pytest.fixture(
    params=[('127.0.0.1', socket.AF_INET, '127.0.0.1'), ('::1', socket.AF_INET6, '[::1]')],
    ids=['ipv4', 'ipv6'],
)
def served(request, tmp_path):
    """A simulated session served by the one builder on a loopback address, as ``(url, host, port)``."""
    address, family, host = request.param
    port = _free_port(address, family)
    settings = complete_settings(live_folder=str(tmp_path))
    settings['rest_api'] = {
        'enabled': True,
        'address': address,
        'port': port,
        'allowed_clients': [],
    }
    session = ScopeSession.create(settings, simulate=True)
    server = build_server(session, shut_down=session.shutdown)
    thread = threading.Thread(target=server.run)
    thread.start()
    while not server.serving:
        assert thread.is_alive(), 'the server ended before it served'
        thread.join(0.01)
    try:
        yield f'http://{host}:{port}{PREFIX}', host, port
    finally:
        server.close()
        thread.join()


def test_a_foreign_host_or_origin_is_refused_before_any_route(served):
    url, host, port = served

    named_elsewhere = httpx.get(f'{url}/status', headers={'Host': f'scope.example:{port}'})
    another_port = httpx.get(f'{url}/status', headers={'Host': f'{host}:{port + 1}'})
    from_another_site = httpx.post(
        f'{url}/scope/illumination/leds_off', headers={'Origin': 'http://site.example'}
    )
    from_no_site = httpx.post(f'{url}/scope/illumination/leds_off', headers={'Origin': 'null'})

    for answer in (named_elsewhere, another_port):
        assert (answer.status_code, answer.json().get('reason')) == (421, 'wrong_host'), answer.text
        assert answer.json()['instance'] == f'urn:uuid:{answer.headers["X-Request-ID"]}'
    for answer in (from_another_site, from_no_site):
        assert (answer.status_code, answer.json().get('reason')) == (403, 'foreign_origin'), (
            answer.text
        )


def test_the_servers_own_address_in_any_spelling_and_its_own_pages_are_served(served):
    url, host, port = served
    spellings = {
        '127.0.0.1': ['127.0.0.1', 'localhost'],
        '[::1]': ['[::1]', '[0:0:0:0:0:0:0:1]', 'localhost'],
    }[host]

    reads = {s: httpx.get(f'{url}/status', headers={'Host': f'{s}:{port}'}) for s in spellings}
    from_its_own_page = httpx.post(
        f'{url}/scope/illumination/leds_off', headers={'Origin': f'http://{host}:{port}'}
    )

    assert {s: a.status_code for s, a in reads.items()} == dict.fromkeys(spellings, 200)
    assert from_its_own_page.status_code == 200, from_its_own_page.text
