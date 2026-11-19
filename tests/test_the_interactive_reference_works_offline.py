# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``/docs`` loads nothing from outside the server, so it works on a computer with no internet.

FastAPI's default page loaded Swagger UI's script and style from
``cdn.jsdelivr.net`` and its icon from ``fastapi.tiangolo.com``: on a scope
PC with no internet the interactive reference was blank. Every file it
loads is now served by the server itself, from the installation.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings

# Every address the page names: an attribute's, and any absolute URL in its script.
_NAMED = re.compile(r"""(?:src|href)=["']([^"']+)["']|(https?://[^\s"'<>)]+)""")


@pytest.fixture
def client(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        with TestClient(build_app(session)) as client:
            yield client
    finally:
        session.shutdown()


def test_the_reference_page_names_no_address_outside_the_server(client):
    page = client.get('/docs')

    named = [a or b for a, b in _NAMED.findall(page.text)]

    assert page.status_code == 200
    assert [n for n in named if not n.startswith('/')] == [], named
    # Its online validator would load an image from validator.swagger.io.
    assert '"validatorUrl": null' in page.text


def test_every_file_the_reference_page_loads_is_served(client):
    page = client.get('/docs')
    named = [a for a, _ in _NAMED.findall(page.text) if a]

    served = {n: client.get(n) for n in named}

    assert len(served) == 3, named
    assert {n: a.status_code for n, a in served.items()} == dict.fromkeys(named, 200)
    assert {a.headers['content-type'].split(';')[0] for a in served.values()} == {
        'text/javascript',
        'text/css',
        'image/png',
    }
