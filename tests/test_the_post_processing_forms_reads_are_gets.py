# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The post-processing forms' two reads are served as reads.

``PostProcessingAPI.graph_options`` and ``load_cell_count_method`` change
nothing -- one answers a graph form's choices, the other a saved method --
yet were served as ``POST``, so a wire client could not tell them from the
writes beside them. They are marked reads and served ``GET``, their
arguments in the query. ``cell_count_size_scale`` stays ``POST``: its
argument is a record, which has no query form.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def openapi(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        with TestClient(build_app(session)) as client:
            yield client.get('/api/v1/openapi.json').json()['paths']
    finally:
        session.shutdown()


def test_the_forms_reads_are_gets_and_the_record_taker_a_post(openapi):
    served = {
        name: sorted(openapi[f'/api/v1/post_processing/{name}'])
        for name in ('graph_options', 'load_cell_count_method', 'cell_count_size_scale')
    }

    assert served == {
        'graph_options': ['get'],
        'load_cell_count_method': ['get'],
        'cell_count_size_scale': ['post'],
    }
