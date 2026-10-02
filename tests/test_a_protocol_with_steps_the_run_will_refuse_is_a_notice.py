# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol holding a step the run will refuse is told once, by the API, as a notice.

The file loader validates no step field and the edit members compose a
step from whatever the live settings hold, so a protocol could carry an
exposure of 0 to the run start with nothing said before. The protocol
panel used to notice after its own edits, with a popup it composed. The
API now reports one notice where the state arises -- at load, and after
an add or an update -- and the load and the edit still succeed, the way
the loader already treats steps that would share a filename.
"""

from __future__ import annotations

import pytest

import modules.config_helpers as config_helpers
from modules.notification_center import notifications
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture


@pytest.fixture
def reported(monkeypatch):
    reports = []
    real = notifications.report_outcome

    def spy(exc, **kw):
        reports.append((type(exc).__name__, kw['category']))
        return real(exc, **kw)

    monkeypatch.setattr(notifications, 'report_outcome', spy)
    return reports


def _acquire_only_bf(session):
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'


def test_a_loaded_file_with_an_invalid_step_loads_and_is_noticed_once(session, reported, tmp_path):
    _acquire_only_bf(session)
    protocol = session.new_protocol()
    protocol.steps().at[0, 'Exposure'] = 0.0
    path = tmp_path / 'bad.tsv'
    protocol.to_file(path)
    reported.clear()

    loaded = session.load_protocol(path)

    assert loaded.num_steps() == protocol.num_steps()
    assert reported == [('ProtocolStepsInvalidNotice', 'Protocol')]


def test_an_add_from_an_invalid_live_setting_adds_and_is_noticed_once(session, reported):
    _acquire_only_bf(session)
    protocol = session.create_empty_protocol()
    session.settings['BF']['exposure_ms'] = 0.0
    reported.clear()

    names = session.add_step(protocol, before_step=0)

    assert len(names) == 1
    assert reported == [('ProtocolStepsInvalidNotice', 'Protocol')]


def test_a_valid_edit_and_a_valid_load_report_nothing(session, reported, tmp_path):
    _acquire_only_bf(session)
    protocol = session.create_empty_protocol()
    session.add_step(protocol, before_step=0)
    path = tmp_path / 'good.tsv'
    protocol.to_file(path)
    reported.clear()

    session.update_step(protocol, 0, layer='BF')
    session.load_protocol(path)

    assert reported == []
