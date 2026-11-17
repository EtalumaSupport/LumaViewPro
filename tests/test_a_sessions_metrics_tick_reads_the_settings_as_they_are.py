# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A session's system metrics tick reads the session's settings as they are when it ticks.

The tick runs on the session's timer and logs the disk free under the live
folder; it takes a copy of the settings at each tick, so a live folder moved
after the session was built is the one the next tick measures, never the one
from construction and never no folder at all.
"""

import psutil
import pytest

from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield session
    finally:
        session.shutdown()


def test_a_tick_measures_the_live_folder_set_after_the_session_was_built(
    session, monkeypatch, tmp_path
):
    probed = []
    real_disk_usage = psutil.disk_usage

    def recorded(path):
        probed.append(path)
        return real_disk_usage(path)

    # a stand-in by design: the host's disk probe, an interchangeable
    # collaborator of the tick under test, recording which folder it asked about.
    monkeypatch.setattr(psutil, 'disk_usage', recorded)
    moved = tmp_path / 'moved'
    moved.mkdir()
    session.set_live_folder(str(moved))

    session.metrics_logger.tick_system_metrics()

    assert probed[0] == str(moved)
