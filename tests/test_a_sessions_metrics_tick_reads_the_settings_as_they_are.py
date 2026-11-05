# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A session's system metrics tick reads the session's settings as they are when it ticks.

The tick runs on the session's timer and logs the disk free under the live
folder; it takes a copy of the settings at each tick, so a live folder moved
after the session was built is the one the next tick measures, never the one
from construction and never no folder at all.
"""

from collections import defaultdict

import pytest

from _lumascope.modules import config_helpers
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
    # a stand-in by design: the disk and process probes are the host's,
    # interchangeable collaborators of the tick under test; this one records
    # which folder the tick asked about.
    monkeypatch.setattr(
        config_helpers.common_utils,
        'check_disk_space',
        lambda *, path: probed.append(path) or 1.0e5,
    )
    monkeypatch.setattr(
        config_helpers.common_utils, 'system_metrics', lambda **k: defaultdict(float)
    )
    moved = tmp_path / 'moved'
    session.set_live_folder(str(moved))

    session.metrics_logger.tick_system_metrics()

    assert probed == [str(moved.resolve())]
