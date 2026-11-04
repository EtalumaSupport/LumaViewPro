# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulator the app and REST run takes a real camera's time to deliver a frame.

The shipped settings choose the realistic simulator tier, where a move and a
home take a stage's time, but the simulated camera always ran in fast
timing, delivering each frame the instant it was made. A frame then never
waited where a real one does, so the stale frames a real camera returns
did not appear in the app's simulator. The camera now follows the tier.
"""

from __future__ import annotations

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _camera_on(tier: str):
    session = ScopeSession.create(
        complete_settings(simulator_tier=tier), simulate=True, warn_pre_release=False
    )
    try:
        # A stand-in by design: no public member says how long the
        # simulated link takes, and the preset is the simulator's own.
        return session.scope._camera_driver._grab_delay
    finally:
        session.shutdown()


@pytest.mark.slow
def test_the_realistic_tier_delivers_with_a_cameras_transfer_time():
    assert _camera_on('realistic') > 0


def test_the_fast_tier_delivers_at_once():
    assert _camera_on('fast') == 0.0
