# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Precision is Z's: ``MotionAPI.set_precision_mode`` takes no axis and sets Z's.

The member took any axis, and on X, Y or T "precision on" wrote the tight
stop threshold over a tighter resting one; every caller passed Z. It now
sets Z's threshold alone, and a wire client still naming an axis is refused
as a body the member does not take.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings
from tests.test_a_command_for_absent_motion_hardware_is_refused import _spy_wire


@pytest.fixture
def session(tmp_path):
    # The firmware tier runs the production motor driver, so the board's
    # register write is what the test reads.
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), simulator_tier='firmware'), simulate=True
    )
    yield s
    s.shutdown()


def test_precision_writes_zs_stop_threshold_and_nothing_else(session, monkeypatch):
    sent = _spy_wire(monkeypatch, session.scope)

    session.scope.motion.set_precision_mode(False)
    session.scope.motion.set_precision_mode(True)

    # Z's VSTOP register (0x4B on the Z/T chip, written at 0xCB): loose, then tight.
    assert sent == ['SPIZ0xcb1000', 'SPIZ0xcb100']


def test_a_wire_client_naming_an_axis_is_refused(session):
    with TestClient(build_app(session)) as client:
        named = client.post(
            '/api/v1/scope/motion/set_precision_mode', json={'axis': 'Z', 'enabled': True}
        )
        unnamed = client.post('/api/v1/scope/motion/set_precision_mode', json={'enabled': True})

    assert named.status_code == 422
    assert [tuple(e['loc']) for e in named.json()['errors']] == [('body', 'axis')]
    assert unnamed.status_code == 200
