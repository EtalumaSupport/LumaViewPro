# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motor board bring-up recorded as missing is not homed at startup.

Bring-up reports the missing motor once, in its own report (partial
hardware, or the one notice when nothing came up). Startup motion that
then homed it anyway had the home refused as "not connected" and showed
that as a second popup for the same missing board, and logged the
skipped turret move at ERROR. The startup sequence reads the record and
issues no motion instead.
"""

from __future__ import annotations

import pytest

from drivers.registry import DriverFallback
from modules.exceptions import (
    HardwareCommandRefusedError,
    NoHardwareDetectedNotice,
    PartialHardwareError,
)
from modules.notification_center import OutcomeKind
from tests.test_bring_up_is_a_record import _bring_up


_NOT_DETECTED = DriverFallback('not_detected', ('MotorBoard',))


@pytest.mark.parametrize(
    ('others', 'report'),
    [
        pytest.param({}, (OutcomeKind.FAULT, PartialHardwareError('').reason), id='only-the-motor'),
        pytest.param(
            {
                'led': DriverFallback('not_detected', ('LEDBoard',)),
                'camera': FileNotFoundError('no camera'),
            },
            (OutcomeKind.NOTICE, NoHardwareDetectedNotice.reason),
            id='no-hardware',
        ),
    ],
)
def test_a_missing_motor_is_reported_once_and_never_homed(
    monkeypatch, tmp_path, centre_posts, others, report
):
    session = _bring_up(monkeypatch, tmp_path, motor=_NOT_DETECTED, microscope='LS850T', **others)
    homed, turret = [], []

    def _home(axis):
        # What the motion API answers for a board that is not there.
        homed.append(axis)
        raise HardwareCommandRefusedError('not_connected', 'home')

    try:
        session.start_application_session(home_fn=_home, turret_fn=turret.append)
    finally:
        session.shutdown()
        session.scope.disconnect()

    assert (homed, turret) == ([], [])
    assert [(n.kind, n.reason) for n in centre_posts] == [report]


def test_a_motor_that_came_up_is_homed(monkeypatch, tmp_path, centre_posts):
    session = _bring_up(monkeypatch, tmp_path, microscope='LS850T')
    homed = []
    try:
        session.start_application_session(home_fn=homed.append, turret_fn=lambda position: None)
    finally:
        session.shutdown()
        session.scope.disconnect()

    assert homed == ['ALL']
