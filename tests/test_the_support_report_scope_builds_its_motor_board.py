# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report's scope hands its motor board the shipped defaults.

`Lumascope.create_diagnostic` builds the boards itself, not through the
registry, and the motor driver takes the shipped motor defaults as a
required argument. Built without them, the constructor raised, the
connect helper's broad catch turned that into "connect failed", and the
support report ran against a null motor board on every machine, telling
the person to check a cable that was fine.
"""

import modules.lumascope_api._lumascope as lumascope_module
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS


class _NoBoard:
    """Answers "not on USB" without touching a port."""

    def __init__(self, **kwargs):
        self.found = False


def test_the_support_report_scope_builds_its_motor_board_with_the_shipped_defaults(
    monkeypatch,
):
    built = []

    def motor_board(**kwargs):
        built.append(kwargs)
        return _NoBoard()

    monkeypatch.setattr(lumascope_module, 'MotorBoard', motor_board)
    monkeypatch.setattr(lumascope_module, 'LEDBoard', _NoBoard)

    scope = lumascope_module.Lumascope.create_diagnostic()
    try:
        assert built == [{'motorconfig_defaults': SHIPPED_MOTOR_DEFAULTS}]
    finally:
        scope.disconnect()
