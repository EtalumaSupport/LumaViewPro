# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report's scope hands its motor board the shipped defaults.

`Lumascope.create_diagnostic` asks the registry for its boards, and the
motor driver takes the shipped motor defaults as a required argument.
Built without them, the constructor raised, the registry's broad catch
turned that into a fallback, and the support report ran against a null
motor board on every machine, telling the person to check a cable that
was fine.
"""

import modules.lumascope_api._lumascope as lumascope_module
from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS


def test_the_support_report_scope_builds_its_motor_board_with_the_shipped_defaults(
    monkeypatch,
):
    asked = []

    def motor_create(name='auto', **kwargs):
        asked.append(kwargs)
        return NullMotionBoard(), None

    monkeypatch.setattr(lumascope_module.motor_registry, 'create_with_fallback', motor_create)
    monkeypatch.setattr(
        lumascope_module.led_registry,
        'create_with_fallback',
        lambda name='auto', **kwargs: (NullLEDBoard(), None),
    )

    scope = lumascope_module.Lumascope.create_diagnostic()
    try:
        assert asked == [{'motorconfig_defaults': SHIPPED_MOTOR_DEFAULTS}]
    finally:
        scope.disconnect()
