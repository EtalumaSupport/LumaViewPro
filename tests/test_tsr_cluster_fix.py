# Copyright Etaluma, Inc.
"""Regression tests for the 2026-05-14 TSR cluster fix.

Per Eric's bench report on SN12062 (LS850 with EL-0940-05 running legacy
2024-09-10 firmware), three classes of bugs in the tech-support report
violated the principle of least astonishment:

1. Driver capability gating was absent -- the TSR sent raw
   DRVSTAT_<axis> / FANSPEED / FAN:<duty> commands and forwarded the
   firmware's raw "ERROR: command 'X' not found:" responses up to the
   user. Per Eric: "shouldn't the TSR be using the driver, and the
   DRIVER should gate on the firmware version? (isn't that the entire
   point of the driver)?"

2. `_motor_ok()` and `_led_ok()` were Wave 7 leftovers --
   `getattr(scope.motion|scope.led, 'found', False)` always returned
   False post-rename because the sub-API namespaces don't have a
   `.found` attribute. Same shape as #648's `hasattr(self, 'camera')`
   regression.

3. A hardware-check writer reported "Overall: PASS" over readings it
   had not actually measured -- words on screen contradicted what the
   user could see on the same page. (The check that surfaced this,
   the power-rail voltage tolerance report, was later removed
   entirely as not useful.)

These tests assert the structural fix: parsed driver methods that
return None for unsupported firmware, capability-aware TSR steps
that render INCONCLUSIVE instead of PASS / FAIL.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# MotorBoard driver-method tests -- the FW-version gating layer.
# ---------------------------------------------------------------------------


class _FakeMotorBoard:
    """Just-enough MotorBoard surface for testing the diagnostic methods.

    Patches `exchange_command` so we can drive the firmware-response side
    without standing up a real serial board. Inherits the real diagnostic
    methods from MotorBoard.
    """

    def __init__(self, response_map):
        from drivers.motorboard import MotorBoard

        self._response_map = response_map
        self._real_methods = MotorBoard

    def exchange_command(self, command, *args, **kwargs):
        return self._response_map.get(command)


def _make_motor_with_responses(response_map):
    """Build an object that has the real diagnostic methods bound, but
    a stubbed exchange_command."""
    from drivers.motorboard import MotorBoard

    fake = _FakeMotorBoard(response_map)
    # Bind the diagnostic methods we care about
    for name in (
        '_diagnostic_query',
        'read_drv_status',
        'read_fanspeed',
        'set_fan_duty',
    ):
        method = getattr(MotorBoard, name)
        setattr(fake, name, method.__get__(fake, _FakeMotorBoard))
    return fake


class TestMotorDriverDiagnosticGating:
    """Driver methods return None on legacy FW, parsed values on new FW."""

    def test_read_drv_status_returns_int(self):
        motor = _make_motor_with_responses(
            {
                'DRVSTAT_X': '0x80000000',
            }
        )
        assert motor.read_drv_status('X') == 0x80000000

    def test_read_drv_status_unsupported_returns_none(self):
        motor = _make_motor_with_responses(
            {
                'DRVSTAT_X': "ERROR: command 'DRVSTAT_X' not found:",
            }
        )
        assert motor.read_drv_status('X') is None

    def test_read_drv_status_invalid_axis_raises(self):
        motor = _make_motor_with_responses({})
        with pytest.raises(ValueError):
            motor.read_drv_status('Q')

    def test_read_fanspeed_unsupported_returns_none(self):
        motor = _make_motor_with_responses(
            {
                'FANSPEED': "ERROR: command 'FANSPEED' not found:",
            }
        )
        assert motor.read_fanspeed() is None

    def test_read_fanspeed_parses_int(self):
        motor = _make_motor_with_responses({'FANSPEED': '1234'})
        assert motor.read_fanspeed() == 1234

    def test_set_fan_duty_unsupported_returns_false(self):
        motor = _make_motor_with_responses(
            {
                'FAN:50': "ERROR: command 'FAN' not found:",
            }
        )
        assert motor.set_fan_duty(50) is False

    def test_set_fan_duty_supported_returns_true(self):
        motor = _make_motor_with_responses({'FAN:50': 'OK'})
        assert motor.set_fan_duty(50) is True

    def test_set_fan_duty_invalid_value_raises(self):
        motor = _make_motor_with_responses({})
        with pytest.raises(ValueError):
            motor.set_fan_duty(150)


# ---------------------------------------------------------------------------
# DiagnosticsAPI sub-API -- thin delegation that handles missing driver.
# ---------------------------------------------------------------------------


class TestDiagnosticsApiDelegation:
    """The sub-API delegates to driver; returns sentinel when driver absent."""

    def test_set_motor_fan_duty_no_driver_returns_false(self):
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        scope = MagicMock(spec=[])
        api = DiagnosticsAPI(scope)
        assert api.set_motor_fan_duty(50) is False


# ---------------------------------------------------------------------------
# _motor_ok / _led_ok use post-Wave-7 connection probes.
# ---------------------------------------------------------------------------


class TestMotorAndLedConnectionProbes:
    """The Wave-7-renamed sub-API namespaces don't expose .found; the
    fix uses scope.motor_connected / scope.led_connected live properties."""

    def test_motor_ok_true_via_live_property(self):
        from modules.tech_support_report import FirmwareDiagnostics

        scope = MagicMock()
        scope.motor_connected = True
        diag = FirmwareDiagnostics(scope=scope)
        assert diag._motor_ok() is True

    def test_motor_ok_false_via_live_property(self):
        from modules.tech_support_report import FirmwareDiagnostics

        scope = MagicMock()
        scope.motor_connected = False
        diag = FirmwareDiagnostics(scope=scope)
        assert diag._motor_ok() is False

    def test_motor_ok_fallback_to_driver_found(self):
        # Older diagnostic Lumascope shapes may lack the live property;
        # fall back to the underlying driver's found attribute.
        from modules.tech_support_report import FirmwareDiagnostics

        scope = MagicMock(spec=['_motion_driver'])
        scope._motion_driver = MagicMock()
        scope._motion_driver.found = True
        diag = FirmwareDiagnostics(scope=scope)
        assert diag._motor_ok() is True

    def test_led_ok_uses_post_wave7_illumination_or_live(self):
        from modules.tech_support_report import FirmwareDiagnostics

        scope = MagicMock()
        scope.led_connected = True
        diag = FirmwareDiagnostics(scope=scope)
        assert diag._led_ok() is True

    def test_target_str_resolves_illumination_not_legacy_led(self):
        # Post-Wave-7 the LED sub-API namespace is `scope.illumination`,
        # not `scope.led`. _target_str must resolve a board object to its
        # canonical 'led'/'motor' string via the renamed namespace so
        # _cmd(self.led_board, ...) routes correctly.
        from modules.tech_support_report import FirmwareDiagnostics

        illum = object()
        motion = object()
        legacy_led = object()
        scope = MagicMock()
        scope.illumination = illum
        scope.motion = motion
        scope.led = legacy_led  # the retired attribute
        diag = FirmwareDiagnostics(scope=scope)

        assert diag._target_str(illum) == 'led'
        assert diag._target_str(motion) == 'motor'
        # A reference to the retired scope.led object must NOT resolve as
        # the LED target -- only scope.illumination does.
        assert diag._target_str(legacy_led) is None
        # String targets pass through unchanged.
        assert diag._target_str('led') == 'led'


# ---------------------------------------------------------------------------
# TSR call-site sanity: no raw VOLTAGE / DRVSTAT / FANSPEED command sends.
# ---------------------------------------------------------------------------


class TestTsrUsesDriverMethods:
    """After the cluster fix, TSR's diagnostic primitives go through the
    DiagnosticsAPI sub-API, not raw `_cmd(motor_board, 'VOLTAGE')` style
    sends. The raw paths violated Rule 22 (use production code paths)."""

    def test_no_raw_drvstat_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, f'DRVSTAT_" not in src

    def test_no_raw_fanspeed_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, 'FANSPEED')" not in src

    def test_no_raw_fan_duty_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, 'FAN:" not in src
