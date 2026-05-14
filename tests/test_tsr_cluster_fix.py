# Copyright Etaluma, Inc.
"""Regression tests for the 2026-05-14 TSR cluster fix.

Per Eric's bench report on SN12062 (LS850 with EL-0940-05 running legacy
2024-09-10 firmware), three classes of bugs in the tech-support report
violated the principle of least astonishment:

1. Driver capability gating was absent -- the TSR sent raw VOLTAGE /
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

3. The voltage check writer reported "Overall: PASS" with three rails
   showing UNKNOWN readings -- words on screen contradicted what the
   user could see on the same page. Per
   `feedback_least_astonishment`.

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
        '_diagnostic_query', 'read_voltages', 'read_drv_status',
        'read_fanspeed', 'set_fan_duty',
    ):
        method = getattr(MotorBoard, name)
        setattr(fake, name, method.__get__(fake, _FakeMotorBoard))
    return fake


class TestMotorDriverDiagnosticGating:
    """Driver methods return None on legacy FW, parsed values on new FW."""

    def test_read_voltages_unsupported_firmware_returns_none(self):
        motor = _make_motor_with_responses({
            'VOLTAGE': "ERROR: command 'VOLTAGE' not found:",
        })
        assert motor.read_voltages() is None

    def test_read_voltages_supported_firmware_parses_dict(self):
        motor = _make_motor_with_responses({
            'VOLTAGE': '24V=OK 5V=5.18 3V3=3.31 1V2=1.24',
        })
        result = motor.read_voltages()
        assert result is not None
        assert result['5V'] == pytest.approx(5.18)
        assert result['3.3V'] == pytest.approx(3.31)
        assert result['1.2V'] == pytest.approx(1.24)
        # 24V uses 'OK' sentinel (non-numeric) -> None per rail
        assert result['24V'] is None

    def test_read_voltages_per_rail_unparseable(self):
        # Firmware accepted VOLTAGE but every rail came back N/A --
        # this is the "supported but no readings" path the writer
        # must render as INCONCLUSIVE.
        motor = _make_motor_with_responses({
            'VOLTAGE': '24V=OK 5V=N/A 3V3=N/A 1V2=N/A',
        })
        result = motor.read_voltages()
        assert result is not None
        assert all(v is None for v in result.values())

    def test_read_drv_status_returns_int(self):
        motor = _make_motor_with_responses({
            'DRVSTAT_X': '0x80000000',
        })
        assert motor.read_drv_status('X') == 0x80000000

    def test_read_drv_status_unsupported_returns_none(self):
        motor = _make_motor_with_responses({
            'DRVSTAT_X': "ERROR: command 'DRVSTAT_X' not found:",
        })
        assert motor.read_drv_status('X') is None

    def test_read_drv_status_invalid_axis_raises(self):
        motor = _make_motor_with_responses({})
        with pytest.raises(ValueError):
            motor.read_drv_status('Q')

    def test_read_fanspeed_unsupported_returns_none(self):
        motor = _make_motor_with_responses({
            'FANSPEED': "ERROR: command 'FANSPEED' not found:",
        })
        assert motor.read_fanspeed() is None

    def test_read_fanspeed_parses_int(self):
        motor = _make_motor_with_responses({'FANSPEED': '1234'})
        assert motor.read_fanspeed() == 1234

    def test_set_fan_duty_unsupported_returns_false(self):
        motor = _make_motor_with_responses({
            'FAN:50': "ERROR: command 'FAN' not found:",
        })
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

    def test_read_motor_voltages_no_driver_returns_none(self):
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        scope = MagicMock(spec=[])  # no _motion_driver attribute
        api = DiagnosticsAPI(scope)
        assert api.read_motor_voltages() is None

    def test_set_motor_fan_duty_no_driver_returns_false(self):
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        scope = MagicMock(spec=[])
        api = DiagnosticsAPI(scope)
        assert api.set_motor_fan_duty(50) is False

    def test_read_motor_voltages_forwards_to_driver(self):
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        scope = MagicMock()
        scope._motion_driver.read_voltages.return_value = {'5V': 5.0}
        api = DiagnosticsAPI(scope)
        assert api.read_motor_voltages() == {'5V': 5.0}


# ---------------------------------------------------------------------------
# TSR check_voltage_tolerance -- tri-state passed (True / False / None).
# ---------------------------------------------------------------------------

class TestVoltageToleranceTriState:
    """Voltage check returns None for INCONCLUSIVE, not True for PASS."""

    def _make_diag(self, motor_connected, voltage_dict):
        from modules.tech_support_report import FirmwareDiagnostics
        scope = MagicMock()
        scope.motor_connected = motor_connected
        scope.diagnostics.read_motor_voltages.return_value = voltage_dict
        # _motor_ok() also probes _motion_driver as fallback.
        scope._motion_driver.found = motor_connected
        return FirmwareDiagnostics(scope=scope)

    def test_unsupported_firmware_yields_passed_none(self):
        diag = self._make_diag(motor_connected=True, voltage_dict=None)
        result = diag.check_voltage_tolerance()
        assert result['passed'] is None
        assert result['supported'] is False
        assert 'firmware' in result['message'].lower()

    def test_all_rails_unparseable_yields_passed_none(self):
        diag = self._make_diag(
            motor_connected=True,
            voltage_dict={'24V': None, '5V': None, '3.3V': None, '1.2V': None},
        )
        result = diag.check_voltage_tolerance()
        assert result['passed'] is None
        assert result['supported'] is True
        # Every rail UNKNOWN -> INCONCLUSIVE, not PASS.
        for rail_data in result['rails'].values():
            assert rail_data['status'] == 'UNKNOWN'

    def test_within_tolerance_yields_passed_true(self):
        diag = self._make_diag(
            motor_connected=True,
            voltage_dict={'24V': None, '5V': 5.02, '3.3V': 3.31, '1.2V': 1.20},
        )
        result = diag.check_voltage_tolerance()
        assert result['passed'] is True

    def test_out_of_tolerance_yields_passed_false(self):
        diag = self._make_diag(
            motor_connected=True,
            voltage_dict={'24V': None, '5V': 6.0, '3.3V': 3.31, '1.2V': 1.20},
        )
        result = diag.check_voltage_tolerance()
        assert result['passed'] is False

    def test_motor_disconnected_does_not_say_passed_true(self):
        diag = self._make_diag(motor_connected=False, voltage_dict=None)
        result = diag.check_voltage_tolerance()
        assert result['passed'] is None
        assert result['supported'] is False


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
        # not `scope.led`. _target_str must follow the rename so
        # _cmd(self.led_board, ...) routes correctly.
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "getattr(self._scope, 'illumination', None)" in src
        # The legacy attribute must not survive in _target_str.
        ts_block = src.split('def _target_str')[1].split('def ')[0]
        assert "'led'" in ts_block  # the return string
        assert "getattr(self._scope, 'led'" not in ts_block


# ---------------------------------------------------------------------------
# TSR call-site sanity: no raw VOLTAGE / DRVSTAT / FANSPEED command sends.
# ---------------------------------------------------------------------------

class TestTsrUsesDriverMethods:
    """After the cluster fix, TSR's diagnostic primitives go through the
    DiagnosticsAPI sub-API, not raw `_cmd(motor_board, 'VOLTAGE')` style
    sends. The raw paths violated Rule 22 (use production code paths)."""

    def test_no_raw_voltage_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, 'VOLTAGE')" not in src

    def test_no_raw_drvstat_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, f'DRVSTAT_" not in src

    def test_no_raw_fanspeed_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, 'FANSPEED')" not in src

    def test_no_raw_fan_duty_command_send(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        assert "self._cmd(self.motor_board, 'FAN:" not in src
