# Copyright Etaluma, Inc.
"""Regression tests for LED ack polling shape + TSR filename token.

Two unrelated bug clusters land here because both surfaced from the
same bench session:

1. ``LEDBoard.led_on(block=True)`` polling shape. An ack is only
   trusted when the response echoes the command back OR contains
   ``'LED'`` + channel + mA as substrings. An empty-string response
   is NOT an ack -- empty responses are observed when the LED
   firmware is wedged (e.g. left mid-engineering-mode by a
   diagnostic flow that exited without draining), and the LED is
   NOT actually energized in that state. The substring check
   protects callers from silently succeeding while the hardware is
   dark.

2. TSR bundler emits ``SN<sn>-TSR-<timestamp>.zip`` with the TSR
   token so it's visually distinct from the ``SNlogs-...``
   user-dump bundle.
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLedOnBlockAckShape:
    """The block=True polling loop must trust only command-echo or
    substring-match acks. Empty responses must keep polling."""

    def _make_led(self, response_sequence):
        """Build a real LEDBoard with exchange_command monkeypatched."""
        from drivers.ledboard import LEDBoard
        led = LEDBoard.__new__(LEDBoard)  # bypass __init__ (needs serial)
        led._validate_and_build_led_cmd = lambda ch, mA: ('BF', f'LED{ch}_{int(mA)}')
        led._update_state_cache = lambda color, mA: None
        responses = list(response_sequence)
        led.exchange_command = MagicMock(
            side_effect=lambda cmd: responses.pop(0) if responses else None
        )
        return led

    def test_command_echo_response_breaks_polling_loop(self):
        """Firmware shape A: response echoes the command back."""
        led = self._make_led(['LED3_2'])
        led.led_on(channel=3, mA=2, block=True, timeout=5.0)
        assert led.exchange_command.call_count == 1

    def test_substring_match_response_breaks_polling_loop(self):
        """Firmware shape B: response includes 'LED' + channel + mA
        ('LED 3 set to 2 mA.')."""
        led = self._make_led(['LED 3 set to 2 mA.'])
        led.led_on(channel=3, mA=2, block=True, timeout=5.0)
        assert led.exchange_command.call_count == 1

    def test_empty_string_response_keeps_polling(self):
        """Empty-string responses indicate the LED firmware is wedged;
        the loop must keep polling, not accept the empty as ack. This
        gives the user a visible timeout + protocol halt rather than a
        silent dark capture."""
        # 100ms timeout; firmware returns '' forever.
        led = self._make_led([''] * 200)
        import time
        t0 = time.monotonic()
        led.led_on(channel=3, mA=2, block=True, timeout=0.1)
        elapsed = time.monotonic() - t0
        assert 0.08 < elapsed < 0.3, f"elapsed={elapsed:.3f}s out of band"
        # Multiple retries during the 0.1s window -- not bailing on first ''.
        assert led.exchange_command.call_count >= 3

    def test_none_response_keeps_polling(self):
        """No serial bytes back at all: same retry-until-timeout shape
        as the wedged-firmware empty case."""
        led = self._make_led([None] * 200)
        import time
        t0 = time.monotonic()
        led.led_on(channel=3, mA=2, block=True, timeout=0.1)
        elapsed = time.monotonic() - t0
        assert 0.08 < elapsed < 0.3, f"elapsed={elapsed:.3f}s out of band"
        assert led.exchange_command.call_count >= 3


class TestLedDriverSubstringCheckPresent:
    """The substring-match guard on the polling loop is load-bearing.
    A future cleanup that removes it would silently accept empty
    responses (and unresponsive-firmware responses) as acks, leaving
    callers thinking the LED was energized when the hardware is dark."""

    def test_substring_match_helper_present(self):
        src = (REPO_ROOT / 'drivers' / 'ledboard.py').read_text()
        assert 'check_each_substr' in src, (
            "check_each_substr helper must be present in ledboard.py -- "
            "the substring-match ack check guards against wedged-firmware "
            "empty responses being misread as acks."
        )

    def test_command_not_in_response_check_present(self):
        src = (REPO_ROOT / 'drivers' / 'ledboard.py').read_text()
        assert 'command not in response' in src, (
            "Polling loop must include the `command not in response` "
            "rejection clause -- this is what makes the loop wait for "
            "a real ack instead of accepting any non-None string."
        )


class TestTsrLedEngineeringRoutesThroughDriver:
    """TSR LED engineering-mode entry / exit must route through the
    diagnostics sub-API (which delegates to the canonical driver
    methods with proper FACTORY / Y / Q handshake + end-marker
    matching + post-Q drain). An open-coded FACTORY / Y / Q sequence
    in the TSR was leaving the LED firmware wedged when LEDREADS or
    SELFTEST timed out mid-eng-mode."""

    def test_enter_engineering_calls_sub_api(self):
        """_enter_engineering must invoke
        scope.diagnostics.enter_led_engineering_mode, not open-coded
        send_diagnostic_command('led', 'FACTORY', ...)."""
        from modules.tech_support_report import FirmwareDiagnostics
        diag = FirmwareDiagnostics.__new__(FirmwareDiagnostics)
        fake_scope = MagicMock()
        fake_scope.led_connected = True
        fake_scope.diagnostics.enter_led_engineering_mode.return_value = True
        diag._scope = fake_scope

        result = diag._enter_engineering()

        assert result is True
        fake_scope.diagnostics.enter_led_engineering_mode.assert_called_once()
        # Must NOT use the open-coded send_diagnostic_command path with
        # 'FACTORY' or 'Y' as the command.
        for call in fake_scope.send_diagnostic_command.call_args_list:
            args, _ = call
            assert 'FACTORY' not in args and 'Y' not in args, (
                f"Open-coded FACTORY/Y send_diagnostic_command call leaked: {call}"
            )

    def test_exit_engineering_calls_sub_api(self):
        """_exit_engineering must invoke
        scope.diagnostics.exit_led_engineering_mode, not _cmd('led', 'Q')."""
        from modules.tech_support_report import FirmwareDiagnostics
        diag = FirmwareDiagnostics.__new__(FirmwareDiagnostics)
        fake_scope = MagicMock()
        fake_scope.led_connected = True
        diag._scope = fake_scope

        diag._exit_engineering()

        fake_scope.diagnostics.exit_led_engineering_mode.assert_called_once()
        # Must NOT use the open-coded send_diagnostic_command path with
        # 'Q' as the command.
        for call in fake_scope.send_diagnostic_command.call_args_list:
            args, _ = call
            assert 'Q' not in args, (
                f"Open-coded Q send_diagnostic_command call leaked: {call}"
            )

    def test_enter_engineering_returns_false_when_led_absent(self):
        """Guard: no LED board connected -> early return, no driver call."""
        from modules.tech_support_report import FirmwareDiagnostics
        diag = FirmwareDiagnostics.__new__(FirmwareDiagnostics)
        fake_scope = MagicMock()
        fake_scope.led_connected = False
        fake_scope._led_driver = None
        diag._scope = fake_scope

        result = diag._enter_engineering()

        assert result is False
        fake_scope.diagnostics.enter_led_engineering_mode.assert_not_called()


class TestBundleFilenameByReportType:
    """The TSR bundle gets a '-TSR-' token; the logs-only user-dump
    bundle stays plain (no token). They're produced by the same
    `_create_zip` helper -- the report_type parameter selects the
    filename shape. Without per-type gating, the TSR token leaks into
    the logs-only filename (the regression Eric flagged: a logs-only
    dump landed as `SNlogs-TSR-...zip` instead of `SNlogs-...zip`)."""

    def _build_report(self, tmp_path, report_type, sn='12062'):
        """Construct a TechSupportReport stub and invoke _create_zip
        with the given report_type. Returns the resulting zip path."""
        from modules.tech_support_report import TechSupportReport
        report = TechSupportReport.__new__(TechSupportReport)
        # _create_zip just needs `tmp` to point at a populated dir; we
        # provide an empty dir for naming-only assertions.
        return report._create_zip(
            tmp_path, sn, tmp_path, report_type=report_type
        )

    def test_tsr_filename_has_tsr_token(self, tmp_path):
        zip_path = self._build_report(tmp_path, report_type='tsr')
        assert '-TSR-' in zip_path.name, (
            f"Full TSR bundle must contain '-TSR-' token; got {zip_path.name!r}"
        )
        assert zip_path.name.startswith('SN12062-TSR-')

    def test_logs_only_filename_has_no_tsr_token(self, tmp_path):
        zip_path = self._build_report(tmp_path, report_type='logs_only')
        assert '-TSR-' not in zip_path.name, (
            f"Logs-only bundle must NOT contain '-TSR-' token; "
            f"got {zip_path.name!r}. The token leaking into logs-only "
            f"bundles confuses support engineers who sort by filename."
        )
        assert zip_path.name.startswith('SN12062-')

    def test_logs_only_with_sn_fallback_has_no_tsr_token(self, tmp_path):
        """When no SN is available, logs-only uses sn='logs'. Resulting
        filename must be `SNlogs-<ts>.zip`, NOT `SNlogs-TSR-<ts>.zip`."""
        zip_path = self._build_report(tmp_path, report_type='logs_only', sn='logs')
        assert '-TSR-' not in zip_path.name, (
            f"SNlogs fallback must NOT contain '-TSR-'; got {zip_path.name!r}"
        )
        assert zip_path.name.startswith('SNlogs-')


class TestLedExitEngineeringRecoversFromWedge:
    """The EL-0925 Gen3 firmware (2024-06-05ESWEA) `factory()` function
    sometimes doesn't exit on Q when the eng-mode body (e.g. LEDREADS)
    has already timed out. The firmware is left stuck inside factory()
    and standard LED commands return ''. exit_engineering_mode must
    detect this via a post-Q INFO probe and run Ctrl-C/B/D soft reset
    to bring the firmware back."""

    def _make_led(self, info_responses, write_log):
        """Build a real LEDBoard stub. info_responses is a list of
        responses returned by exchange_command('INFO', ...). The Q
        call is matched first and returns a sentinel."""
        from drivers.ledboard import LEDBoard
        led = LEDBoard.__new__(LEDBoard)
        led._lock = threading.RLock()
        led.driver = MagicMock()
        led.driver.in_waiting = 0
        led.driver.read = MagicMock(return_value=b'')

        info_iter = iter(info_responses)

        def fake_exchange(cmd, *args, **kwargs):
            if cmd == 'Q':
                return ''
            if cmd == 'INFO':
                try:
                    return next(info_iter)
                except StopIteration:
                    return ''
            return None

        led.exchange_command = MagicMock(side_effect=fake_exchange)
        led._safe_write = MagicMock(
            side_effect=lambda data, context='': write_log.append(data)
        )
        return led

    def test_happy_path_no_recovery_when_info_returns_banner(self):
        """Post-Q INFO returns a healthy banner -> no Ctrl-D recovery
        fires, no extra delay."""
        write_log = []
        led = self._make_led(
            info_responses=['Version: EL-0925 Gen3 LED Controller'
                            'Firmware: 2024-06-05ESWEA Copyright: Etaluma, Inc.'],
            write_log=write_log,
        )
        led.exit_engineering_mode()

        # Only Q + one INFO. No Ctrl-D bytes written.
        assert led._safe_write.call_count == 0
        assert all(b not in write_log for b in (b'\x03', b'\x02', b'\x04'))

    def test_wedged_firmware_triggers_ctrl_d_recovery_and_succeeds(self):
        """Post-Q INFO returns '' (firmware wedged in factory).
        Recovery sequence Ctrl-C/Ctrl-C/Ctrl-B/Ctrl-D fires; second
        INFO returns healthy banner; method returns normally."""
        write_log = []
        led = self._make_led(
            info_responses=[
                '',  # first probe: wedged
                'Version: EL-0925 Gen3 LED Controller Etaluma, Inc.',  # post-recovery
            ],
            write_log=write_log,
        )
        # Patch time.sleep to skip the 5s firmware-boot wait.
        import drivers.ledboard as ledboard_mod
        original_sleep = ledboard_mod.time.sleep
        ledboard_mod.time.sleep = MagicMock()
        try:
            led.exit_engineering_mode()
        finally:
            ledboard_mod.time.sleep = original_sleep

        # Ctrl-C x2, Ctrl-B, Ctrl-D all written in order.
        assert write_log == [b'\x03', b'\x03', b'\x02', b'\x04']

    def test_wedged_firmware_unrecoverable_raises(self):
        """Post-Q INFO returns '' and the post-recovery INFO still
        returns '' -- firmware is genuinely unrecoverable. Method
        raises HardwareError so the caller can surface the failure."""
        from drivers.exceptions import HardwareError
        import pytest as _pytest

        write_log = []
        led = self._make_led(
            info_responses=['', ''],  # both probes wedged
            write_log=write_log,
        )
        import drivers.ledboard as ledboard_mod
        original_sleep = ledboard_mod.time.sleep
        ledboard_mod.time.sleep = MagicMock()
        try:
            with _pytest.raises(HardwareError, match='wedged'):
                led.exit_engineering_mode()
        finally:
            ledboard_mod.time.sleep = original_sleep
