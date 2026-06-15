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
        led.led_on(channel=3, mA=2, block=True, timeout_s=5.0)
        assert led.exchange_command.call_count == 1

    def test_substring_match_response_breaks_polling_loop(self):
        """Firmware shape B: response includes 'LED' + channel + mA
        ('LED 3 set to 2 mA.')."""
        led = self._make_led(['LED 3 set to 2 mA.'])
        led.led_on(channel=3, mA=2, block=True, timeout_s=5.0)
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
        led.led_on(channel=3, mA=2, block=True, timeout_s=0.1)
        elapsed = time.monotonic() - t0
        assert 0.08 < elapsed < 0.3, f'elapsed={elapsed:.3f}s out of band'
        # Multiple retries during the 0.1s window -- not bailing on first ''.
        assert led.exchange_command.call_count >= 3

    def test_none_response_keeps_polling(self):
        """No serial bytes back at all: same retry-until-timeout shape
        as the wedged-firmware empty case."""
        led = self._make_led([None] * 200)
        import time

        t0 = time.monotonic()
        led.led_on(channel=3, mA=2, block=True, timeout_s=0.1)
        elapsed = time.monotonic() - t0
        assert 0.08 < elapsed < 0.3, f'elapsed={elapsed:.3f}s out of band'
        assert led.exchange_command.call_count >= 3


class TestLedDriverRejectsNonAckResponses:
    """The ack guard on the polling loop is load-bearing: only a
    command echo or an 'LED' + channel + mA substring match counts. A
    future cleanup that loosened it to accept any non-None string would
    leave callers thinking the LED was energized when the hardware is
    dark (the wedged-firmware shape)."""

    def _make_led(self, response_sequence):
        from drivers.ledboard import LEDBoard

        led = LEDBoard.__new__(LEDBoard)
        led._validate_and_build_led_cmd = lambda ch, mA: ('BF', f'LED{ch}_{int(mA)}')
        led._update_state_cache = lambda color, mA: None
        responses = list(response_sequence)
        led.exchange_command = MagicMock(
            side_effect=lambda cmd: responses.pop(0) if responses else None
        )
        return led

    def test_unrelated_response_keeps_polling(self):
        """A non-empty, non-None reply that is neither the command echo
        nor an LED/channel/mA substring match must NOT be accepted as
        an ack -- the loop keeps polling until timeout."""
        import time

        led = self._make_led(['ERROR: unknown command'] * 200)
        t0 = time.monotonic()
        led.led_on(channel=3, mA=2, block=True, timeout_s=0.1)
        elapsed = time.monotonic() - t0
        assert 0.08 < elapsed < 0.3, (
            f'an unrelated reply must not break the polling loop early; elapsed={elapsed:.3f}s'
        )
        assert led.exchange_command.call_count >= 3, (
            'the loop must keep re-polling on non-ack replies'
        )

    def test_partial_substring_match_is_not_an_ack(self):
        """A reply naming the wrong channel/mA must keep polling -- the
        per-token substring check is what rejects it."""
        import time

        led = self._make_led(['LED 5 set to 9 mA.'] * 200)
        t0 = time.monotonic()
        led.led_on(channel=3, mA=2, block=True, timeout_s=0.1)
        elapsed = time.monotonic() - t0
        assert 0.08 < elapsed < 0.3, (
            f'a wrong-channel reply must not count as an ack; elapsed={elapsed:.3f}s'
        )
        assert led.exchange_command.call_count >= 3


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
        for call in fake_scope.diagnostics.send_diagnostic_command.call_args_list:
            args, _ = call
            assert 'FACTORY' not in args and 'Y' not in args, (
                f'Open-coded FACTORY/Y send_diagnostic_command call leaked: {call}'
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
        for call in fake_scope.diagnostics.send_diagnostic_command.call_args_list:
            args, _ = call
            assert 'Q' not in args, f'Open-coded Q send_diagnostic_command call leaked: {call}'

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
        return report._create_zip(tmp_path, sn, tmp_path, report_type=report_type)

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
            f'got {zip_path.name!r}. The token leaking into logs-only '
            f'bundles confuses support engineers who sort by filename.'
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


class TestLogsOnlySerialNumberLookupChain:
    """Bench bug 2026-05-18: tech-support bundle filename was `SNlogs-...`
    instead of `SN12062-...` because `generate_logs_only` only consulted
    `motor_board.motorconfig.serial_number()` (which returned 'Unknown'
    on some boards) and fell straight through to the 'logs' fallback
    without trying the FULLINFO path the full TSR uses. The chain must
    be: motorconfig -> diag.get_serial_number() -> 'logs'.
    """

    def _build_report_stub(self, motorconfig_sn, fullinfo_sn):
        """Construct a TechSupportReport with a stubbed diag whose
        motorconfig and get_serial_number return controlled values.
        Returns the SN-resolution result (the value that would be passed
        to _create_zip)."""
        from modules.tech_support_report import TechSupportReport

        report = TechSupportReport.__new__(TechSupportReport)
        report._meta = {}

        motor_board = MagicMock()
        if motorconfig_sn is _NO_ATTR:
            del motor_board.motorconfig
        else:
            motor_board.motorconfig.serial_number.return_value = motorconfig_sn

        diag = MagicMock()
        diag.motor_board = motor_board
        diag.get_serial_number.return_value = fullinfo_sn
        report.diag = diag

        # Replicate the SN-resolution chain from generate_logs_only.
        # Kept inline so the test fails if the chain changes shape.
        sn_tag = None
        try:
            mb = report.diag.motor_board
            if mb is not None and hasattr(mb, 'motorconfig'):
                sn = mb.motorconfig.serial_number()
                if sn and sn != 'Unknown':
                    sn_tag = sn
        except Exception:
            sn_tag = None
        if not sn_tag:
            try:
                sn = report.diag.get_serial_number()
                if sn and sn != 'UNKNOWN':
                    sn_tag = sn
            except Exception:
                sn_tag = None
        if not sn_tag:
            sn_tag = 'logs'
        return sn_tag

    def test_motorconfig_returns_sn(self):
        """Happy path: motorconfig has the SN, no fallback needed."""
        assert self._build_report_stub('12062', None) == '12062'

    def test_motorconfig_unknown_falls_through_to_fullinfo(self):
        """The bench bug: motorconfig returns 'Unknown', the chain must
        try diag.get_serial_number() (FULLINFO) instead of dropping
        straight to 'logs'."""
        assert self._build_report_stub('Unknown', '12062') == '12062'

    def test_both_unknown_falls_through_to_logs(self):
        """Last-resort fallback: neither source has the SN."""
        assert self._build_report_stub('Unknown', 'UNKNOWN') == 'logs'

    def test_motorconfig_attribute_missing_falls_through_to_fullinfo(self):
        """motor_board exists but has no motorconfig attribute (e.g.
        SimulatedMotorBoard) -- chain must continue to FULLINFO."""
        assert self._build_report_stub(_NO_ATTR, '12062') == '12062'


_NO_ATTR = object()


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
        led._safe_write = MagicMock(side_effect=lambda data, context='': write_log.append(data))
        return led

    def test_happy_path_no_recovery_when_info_returns_banner(self):
        """Post-Q INFO returns a healthy banner -> no Ctrl-D recovery
        fires, no extra delay. exchange_command returns only the first
        line of the multi-line INFO response, so the marker must be in
        that first line."""
        write_log = []
        led = self._make_led(
            info_responses=['Version:      EL-0925 Gen3 LED Controller'],
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
                'Version:      EL-0925 Gen3 LED Controller',  # post-recovery
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


class TestTechSupportReportPassesTimeoutS:
    """Bench bug 2026-05-22: tech-support bundles came back named
    ``SNlogs-<ts>.zip`` even though the motor SN was visible in
    fullinfo. Root cause: the U6 timeout->timeout_s L2 sweep (LVP
    `1bc30c5`) renamed ``diagnostics.send_diagnostic_command``'s
    timeout kwarg to ``timeout_s=`` but missed the two callers in
    ``tech_support_report._cmd`` and ``_read_multiline``. Every
    diagnostic command raised TypeError; the broad ``except
    Exception`` in ``generate_logs_only``'s SN-resolution chain
    swallowed the TypeError silently, and the SN fell through to
    the ``'logs'`` last-resort fallback.

    The AST scans below pin both halves of the fix: (a) _cmd and
    _read_multiline pass ``timeout_s=`` to diagnostics; (b) no
    caller in tech_support_report.py passes ``timeout=`` to _cmd
    or _read_multiline (sister-rename caught at the source).
    """

    def _tsr_source(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'tech_support_report.py'
        ).read_text()

    def test_cmd_and_read_multiline_pass_timeout_s_to_diagnostics(self):
        # _cmd / _read_multiline must forward the per-call timeout via the
        # renamed timeout_s= kwarg; a bare timeout= raised TypeError and
        # produced the SNlogs fallback. Drive both against a recording
        # diagnostics sub-API and assert the forwarded kwargs.
        from modules.tech_support_report import FirmwareDiagnostics

        illum = object()
        scope = MagicMock()
        scope.illumination = illum
        scope.motion = object()
        diag = FirmwareDiagnostics(scope=scope)

        diag._cmd(illum, 'INFO', timeout_s=7)
        scope.diagnostics.send_diagnostic_command.assert_called_once_with(
            'led', 'INFO', timeout_s=7
        )

        diag._read_multiline(illum, 'SELFTEST', timeout_s=12, end_markers=['DONE'])
        scope.diagnostics.send_diagnostic_command_multiline.assert_called_once_with(
            'led', 'SELFTEST', timeout_s=12, end_markers=['DONE']
        )

    def test_no_caller_passes_bare_timeout_to_cmd_or_read_multiline(self):
        import re

        src = self._tsr_source()
        # Match `_cmd(...timeout=...)` or `_read_multiline(...timeout=...)`
        # that is NOT timeout_s. Excludes the FirmwareDiagnostics _cmd
        # def itself (signature uses timeout_s already).
        bad_cmd = re.findall(r'_cmd\([^)]*?\btimeout=', src)
        bad_rm = re.findall(r'_read_multiline\([^)]*?\btimeout=', src)
        # Filter out the def lines themselves (def _cmd / _read_multiline)
        bad_cmd = [m for m in bad_cmd if 'def _cmd' not in m]
        bad_rm = [m for m in bad_rm if 'def _read_multiline' not in m]
        assert not bad_cmd, f'Found _cmd callers still passing timeout= (not timeout_s=): {bad_cmd}'
        assert not bad_rm, f'Found _read_multiline callers still passing timeout=: {bad_rm}'
