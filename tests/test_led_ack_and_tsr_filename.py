# Copyright Etaluma, Inc.
"""Regression tests for LED ack polling shape + TSR filename token.

Two unrelated bug clusters land here because both surfaced from the
same bench session:

1. ``LEDBoard.led_on(block=True)`` polling shape. An ack is only
   trusted when the response echoes the command back OR contains
   ``'LED'`` + channel + mA as substrings. No reply is NOT an ack --
   it is observed when the LED firmware is wedged (e.g. left mid-engineering-mode by a
   diagnostic flow that exited without draining), and the LED is
   NOT actually energized in that state. The substring check
   protects callers from silently succeeding while the hardware is
   dark.

2. TSR bundler emits ``SN<sn>-TSR-<timestamp>.zip`` with the TSR
   token so it's visually distinct from the ``SNlogs-...``
   user-dump bundle.
"""

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from drivers.exceptions import BoardNoReplyError, BoardReplyUnreadableError
from tests.wire_boards import led_on_the_wire

REPO_ROOT = Path(__file__).resolve().parent.parent


def _no_boot_wait():
    """The LED board module's own clock with no sleep; every other thread's is real."""
    import drivers.ledboard as ledboard_mod

    return patch.object(
        ledboard_mod, 'time', SimpleNamespace(sleep=MagicMock(), monotonic=time.monotonic)
    )


def _asked(caplog, command):
    return [
        r for r in caplog.records if r.name == 'LVP.serial' and f' {command} -> ' in r.getMessage()
    ]


class TestLedOnBlockAckShape:
    """The block=True polling loop must trust only command-echo or
    substring-match acks; no reply keeps polling.

    The field LED firmware's answer runs on the firmware-backed simulator.
    The bare command echo (shape A) and a board that connected silent are
    answers no emulated firmware gives, so those cases script the exchange."""

    def _make_led(self, response_sequence):
        """Build a real LEDBoard with exchange_command monkeypatched."""
        from drivers.ledboard import LEDBoard

        led = LEDBoard.__new__(LEDBoard)  # bypass __init__ (needs serial)
        # a stand-in by design: the subject is led_on's poll, and the wire's
        # exchange is its interchangeable collaborator; an entry that is
        # an exception is raised, as the wire raises a dropped reply
        led.exchange_command = MagicMock(side_effect=list(response_sequence))
        return led

    def test_command_echo_response_breaks_polling_loop(self):
        """Firmware shape A: response echoes the command back."""
        led = self._make_led(['LED3_2'])
        led.led_on(channel=3, mA=2, block=True, timeout_s=5.0)
        assert led.exchange_command.call_count == 1

    @pytest.mark.firmware_tier
    def test_substring_match_response_breaks_polling_loop(self, caplog):
        """Firmware shape B, the field firmware's: the response includes
        'LED' + channel + mA ('LED 3 set to 2 mA.')."""
        led, _sim = led_on_the_wire()
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            led.led_on(channel=3, mA=2, block=True, timeout_s=5.0)
        [asked] = _asked(caplog, 'LED3_2')
        assert 'LED 3 set to 2 mA.' in asked.getMessage()

    @pytest.mark.firmware_tier
    def test_no_reply_keeps_polling(self, caplog):
        """No serial bytes back at all: the loop keeps asking until its
        timeout, then raises rather than return to a silent dark capture.
        The board's reply is held past the whole poll."""
        import time

        led, sim = led_on_the_wire()
        sim.delay_next_reply(3.0)
        t0 = time.monotonic()
        with caplog.at_level(logging.INFO, logger='LVP.serial'), pytest.raises(BoardNoReplyError):
            led.led_on(channel=3, mA=2, block=True, timeout_s=1.0)
        assert time.monotonic() - t0 >= 1.0
        assert len(_asked(caplog, 'LED3_2')) >= 2, 'the loop gave up after one ask'

    def test_firmware_silent_aborts_polling_immediately(self):
        """A board that connected silent never echoes a command until a power
        cycle, so the block loop must abort at once rather than poll the full
        timeout -- otherwise every confirm-on caller (a protocol step light,
        autofocus entry) stalls the whole timeout, compounding to minutes over a
        run. Distinct from a live-but-transiently-empty firmware (above), which
        keeps polling for a visible timeout."""
        led = self._make_led(
            [BoardNoReplyError('LED3_2: the board sent nothing at connect; power cycle it')] * 200
        )
        led.firmware_silent = True
        import time

        t0 = time.monotonic()
        with pytest.raises(BoardNoReplyError, match='power cycle it'):
            led.led_on(channel=3, mA=2, block=True, timeout_s=5.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f'silent-board block must abort fast, not poll: elapsed={elapsed:.3f}s'
        )
        # Only the first exchange ran; the loop aborted on the silent check
        # before any retry.
        assert led.exchange_command.call_count == 1


class TestLedDriverRejectsNonAckResponses:
    """The ack guard on the polling loop is load-bearing: only a
    command echo or an 'LED' + channel + mA substring match counts. A
    future cleanup that loosened it to accept any non-None string would
    leave callers thinking the LED was energized when the hardware is
    dark (the wedged-firmware shape)."""

    def _make_led(self, response_sequence):
        from drivers.ledboard import LEDBoard

        led = LEDBoard.__new__(LEDBoard)
        # a stand-in by design: the subject is led_on's poll, and the wire's
        # exchange is its interchangeable collaborator; an entry that is
        # an exception is raised, as the wire raises a dropped reply
        led.exchange_command = MagicMock(side_effect=list(response_sequence))
        return led

    def test_unrelated_response_keeps_polling(self):
        """A non-empty, non-None reply that is neither the command echo
        nor an LED/channel/mA substring match must NOT be accepted as
        an ack -- the loop keeps polling until timeout, then raises."""
        import time

        led = self._make_led(['ERROR: unknown command'] * 200)
        t0 = time.monotonic()
        with pytest.raises(BoardReplyUnreadableError, match='never confirmed LED3_2'):
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
        with pytest.raises(BoardReplyUnreadableError, match='never confirmed LED3_2'):
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
        fake_scope.diagnostics.enter_led_engineering_mode.return_value = None
        diag._scope = fake_scope

        result = diag._enter_engineering()

        assert result is None, 'None: the board is in engineering mode'
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

    def test_enter_engineering_answers_the_refusals_words_when_led_absent(self):
        """No LED board connected: the member's own refusal is the answer, in its words."""
        from modules.exceptions import HardwareCommandRefusedError, MissingPart
        from modules.tech_support_report import FirmwareDiagnostics

        diag = FirmwareDiagnostics.__new__(FirmwareDiagnostics)
        fake_scope = MagicMock()
        part = MissingPart.LED_CONTROLLER
        fake_scope.diagnostics.enter_led_engineering_mode.side_effect = HardwareCommandRefusedError(
            part.reason, 'enter_led_engineering_mode', missing=part
        )
        diag._scope = fake_scope

        result = diag._enter_engineering()

        assert result == part.sentence


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


class TestLedExitEngineeringRecoversFromWedge:
    """The EL-0925 Gen3 firmware (2024-06-05ESWEA) `factory()` function
    sometimes doesn't exit on Q when the eng-mode body (e.g. LEDREADS)
    has already timed out. The firmware is left stuck inside factory()
    and standard LED commands return ''. exit_engineering_mode must
    detect this via a post-Q INFO probe and run Ctrl-C/B/D soft reset
    to bring the firmware back.

    The way out without a recovery is
    `test_sim_wire_led.py`'s enter-and-leave test on the field firmware. A
    wedged factory() is scripted here: a lost reply on the emulated board
    takes an echo line and INFO still answers, so the wedge needs the
    firmware itself left waiting inside factory()."""

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
        # The board module's own clock skips the 5s firmware-boot wait.
        with _no_boot_wait():
            led.exit_engineering_mode()

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
        with _no_boot_wait(), _pytest.raises(HardwareError, match='wedged'):
            led.exit_engineering_mode()


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
