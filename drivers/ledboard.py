#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import math
import re
import time

import serial
from lvp_logger import logger
from drivers.exceptions import (
    BoardNoReplyError,
    BoardPortFailedError,
    BoardReplyUnreadableError,
    BoardReportedError,
    LedFirmwareWedgedError,
)
from drivers.serialboard import SerialBoard
from drivers.serial_backend import PYSERIAL, SerialBackend
from drivers.registry import led_registry

# The firmware's CH_MAX: the absolute per-channel current limit the
# EL-0940 LED command accepts. One home in LVP; the simulated board
# imports it because it emulates this firmware.
FIRMWARE_LED_CH_MAX_MA = 1000


def firmware_commanded_ma(mA: float) -> float:
    """The current the LED firmware is commanded for a request, in mA.

    The field firmware parses ``LEDn_<mA>`` as an integer, so a request goes
    to the nearest whole mA, a half going up. A request above zero but under
    1 mA goes to 1 mA: rounding it to 0 would leave a channel dark that was
    asked to light. 0 stays 0. The simulated board answers with this same
    function, because it emulates this firmware.
    """
    if mA <= 0:
        return 0.0
    return float(max(1, math.floor(float(mA) + 0.5)))


# Same dedicated serial logger SerialBoard.exchange_command() writes to, so
# the bespoke STIM capability probe (which scans multiple lines and cannot
# use exchange_command) still lands its send + reply in serial.log instead
# of going dark.
_serial_log = logging.getLogger('LVP.serial')


@led_registry.register('rp2040', priority=100)
class LEDBoard(SerialBoard):
    # ----------------------------------------------------------
    # Initialize connection through microcontroller
    # ----------------------------------------------------------
    def __init__(self, backend: SerialBackend = PYSERIAL, **kwargs):
        # The slowest shipped LED firmware (2024-06-05) answers a 1000 mA
        # write in about 150 ms; the motor board's 0.1 s would give up first.
        super().__init__(vid=0x0424, pid=0x704C, label='[LED Class ]', timeout=0.3, backend=backend)

        # Set by _safety_leds_off() at connect time. None when the board
        # confirmed LEDS_OFF; "ExceptionType: message" when it did not
        # answer or the exchange raised. API layer reads this on
        # construction to fire a notification.
        self.last_safety_off_error: str | None = None

        # A board the port search did not find, or whose connect failed, is
        # sent nothing: the registry judges it and bring-up reports it once.
        if not self.found:
            return
        self.connect()
        if not self.is_connected():
            return

        # Safety: immediately turn off all LEDs after connecting.
        # Old crashed LED firmware (pre-v3.0.4) can leave all LEDs stuck on
        # at full current (~500mA x 6 channels = 3A), causing thermal damage
        # to the board (measured 62degC). New v3.0.4+ firmware initializes LEDs
        # off on boot, but this guard protects against old firmware and
        # interrupted previous sessions.
        self._safety_leds_off()

    def _safety_leds_off(self):
        """Turn off all LEDs immediately after connect.

        Guards against pre-v3.0.4 LED firmware that could leave channels
        stuck on at full current after a crash / interrupted session,
        risking thermal damage (62 degC measured at 3 A continuous) and
        sample photobleaching. The reply is read like any other command's:
        a reply left unread is taken by the next command as its own. If
        the board doesn't answer, the failure is recorded in
        self.last_safety_off_error so the API layer can fire a
        user-visible notification.
        """
        try:
            self.leds_off()
            logger.info('[LED Class ] Safety LEDS_OFF confirmed on connect')
            self.last_safety_off_error = None
        except Exception as e:
            logger.error(f'[LED Class ] Safety LEDS_OFF failed: {e}')
            self.last_safety_off_error = f'{type(e).__name__}: {e}'

    def available_channels(self) -> tuple[int, ...]:
        """The LED addresses this board drives, 0 to ``_MAX_CHANNEL``.

        Addresses only: which layer an address lights is the unit's layer
        identity's to say, never the board's.
        """
        return tuple(range(self._MAX_CHANNEL + 1))

    # interperet commands
    # ------------------------------------------
    # board status: 'STATUS' case insensitive
    # LED enable:   'LED' channel '_ENT' where channel is numbers 0 through 7, or S (plural/all)
    # LED disable:  'LED' channel '_ENF' where channel is numbers 0 through 7, or S (plural/all)
    # LED on:       'LED' channel '_MA' where channel is numbers 0 through 7, or S (plural/all)
    #                and MA is numerical representation of mA
    # LED off:      'LED' channel '_OFF' where channel is numbers 0 through 7, or S (plural/all)

    # v3.0 STUB: LED command builders for JSON Lines protocol
    # When v3.0 is active, commands will use structured JSON format:
    #   {"cmd": "LED_ON", "ch": 0, "mA": 100}
    #   {"cmd": "LED_OFF", "ch": 0}
    #   {"cmd": "LEDS_OFF"}
    #   {"cmd": "LED_ENABLE"}
    #   {"cmd": "LED_DISABLE"}
    # Currently all commands use the legacy text format.

    def leds_enable(self) -> None:
        """Enable all LED channels (master enable).

        Sends ``LEDS_ENT``.

        Raises:
            HardwareError: the board did not answer, or its port failed.
        """
        self.exchange_command('LEDS_ENT')

    def leds_disable(self) -> None:
        """Disable all LED channels (master disable).

        Sends ``LEDS_ENF``.

        Raises:
            HardwareError: the board did not answer, or its port failed; a
                stuck-enabled board can still emit light.
        """
        self.exchange_command('LEDS_ENF')

    def supports_firmware_stim(self) -> bool:
        """Probe firmware for STIM command support. Result cached after first call.

        Host-side pulse scheduling is unreliable below ~20 ms pulse width
        because the USB-UART bridge batches back-to-back fast-path writes:
        host-scheduled 50 ms pulses can collapse to ~3 ms physical LED on-
        time. Firmware STIM (LED firmware v3.0.8+) runs the pulse train
        inside the LED firmware with sub-microsecond pulse-edge accuracy
        via ticks_us busy-wait, eliminating the bridge-batching problem.

        Probe sends `STIM 0 0 1 2 1` (intentionally invalid mA=0). v3.0.8+
        replies with `STIM: mA must be > 0` (parser recognized). Pre-v3.0.8
        firmware echoes the command and returns `Command not recognized`.

        Returns:
            bool: True if firmware understands the STIM command (v3.0.8+),
            False if it answered that it does not (pre-v3.0.8). Only an
            answer the board gave is cached on the instance, so subsequent
            calls return without re-probing the bus.

        Raises:
            BoardNoReplyError: no line arrived before the probe's deadline,
                or the port failed after the probe was sent, which closes
                it; nothing is cached, so the next call asks again.
            BoardReplyUnreadableError: the board answered lines that are
                neither answer; nothing is cached.
            BoardPortFailedError: the port is closed, or failed before the
                probe was sent, which closes it.
        """
        if hasattr(self, '_supports_stim_cached'):
            return self._supports_stim_cached
        _probe_cmd = 'STIM 0 0 1 2 1'
        with self._lock:
            if self.driver is None:
                raise BoardPortFailedError(f'{_probe_cmd}: the port is closed')
            saved_timeout = self.driver.timeout
            self.driver.timeout = 0.3
            sent = False
            try:
                self.driver.reset_input_buffer()
                _t0 = time.monotonic()
                self.driver.write(b'STIM 0 0 1 2 1\n')
                sent = True
                got_stim = None
                resp = ''
                heard = []
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    line = self.driver.readline()
                    if not line:
                        continue
                    s = line.decode('utf-8', 'ignore').strip()
                    if 'Command not recognized' in s:
                        got_stim = False
                        resp = s
                        break
                    if s.startswith('STIM:') or s.startswith('STIM_DIAG:'):
                        got_stim = True
                        resp = s
                        break
                    if s:
                        heard.append(s)
                # Record the probe send + reply in serial.log (this path scans
                # multiple lines and so does not go through exchange_command,
                # which logs its own traffic).
                _shown = resp or (f'(no answer in {heard!r})' if heard else '(no reply)')
                _serial_log.info(
                    f'{self._label} {_probe_cmd} (firmware-stim probe) -> '
                    f'{_shown!r} ({(time.monotonic() - _t0) * 1000.0:.1f}ms)'
                )
                # Drain residual bytes so subsequent commands see a clean buffer
                time.sleep(0.2)
                if self.driver.in_waiting:
                    self.driver.read(self.driver.in_waiting)
            except (serial.SerialException, OSError) as e:
                self._close_driver()
                if sent:
                    raise BoardNoReplyError(
                        f'{_probe_cmd}: the port failed after the probe was sent: {e}'
                    ) from e
                raise BoardPortFailedError(f'{_probe_cmd}: the port failed: {e}') from e
            finally:
                if self.driver is not None:
                    self.driver.timeout = saved_timeout
        if got_stim is None:
            if heard:
                raise BoardReplyUnreadableError(
                    f'{_probe_cmd}: the board answered neither STIM nor not-recognized: {heard!r}'
                )
            raise BoardNoReplyError(f'{_probe_cmd}: no reply from the board')
        self._supports_stim_cached = got_stim
        logger.info(f'{self._label} firmware STIM support: {got_stim}')
        return got_stim

    def get_status(self) -> None:
        """Stub -- LED firmware does not implement a STATUS command.

        Returns:
            None: Always. Logs a warning to flag the call site.
        """
        # NOTE: LED firmware does not implement a STATUS command.
        # This always returns "Command not recognized". Do not use.
        logger.warning('[LED Class ] get_status() called but LED firmware has no STATUS command')
        return None

    def wait_until_on(self, timeout_s: float = 5.0) -> bool:
        """Stub -- depends on STATUS command which the firmware lacks.

        Args:
            timeout_s: Maximum wait time in seconds (currently unused).

        Returns:
            bool: True if LED confirmed on, False on timeout / unsupported.
            Currently always False until v3.1 firmware ships STATUS.
        """
        # NOTE: Relies on get_status() which is not implemented in LED firmware.
        # This always returns False. Do not use.
        logger.warning(
            '[LED Class ] wait_until_on() called but STATUS command not implemented in firmware'
        )
        return False

    # The driver holds no LED state: which address is lit is
    # IlluminationAPI's store.

    # Safety limits -- defense-in-depth validation at driver level.
    # The API layer (lumascope_api.py) also validates, but the driver
    # must enforce independently in case of direct calls.
    _MAX_CHANNEL = 7
    _MAX_MA = FIRMWARE_LED_CH_MAX_MA

    def max_ma(self) -> int:
        return self._MAX_MA

    def commanded_ma(self, mA: float) -> float:
        return firmware_commanded_ma(mA)

    def _validate_and_build_led_cmd(self, channel, mA) -> str:
        """Validate channel/mA and return the command string.

        Shared by led_on() and led_on_fast() to eliminate duplicate validation.
        """
        if not (0 <= int(channel) <= self._MAX_CHANNEL):
            raise ValueError(f'LED channel {channel} out of range [0-{self._MAX_CHANNEL}]')
        if not (0 <= int(mA) <= self._MAX_MA):
            raise ValueError(f'LED current {mA} mA out of safe range [0-{self._MAX_MA}]')
        return 'LED' + str(int(channel)) + '_' + str(int(mA))

    def led_on(self, channel: int, mA: int, block: bool = False, timeout_s: float = 5.0) -> None:
        """Turn on the LED on a channel at a given current.

        Args:
            channel: LED address (0-7).
            mA: Drive current in milliamps (0-1000).
            block: When True, poll until the firmware echoes the command
                or until ``timeout_s`` elapses.
            timeout_s: Block timeout in seconds.

        Raises:
            ValueError: ``channel`` or ``mA`` is outside the safe range.
            HardwareError: the board did not answer, or its port failed;
                with ``block``, its last ask before the poll gave up.
            BoardReplyUnreadableError: with ``block``, the board answered
                until the poll gave up but never confirmed the command, so
                the LED is not known to be lit.
        """
        command = self._validate_and_build_led_cmd(channel, mA)
        if mA == 0:
            # The field firmware drives ``LEDn_0`` at its DAC offset, not off;
            # only the board's off command darkens the channel.
            self.led_off(channel)
            return
        if not block:
            self.exchange_command(command)
            return

        def check_each_substr(substrings, result):
            return all(sub_str in result for sub_str in substrings)

        # Poll until the firmware echoes the command back, OR the response
        # contains 'LED' + channel + mA as substrings (the firmware response
        # shape: 'LED N set to X mA.'). No reply is NOT an ack: it is seen
        # when the LED firmware is wedged (e.g. left mid-engineering-mode by
        # a diagnostic flow that exited without draining), and in that state
        # the LED is NOT actually energized. The substring check protects
        # callers from silently succeeding while the hardware is dark.
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                response = self.exchange_command(command)
            except BoardNoReplyError:
                # A board that connected silent never echoes a command until
                # a power cycle, so polling it to the deadline only burns the
                # full timeout, which on a multi-step protocol compounds into
                # minutes. Any other dropped reply is asked again.
                if getattr(self, 'firmware_silent', False) or time.monotonic() > deadline:
                    raise
            else:
                if command in response or check_each_substr(
                    ['LED', str(int(channel)), str(int(mA))], response
                ):
                    return
                if time.monotonic() > deadline:
                    raise BoardReplyUnreadableError(
                        f'led_on(ch={channel}, mA={mA}): the board answered {response!r} '
                        f'for {timeout_s}s and never confirmed {command}'
                    )
            time.sleep(0.01)

    def led_off(self, channel: int) -> None:
        """Turn off the LED on a channel.

        Args:
            channel: LED address (0-7).

        Raises:
            HardwareError: the board did not answer, or its port failed.
        """
        command = 'LED' + str(int(channel)) + '_OFF'
        self.exchange_command(command)

    def led_on_fast(self, channel: int, mA: int) -> None:
        """Fast write-only version of led_on for time-critical toggling.

        Dispatches the command without waiting for a response.

        Args:
            channel: LED address (0-7).
            mA: Drive current in milliamps (0-1000).

        Raises:
            ValueError: ``channel`` or ``mA`` is outside the safe range.
            HardwareError: the write timed out, or the port failed.
        """
        self._write_command_fast(self._validate_and_build_led_cmd(channel, mA))

    def led_off_fast(self, channel: int) -> None:
        """Fast write-only version of led_off for time-critical toggling.

        Args:
            channel: LED address (0-7).

        Raises:
            HardwareError: the write timed out, or the port failed.
        """
        command = 'LED' + str(int(channel)) + '_OFF'
        self._write_command_fast(command)

    def leds_off(self) -> None:
        """Turn off every LED channel.

        Raises:
            HardwareError: the board did not answer, or its port failed.
        """
        self.exchange_command('LEDS_OFF')

    def leds_off_fast(self) -> None:
        """Fast write-only version of leds_off.

        Dispatches LEDS_OFF without waiting for a response.

        Raises:
            HardwareError: the write timed out, or the port failed.
        """
        command = 'LEDS_OFF'
        self._write_command_fast(command)

    # ------------------------------------------------------------------
    # Engineering mode and diagnostics
    # ------------------------------------------------------------------
    def enter_engineering_mode(self, timeout: float = 5.0) -> bool:
        """Enter engineering mode (FACTORY command with Y/N confirmation).

        Sends FACTORY, waits for Y/N prompt, sends Y, drains help text.

        Returns:
            bool: True on success.

        Raises:
            HardwareError: No response from the LED board (timeout or
                disconnect), the firmware did not present a Y/N
                prompt (likely too old to support engineering mode),
                or it did not print its engineering-mode banner after Y.
        """
        resp = self.exchange_multiline(
            'FACTORY', timeout=timeout, end_markers=['Y/N', 'y/n', 'FACTORY']
        )
        if 'Y/N' not in resp.upper():
            raise BoardReportedError(
                f'enter_engineering_mode(): no Y/N prompt seen -- '
                f'firmware may be too old to support engineering mode. '
                f'Response: {resp!r}'
            )
        # The firmware asks with input(), which on MicroPython 1.19 ends a
        # line on a carriage return only: a Y ended by a newline leaves
        # factory() waiting, and every command after it goes unanswered.
        # A bare CR, not CRLF: once input() returns, a trailing newline
        # would reach the main loop as an empty command.
        # A caller told entry failed has no reason to call exit, so the
        # board is brought back here: out of factory() if it is still
        # waiting there, back to safe mode if it declined.
        try:
            confirm = self.exchange_multiline(
                'Y',
                timeout=timeout,
                end_markers=['FACTORY', 'Engineering', 'RAW', 'ADC'],
                line_end=b'\r',
            )
        except BoardNoReplyError:
            self.exit_engineering_mode()
            raise
        if 'ENGINEERING MODE' not in confirm.upper():
            self.exit_engineering_mode()
            raise BoardReportedError(
                f'enter_engineering_mode(): the LED board did not enter engineering '
                f'mode after Y. Response: {confirm!r}'
            )
        # Drain any remaining help text
        time.sleep(0.5)
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Entered engineering mode')
        return True

    def exit_engineering_mode(self) -> bool:
        """Exit engineering mode back to safe mode (Q command).

        The EL-0925 Gen3 firmware (2024-06-05ESWEA) has a `factory()`
        function that does not reliably exit on Q -- if the eng-mode
        body (e.g. LEDREADS) timed out before we get to send Q, the
        firmware input loop is left waiting for an eng-mode response
        and standard commands return ''. In that state, only a Ctrl-D
        soft reset rescues the board. This method probes for the wedge
        via a post-Q INFO and, if wedged, drives the Ctrl-C/B/D
        recovery inline.

        Returns:
            bool: True once INFO confirms the board is back in safe
                mode, with or without the recovery. The Q reply itself
                is not the answer: a wedged board can echo Q and stay in
                factory(), and a board that left it may not answer Q.

        Raises:
            HardwareError: Q failed AND the Ctrl-D soft reset did not
                bring the firmware back. Power-cycle is needed.
        """
        try:
            self.exchange_command('Q', timeout=3)
        except BoardNoReplyError:
            # A board that left factory() may not answer Q: INFO decides.
            logger.info('[LED Class ] exit_engineering_mode: no reply to Q')
        time.sleep(0.3)
        # Drain any remaining output from Q (firmware may print help).
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)

        # Verify firmware actually returned from factory(). INFO is the
        # cheap responsiveness probe -- a wedged firmware gives no reply
        # or garbage; a healthy firmware responds with the version
        # banner whose first line begins with 'Version:' (followed by
        # 'EL-0925 Gen3 LED Controller' or similar -- the exact
        # controller string depends on hardware revision, so 'Version:'
        # is the stable marker across revisions).
        try:
            info_resp = self.exchange_command('INFO', timeout=2)
        except BoardNoReplyError:
            info_resp = None
        if info_resp and 'Version' in info_resp:
            logger.info('[LED Class ] Exited engineering mode')
            return True

        # Wedged inside factory(). Run the Ctrl-C/B/D soft-reset
        # recovery sequence inline. Same shape as SerialBoard
        # _reset_firmware step 4; this narrower version skips the
        # boot-state detection that doesn't apply mid-session.
        logger.warning(
            f'[LED Class ] exit_engineering_mode: INFO returned '
            f'{info_resp!r}; firmware appears wedged in factory() -- '
            f'attempting Ctrl-D recovery'
        )
        self._safe_write(b'\x03', context='eng-mode recovery Ctrl-C #1')
        time.sleep(0.2)
        self._safe_write(b'\x03', context='eng-mode recovery Ctrl-C #2')
        time.sleep(0.2)
        self._safe_write(b'\x02', context='eng-mode recovery Ctrl-B raw-REPL exit')
        time.sleep(0.2)
        self._safe_write(b'\x04', context='eng-mode recovery Ctrl-D soft reset')
        time.sleep(5.0)  # firmware boot
        # Drain boot output
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)

        # Re-verify after recovery
        try:
            info_resp2 = self.exchange_command('INFO', timeout=3)
        except BoardNoReplyError:
            info_resp2 = None
        if info_resp2 and 'Version' in info_resp2:
            logger.info('[LED Class ] Exited engineering mode (after Ctrl-D recovery)')
            return True

        logger.error(
            f'[LED Class ] exit_engineering_mode: INFO still returns '
            f'{info_resp2!r} after Ctrl-D recovery; LED firmware '
            f'unrecoverable from eng-mode wedge'
        )
        raise LedFirmwareWedgedError(
            'LED firmware wedged in engineering mode and did not '
            'recover after a Ctrl-D soft reset. Power-cycle the unit.'
        )

    def selftest(self, timeout: float = 180) -> list:
        """Run LED SELFTEST and return parsed results.

        Sends SELFTEST, collects the multiline response (one line per
        channel with settle delays between), and returns the result lines.
        The response ends with a 'Complete' marker.

        Args:
            timeout: Maximum wait time in seconds (the firmware can take
                a while since it sweeps every channel).

        Returns:
            list: One stripped line per row of the firmware response.

        Raises:
            HardwareError: the board did not answer, or its port failed.
        """
        resp = self.exchange_multiline(
            'SELFTEST', timeout=timeout, end_markers=['Complete', 'COMPLETE', 'DONE', 'ERROR']
        )
        lines = [line.strip() for line in resp.split('\n') if line.strip()]
        logger.info(f'[LED Class ] selftest(): {len(lines)} lines')
        return lines

    def get_info(self) -> dict:
        """Send INFO and return parsed dict.

        No caller on the 4.0 line; the connect-latency bench harness on
        the 4.1 development line registers this as a bench callable. Not
        dead code -- a caller search in this branch alone is misleading.

        Returns:
            dict: Parsed fields including ``version``, ``date``,
                ``cal_status``, and ``raw`` (full response text).

        Raises:
            HardwareError: the board did not answer, or its port failed.
        """
        resp = self.exchange_command('INFO', response_numlines=6, timeout=2)
        if isinstance(resp, list):
            raw = '\n'.join(resp)
        else:
            raw = resp
        result = {'raw': raw}
        # Parse version
        import re as _re

        ver_match = _re.search(r'v(\d+\.\d+(?:\.\d+)?)', raw)
        if ver_match:
            result['version'] = ver_match.group(1)
        date_match = _re.search(r'(\d{4}-\d{2}-\d{2})', raw)
        if date_match:
            result['date'] = date_match.group(1)
        if 'Cal:' in raw or 'Calibrated' in raw:
            result['cal_status'] = 'calibrated' if 'Calibrated' in raw else 'default'
        return result

    def read_led_current(self, channel: int) -> float | None:
        """Read measured LED current (mA) from ADC feedback.

        Requires v2.0+ firmware in engineering mode.

        Args:
            channel: LED address (0-7).

        Returns:
            float | None: Measured current in mA, or None when the firmware
                is too old to sense current (not v2+).

        Raises:
            BoardNoReplyError: the board did not answer.
            BoardPortFailedError: the port is closed, or failed before the
                read was sent.
            BoardReplyUnreadableError: the board answered with no current
                in the reply.
        """
        if not self.is_v2:
            return None
        command = f'LEDREAD{int(channel)}'
        # Firmware sends: echo (handled by exchange_command), I_SENS line, LED_K line
        lines = self.exchange_command(command, response_numlines=3)
        # Parse I_SENS line: "LED0 I_SENS  (AIN14): 1.2800V  ->   200.1 mA"
        for line in lines:
            if 'I_SENS' in line and 'mA' in line:
                m = re.search(r'(\d+(?:\.\d+)?)\s*mA', line)
                if m:
                    return float(m.group(1))
        raise BoardReplyUnreadableError(f'{command}: no current in the reply {lines!r}')
