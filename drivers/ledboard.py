#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import re
import threading
import time
from lvp_logger import logger
from drivers.exceptions import HardwareError
from drivers.serialboard import SerialBoard
from drivers.registry import led_registry


@led_registry.register('rp2040', priority=100)
class LEDBoard(SerialBoard):
    # ----------------------------------------------------------
    # Initialize connection through microcontroller
    # ----------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(vid=0x0424, pid=0x704C, label='[LED Class ]')

        self._state_lock = threading.Lock()
        self.led_ma = {
            'BF': -1,
            'PC': -1,
            'DF': -1,
            'Red': -1,
            'Blue': -1,
            'Green': -1,
        }

        # Set by _safety_leds_off() at connect time. None when LEDS_OFF
        # send succeeded; "ExceptionType: message" when it failed. API
        # layer reads this on construction to fire a notification.
        self.last_safety_off_error: 'str | None' = None

        # Set by leds_off / led_on / led_off / leds_enable / leds_disable
        # per call. None on success; a short dict with op name + reason
        # when the LED board did not acknowledge the command. The API
        # layer reads this after each call and fires a sample-safety
        # notification with op context. notification_center auto-
        # suppresses during shutdown so stale fields during atexit are
        # harmless. The five runtime methods share one field because
        # the recovery / notification shape is identical; only the op
        # label varies.
        self.last_command_error: 'dict | None' = None

        try:
            self.connect()
        except Exception:
            logger.error('[LED Class ] Failed to connect to LED controller')
            raise

        # Safety: immediately turn off all LEDs after connecting.
        # Old crashed LED firmware (pre-v3.0.4) can leave all LEDs stuck on
        # at full current (~500mA × 6 channels = 3A), causing thermal damage
        # to the board (measured 62°C). New v3.0.4+ firmware initializes LEDs
        # off on boot, but this guard protects against old firmware and
        # interrupted previous sessions.
        self._safety_leds_off()

    def _safety_leds_off(self):
        """Turn off all LEDs immediately after connect.

        Guards against pre-v3.0.4 LED firmware that could leave channels
        stuck on at full current after a crash / interrupted session,
        risking thermal damage (62 degC measured at 3 A continuous) and
        sample photobleaching. Uses fire-and-forget write to minimize
        delay. If the board doesn't respond, this is a best-effort
        attempt; the failure is recorded in self.last_safety_off_error
        so the API layer can fire a Rule 14 notification.
        """
        try:
            self._write_command_fast('LEDS_OFF')
            logger.info('[LED Class ] Safety LEDS_OFF sent on connect')
            self.last_safety_off_error = None
        except Exception as e:
            logger.error(f'[LED Class ] Safety LEDS_OFF failed: {e}')
            self.last_safety_off_error = f'{type(e).__name__}: {e}'

    def _on_disconnect(self):
        """Clear LED state cache on disconnect (called under self._lock)."""
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
        logger.info('[LED Class ] LED state cache cleared on disconnect')

    _COLOR_TO_CH = {
        'Blue': 0,
        'Green': 1,
        'Red': 2,
        'BF': 3,
        'PC': 4,
        'DF': 5,
    }

    _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}

    def color2ch(self, color: str) -> int:
        """Convert color name to numerical channel.

        Args:
            color: Color name (e.g. 'BF', 'Red', 'Blue').

        Returns:
            int: Channel number (0-5). Defaults to 3 (BF) for unknown names.
        """
        return self._COLOR_TO_CH.get(color, 3)

    def ch2color(self, channel: int) -> str:
        """Convert numerical channel to color name.

        Args:
            channel: Channel number (0-5).

        Returns:
            str: Color name. Defaults to 'BF' for unknown channels.
        """
        return self._CH_TO_COLOR.get(channel, 'BF')

    def available_channels(self) -> tuple:
        """Return all known LED channel numbers.

        Returns:
            tuple: Channel numbers (ints) supported by this board.
        """
        return tuple(self._COLOR_TO_CH.values())

    def available_colors(self) -> tuple:
        """Return all known LED color names.

        Returns:
            tuple: Color name strings supported by this board.
        """
        return tuple(self._COLOR_TO_CH.keys())

    # interperet commands
    # ------------------------------------------
    # board status: 'STATUS' case insensitive
    # LED enable:   'LED' channel '_ENT' where channel is numbers 0 through 5, or S (plural/all)
    # LED disable:  'LED' channel '_ENF' where channel is numbers 0 through 5, or S (plural/all)
    # LED on:       'LED' channel '_MA' where channel is numbers 0 through 5, or S (plural/all)
    #                and MA is numerical representation of mA
    # LED off:      'LED' channel '_OFF' where channel is numbers 0 through 5, or S (plural/all)

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

        Sends ``LEDS_ENT``. On no-response sets self.last_command_error
        so the API layer can notify; clears the field on success.
        """
        command = 'LEDS_ENT'
        response = self.exchange_command(command)
        if response is None:
            logger.error('[LED Class ] leds_enable() got no response')
            self.last_command_error = {
                'op': 'leds_enable',
                'reason': 'no response from LED board',
            }
        else:
            self.last_command_error = None

    def leds_disable(self) -> None:
        """Disable all LED channels (master disable) and clear the cache.

        Sends ``LEDS_ENF``. On success, clears the cached per-channel mA
        state so subsequent reads reflect the disabled condition. On no-
        response sets self.last_command_error so the API layer can
        notify the user (sample safety -- a stuck-enabled board can
        still emit light).
        """
        command = 'LEDS_ENF'
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
            self.last_command_error = None
        else:
            logger.error('[LED Class ] leds_disable() got no response')
            self.last_command_error = {
                'op': 'leds_disable',
                'reason': 'no response from LED board',
            }

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
            False otherwise (pre-v3.0.8 or no LED board connected). Result
            is cached on the instance so subsequent calls return without
            re-probing the bus.
        """
        if hasattr(self, '_supports_stim_cached'):
            return self._supports_stim_cached
        with self._lock:
            if self.driver is None:
                return False
            saved_timeout = self.driver.timeout
            self.driver.timeout = 0.3
            try:
                self.driver.reset_input_buffer()
                self.driver.write(b'STIM 0 0 1 2 1\n')
                got_stim = False
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    line = self.driver.readline()
                    if not line:
                        continue
                    s = line.decode('utf-8', 'ignore').strip()
                    if 'Command not recognized' in s:
                        got_stim = False
                        break
                    if s.startswith('STIM:') or s.startswith('STIM_DIAG:'):
                        got_stim = True
                        break
                # Drain residual bytes so subsequent commands see a clean buffer
                time.sleep(0.2)
                if self.driver.in_waiting:
                    self.driver.read(self.driver.in_waiting)
            finally:
                if self.driver is not None:
                    self.driver.timeout = saved_timeout
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
        # TODO: Add STATUS handler to LED firmware in 4.1, or remove this method.
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
        # TODO: Implement in 4.1 with v3.1 protocol, or remove.
        logger.warning(
            '[LED Class ] wait_until_on() called but STATUS command not implemented in firmware'
        )
        return False

    # State-query methods (get_led_ma / is_led_on / get_led_state /
    # get_led_states) retired in Wave 7 Phase 3d.5. LED state is
    # API-primary (single SoT on IlluminationAPI). The `self.led_ma`
    # dict + `_update_state_cache` writes remain as driver-internal
    # state with no external readers; eligible for follow-up dead-code
    # removal.

    # Safety limits — defense-in-depth validation at driver level.
    # The API layer (lumascope_api.py) also validates, but the driver
    # must enforce independently in case of direct calls.
    _MAX_CHANNEL = 5
    _MAX_MA = 1000  # Firmware CH_MAX — absolute hardware limit

    def _validate_and_build_led_cmd(self, channel, mA):
        """Validate channel/mA and return (color, command) string.

        Shared by led_on() and led_on_fast() to eliminate duplicate validation.
        """
        if not (0 <= int(channel) <= self._MAX_CHANNEL):
            raise ValueError(f'LED channel {channel} out of range [0-{self._MAX_CHANNEL}]')
        if not (0 <= int(mA) <= self._MAX_MA):
            raise ValueError(f'LED current {mA} mA out of safe range [0-{self._MAX_MA}]')
        color = self.ch2color(channel=channel)
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        return color, command

    def _update_state_cache(self, color: str, mA):
        """Update the cached LED state under lock."""
        with self._state_lock:
            self.led_ma[color] = mA

    def led_on(self, channel: int, mA: int, block: bool = False, timeout_s: float = 5.0) -> None:
        """Turn on the LED on a channel at a given current.

        Args:
            channel: Channel number (0-5).
            mA: Drive current in milliamps (0-1000).
            block: When True, poll until the firmware echoes the command
                or until ``timeout_s`` elapses.
            timeout_s: Block timeout in seconds.

        Raises:
            ValueError: ``channel`` or ``mA`` is outside the safe range.
        """
        color, command = self._validate_and_build_led_cmd(channel, mA)
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, mA)
            self.last_command_error = None
        else:
            logger.error(f'[LED Class ] led_on(ch={channel}, mA={mA}) got no response')
            self.last_command_error = {
                'op': f'led_on(ch={channel}, mA={mA})',
                'reason': 'no response from LED board',
            }

        def check_each_substr(substrings, result):
            for sub_str in substrings:
                if sub_str not in result:
                    return False
            return True

        if block:
            # Poll until the firmware echoes the command back, OR the
            # response contains 'LED' + channel + mA as substrings (the
            # firmware response shape: 'LED N set to X mA.'). An empty
            # response is NOT treated as ack: empty responses are observed
            # when the LED firmware is wedged (e.g. left mid-engineering-
            # mode by a diagnostic flow that exited without draining), and
            # in that state the LED is NOT actually energized. The
            # substring check protects callers from silently succeeding
            # while the hardware is dark.
            deadline = time.monotonic() + timeout_s
            while response is None or (
                command not in response
                and not check_each_substr(['LED', str(int(channel)), str(int(mA))], response)
            ):
                if time.monotonic() > deadline:
                    logger.warning(
                        f'[LED Class ] led_on(ch={channel}, mA={mA}, block=True) '
                        f'timed out after {timeout_s}s'
                    )
                    break
                time.sleep(0.01)
                response = self.exchange_command(command)
                if response is not None:
                    self._update_state_cache(color, mA)

    def led_off(self, channel: int) -> None:
        """Turn off the LED on a channel.

        Args:
            channel: Channel number (0-5).
        """
        color = self.ch2color(channel=channel)

        command = 'LED' + str(int(channel)) + '_OFF'
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, -1)
            self.last_command_error = None
        else:
            logger.error(f'[LED Class ] led_off(ch={channel}) got no response')
            self.last_command_error = {
                'op': f'led_off(ch={channel})',
                'reason': 'no response from LED board',
            }

    def led_on_fast(self, channel: int, mA: int) -> None:
        """Fast write-only version of led_on for time-critical toggling.

        Updates the cached state and dispatches the command without waiting
        for a response.

        Args:
            channel: Channel number (0-5).
            mA: Drive current in milliamps (0-1000).

        Raises:
            ValueError: ``channel`` or ``mA`` is outside the safe range.
        """
        color, command = self._validate_and_build_led_cmd(channel, mA)
        self._update_state_cache(color, mA)
        self._write_command_fast(command)

    def led_off_fast(self, channel: int) -> None:
        """Fast write-only version of led_off for time-critical toggling.

        Args:
            channel: Channel number (0-5).
        """
        color = self.ch2color(channel=channel)
        self._update_state_cache(color, -1)
        command = 'LED' + str(int(channel)) + '_OFF'
        self._write_command_fast(command)

    def leds_off(self) -> None:
        """Turn off every LED channel.

        On success, clears the cached per-channel mA state and resets
        self.last_command_error to None. On no-response, sets
        self.last_command_error so the API-layer wrap can fire a
        sample-safety notification.
        """
        command = 'LEDS_OFF'
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
            self.last_command_error = None
        else:
            logger.error('[LED Class ] leds_off() got no response')
            self.last_command_error = {
                'op': 'leds_off',
                'reason': 'no response from LED board',
            }

    def leds_off_fast(self) -> None:
        """Fast write-only version of leds_off.

        Clears the cached state and dispatches LEDS_OFF without waiting
        for a response.
        """
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
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
                disconnect), or the firmware did not present a Y/N
                prompt (likely too old to support engineering mode).
        """
        resp = self.exchange_multiline(
            'FACTORY', timeout=timeout, end_markers=['Y/N', 'y/n', 'FACTORY']
        )
        if resp is None:
            raise HardwareError(
                'enter_engineering_mode(): no response from LED board (timeout or disconnect)'
            )
        if 'Y/N' not in resp.upper():
            raise HardwareError(
                f'enter_engineering_mode(): no Y/N prompt seen -- '
                f'firmware may be too old to support engineering mode. '
                f'Response: {resp!r}'
            )
        # Confirm with Y
        confirm_resp = self.exchange_multiline(
            'Y', timeout=timeout, end_markers=['FACTORY', 'Engineering', 'RAW', 'ADC']
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

    def exit_engineering_mode(self) -> str | None:
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
            str | None: Raw Q response, or None if no response was
                received. Returns even when post-Q recovery had to
                fire -- caller can ignore the value.

        Raises:
            HardwareError: Q failed AND the Ctrl-D soft reset did not
                bring the firmware back. Power-cycle is needed.
        """
        resp = self.exchange_command('Q', timeout=3)
        time.sleep(0.3)
        # Drain any remaining output from Q (firmware may print help).
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)

        # Verify firmware actually returned from factory(). INFO is the
        # cheap responsiveness probe -- a wedged firmware returns ''
        # or garbage; a healthy firmware responds with the version
        # banner whose first line begins with 'Version:' (followed by
        # 'EL-0925 Gen3 LED Controller' or similar -- the exact
        # controller string depends on hardware revision, so 'Version:'
        # is the stable marker across revisions).
        info_resp = self.exchange_command('INFO', timeout=2)
        if info_resp and 'Version' in info_resp:
            logger.info('[LED Class ] Exited engineering mode')
            return resp

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
        info_resp2 = self.exchange_command('INFO', timeout=3)
        if info_resp2 and 'Version' in info_resp2:
            logger.info('[LED Class ] Exited engineering mode (after Ctrl-D recovery)')
            return resp

        logger.error(
            f'[LED Class ] exit_engineering_mode: INFO still returns '
            f'{info_resp2!r} after Ctrl-D recovery; LED firmware '
            f'unrecoverable from eng-mode wedge'
        )
        raise HardwareError(
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
                Empty list when no response was received.
        """
        resp = self.exchange_multiline(
            'SELFTEST', timeout=timeout, end_markers=['Complete', 'COMPLETE', 'DONE', 'ERROR']
        )
        if resp is None:
            logger.warning('[LED Class ] selftest(): no response')
            return []
        lines = [line.strip() for line in resp.split('\n') if line.strip()]
        logger.info(f'[LED Class ] selftest(): {len(lines)} lines')
        return lines

    def get_info(self) -> dict:
        """Send INFO and return parsed dict.

        Returns:
            dict: Parsed fields including ``version``, ``date``,
                ``cal_status``, and ``raw`` (full response text). Empty
                dict when no response was received.
        """
        resp = self.exchange_command('INFO', response_numlines=6, timeout=2)
        if resp is None:
            return {}
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
            channel: Channel number (0-5).

        Returns:
            float | None: Measured current in mA, or None on error or
                when the firmware is too old.
        """
        if not self.is_v2:
            return None
        command = f'LEDREAD{int(channel)}'
        try:
            # Firmware sends: echo (handled by exchange_command), I_SENS line, LED_K line
            lines = self.exchange_command(command, response_numlines=3)
            if lines is None:
                return None
            # Parse I_SENS line: "LED0 I_SENS  (AIN14): 1.2800V  ->   200.1 mA"
            for line in lines:
                if 'I_SENS' in line and 'mA' in line:
                    m = re.search(r'([\d.]+)\s*mA', line)
                    if m:
                        return float(m.group(1))
        except Exception as e:
            logger.error(f'[LED Class ] read_led_current({channel}) failed: {e}')
        return None
