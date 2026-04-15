#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import re
import threading
import time
from lvp_logger import logger
from drivers.serialboard import SerialBoard
from drivers.registry import led_registry


@led_registry.register('rp2040', priority=100)
class LEDBoard(SerialBoard):

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
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
        """Turn off all LEDs immediately after connect (thermal safety).

        Uses fire-and-forget write to minimize delay. If the board doesn't
        respond, this is a best-effort attempt — the board may be in a
        state where it can't process commands.
        """
        try:
            self._write_command_fast('LEDS_OFF')
            logger.info('[LED Class ] Safety LEDS_OFF sent on connect')
        except Exception as e:
            logger.warning(f'[LED Class ] Safety LEDS_OFF failed: {e}')

    def _on_disconnect(self):
        """Clear LED state cache on disconnect (called under self._lock)."""
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
        logger.info('[LED Class ] LED state cache cleared on disconnect')

    _COLOR_TO_CH = {
        'Blue': 0, 'Green': 1, 'Red': 2,
        'BF': 3, 'PC': 4, 'DF': 5,
    }

    _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}

    def color2ch(self, color):
        """ Convert color name to numerical channel """
        return self._COLOR_TO_CH.get(color, 3)

    def ch2color(self, channel):
        """ Convert numerical channel to color name """
        return self._CH_TO_COLOR.get(channel, 'BF')

    def available_channels(self):
        return tuple(self._COLOR_TO_CH.values())

    def available_colors(self):
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

    def leds_enable(self):
        command = 'LEDS_ENT'
        response = self.exchange_command(command)
        if response is None:
            logger.warning('[LED Class ] leds_enable() got no response')

    def leds_disable(self):
        command = 'LEDS_ENF'
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
        else:
            logger.warning('[LED Class ] leds_disable() got no response')

    def get_status(self):
        # NOTE: LED firmware does not implement a STATUS command.
        # This always returns "Command not recognized". Do not use.
        # TODO: Add STATUS handler to LED firmware in 4.1, or remove this method.
        logger.warning('[LED Class ] get_status() called but LED firmware has no STATUS command')
        return None

    def wait_until_on(self, timeout: float = 5.0):
        # NOTE: Relies on get_status() which is not implemented in LED firmware.
        # This always times out. Do not use.
        # TODO: Implement in 4.1 with v3.1 protocol, or remove.
        logger.warning('[LED Class ] wait_until_on() called but STATUS command not implemented in firmware')
        return

    def get_led_ma(self, color):
        with self._state_lock:
            return self.led_ma.get(color, -1)

    def is_led_on(self, color) -> bool:
        with self._state_lock:
            return self.led_ma.get(color, -1) > 0

    def get_led_state(self, color) -> dict:
        with self._state_lock:
            mA = self.led_ma.get(color, -1)
            enabled = mA > 0
        return {
            'enabled': enabled,
            'illumination': mA,
        }

    def get_led_states(self) -> dict:
        with self._state_lock:
            snapshot = {color: {'enabled': mA > 0, 'illumination': mA}
                        for color, mA in self.led_ma.items()}
        return snapshot

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
            raise ValueError(f"LED channel {channel} out of range [0-{self._MAX_CHANNEL}]")
        if not (0 <= int(mA) <= self._MAX_MA):
            raise ValueError(f"LED current {mA} mA out of safe range [0-{self._MAX_MA}]")
        color = self.ch2color(channel=channel)
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        return color, command

    def _update_state_cache(self, color: str, mA):
        """Update the cached LED state under lock."""
        with self._state_lock:
            self.led_ma[color] = mA

    def led_on(self, channel, mA, block=False, timeout: float = 5.0):
        """
        Turn on LED at channel number at mA power
        If block=True, verify correct callback before returning (with timeout)
        """
        color, command = self._validate_and_build_led_cmd(channel, mA)
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, mA)
        else:
            logger.warning(f'[LED Class ] led_on(ch={channel}, mA={mA}) got no response')

        def check_each_substr(substrings, result):
            for sub_str in substrings:
                if sub_str not in result:
                    return False
            return True

        if block:
            deadline = time.monotonic() + timeout
            while response is None or (command not in response and not check_each_substr(['LED', str(int(channel)), str(int(mA))], response)):
                if time.monotonic() > deadline:
                    logger.warning(f'[LED Class ] led_on(ch={channel}, mA={mA}, block=True) timed out after {timeout}s')
                    break
                time.sleep(0.01)  # Prevent busy-wait CPU burn
                response = self.exchange_command(command)
                if response is not None:
                    self._update_state_cache(color, mA)

    def led_off(self, channel):
        """ Turn off LED at channel number """
        color = self.ch2color(channel=channel)

        command = 'LED' + str(int(channel)) + '_OFF'
        response = self.exchange_command(command)

        if response is not None:
            self._update_state_cache(color, -1)
        else:
            logger.warning(f'[LED Class ] led_off(ch={channel}) got no response')

    def led_on_fast(self, channel, mA):
        """Fast write-only version of led_on for time-critical toggling."""
        color, command = self._validate_and_build_led_cmd(channel, mA)
        self._update_state_cache(color, mA)
        self._write_command_fast(command)

    def led_off_fast(self, channel):
        """Fast write-only version of led_off for time-critical toggling."""
        color = self.ch2color(channel=channel)
        self._update_state_cache(color, -1)
        command = 'LED' + str(int(channel)) + '_OFF'
        self._write_command_fast(command)

    def leds_off(self):
        """ Turn off all LEDs """
        command = 'LEDS_OFF'
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                for color in self.led_ma:
                    self.led_ma[color] = -1
        else:
            logger.warning('[LED Class ] leds_off() got no response')

    def leds_off_fast(self):
        """Fast write-only version to turn off all LEDs."""
        with self._state_lock:
            for color in self.led_ma:
                self.led_ma[color] = -1
        command = 'LEDS_OFF'
        self._write_command_fast(command)

    # ------------------------------------------------------------------
    # Engineering mode and diagnostics
    # ------------------------------------------------------------------
    def enter_engineering_mode(self, timeout=5.0):
        """Enter engineering mode (FACTORY command with Y/N confirmation).

        Sends FACTORY, waits for Y/N prompt, sends Y, drains help text.
        Returns True on success, False if prompt not seen or timeout.
        """
        resp = self.exchange_multiline(
            'FACTORY', timeout=timeout,
            end_markers=['Y/N', 'y/n', 'FACTORY'])
        if resp is None:
            logger.warning('[LED Class ] enter_engineering_mode(): no response')
            return False
        if 'Y/N' not in resp.upper():
            logger.warning(f'[LED Class ] enter_engineering_mode(): no Y/N prompt in: {resp!r}')
            return False
        # Confirm with Y
        confirm_resp = self.exchange_multiline(
            'Y', timeout=timeout,
            end_markers=['FACTORY', 'Engineering', 'RAW', 'ADC'])
        # Drain any remaining help text
        time.sleep(0.5)
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Entered engineering mode')
        return True

    def exit_engineering_mode(self):
        """Exit engineering mode back to safe mode (Q command).

        Returns response string.
        """
        resp = self.exchange_command('Q', timeout=3)
        time.sleep(0.3)
        # Drain any remaining output
        with self._lock:
            if self.driver is not None:
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
        logger.info('[LED Class ] Exited engineering mode')
        return resp

    def selftest(self, timeout=180):
        """Run LED SELFTEST and return parsed results.

        Sends SELFTEST, collects multiline response (one line per channel
        with settle delays between), returns list of result line strings.
        The response ends with a 'Complete' marker.
        """
        resp = self.exchange_multiline(
            'SELFTEST', timeout=timeout,
            end_markers=['Complete', 'COMPLETE', 'DONE', 'ERROR'])
        if resp is None:
            logger.warning('[LED Class ] selftest(): no response')
            return []
        lines = [line.strip() for line in resp.split('\n') if line.strip()]
        logger.info(f'[LED Class ] selftest(): {len(lines)} lines')
        return lines

    def get_info(self):
        """Send INFO and return parsed dict.

        Returns dict with keys like 'version', 'date', 'cal_status',
        and 'raw' (the full response text). Returns empty dict on failure.
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

    def read_led_current(self, channel):
        """Read measured LED current (mA) from ADC feedback. Requires v2.0+ firmware in engineering mode.
        Returns measured current in mA, or None on error/unsupported."""
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
