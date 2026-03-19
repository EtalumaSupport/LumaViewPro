#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import re
import threading
import time
from lvp_logger import logger
from drivers.serialboard import SerialBoard


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
        self.exchange_command(command)

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
        command = 'STATUS'
        return self.exchange_command(command)

    def wait_until_on(self, timeout: float = 5.0):
        # Waits in loop until ledboard confirms that an LED is on (not turned off)

        deadline = time.monotonic() + timeout
        status = self.get_status()
        while status is None or "STATUS" not in status:
            if time.monotonic() > deadline:
                logger.warning(f"[LED] wait_until_on() timed out after {timeout}s")
                return
            status = self.get_status()

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

    def led_on(self, channel, mA, block=False, timeout: float = 5.0):
        """
        Turn on LED at channel number at mA power
        If block=True, verify correct callback before returning (with timeout)
        """
        color = self.ch2color(channel=channel)

        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                self.led_ma[color] = mA
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
                response = self.exchange_command(command)
                if response is not None:
                    with self._state_lock:
                        self.led_ma[color] = mA

    def led_off(self, channel):
        """ Turn off LED at channel number """
        color = self.ch2color(channel=channel)

        command = 'LED' + str(int(channel)) + '_OFF'
        response = self.exchange_command(command)

        if response is not None:
            with self._state_lock:
                self.led_ma[color] = -1
        else:
            logger.warning(f'[LED Class ] led_off(ch={channel}) got no response')

    def led_on_fast(self, channel, mA):
        """Fast write-only version of led_on for time-critical toggling."""
        color = self.ch2color(channel=channel)
        with self._state_lock:
            self.led_ma[color] = mA
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        self._write_command_fast(command)

    def led_off_fast(self, channel):
        """Fast write-only version of led_off for time-critical toggling."""
        color = self.ch2color(channel=channel)
        with self._state_lock:
            self.led_ma[color] = -1
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
