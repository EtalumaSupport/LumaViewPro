"""
Simulated LED Board — drop-in replacement for LEDBoard.

No serial hardware required. Tracks LED state, returns realistic responses,
and supports configurable delays to simulate real timing.
"""

import threading
import time
from lvp_logger import logger


class SimulatedLEDBoard:

    def __init__(self, delay: float = 0.001, **kwargs):
        logger.info('[LED Sim   ] SimulatedLEDBoard.__init__()')
        self.found = True
        self._lock = threading.RLock()
        self.port = '/dev/simulated_led'
        self.baudrate = 115200
        self.driver = True  # truthy sentinel — not a real serial port
        self._delay = delay
        self.led_ma = {
            'BF': -1,
            'PC': -1,
            'DF': -1,
            'Red': -1,
            'Blue': -1,
            'Green': -1,
        }
        self._enabled = True
        self._channel_states = {i: 0 for i in range(6)}  # channel -> mA

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        with self._lock:
            self.driver = True
            logger.info('[LED Sim   ] SimulatedLEDBoard.connect()')

    def disconnect(self):
        with self._lock:
            self.driver = None
            self.port = None
            for ch in self._channel_states:
                self._channel_states[ch] = 0
            logger.info('[LED Sim   ] SimulatedLEDBoard.disconnect()')

    def is_connected(self) -> bool:
        return self.driver is not None

    # ------------------------------------------------------------------
    # Serial simulation
    # ------------------------------------------------------------------
    def _sim_delay(self):
        if self._delay > 0:
            time.sleep(self._delay)

    def exchange_command(self, command):
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return None
            if self.driver is None:
                return None

            self._sim_delay()
            response = f"RE: {command}"
            logger.debug(f'[LED Sim   ] exchange_command({command}) -> {response}')
            return response

    def _write_command_fast(self, command: str):
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return
            if self.driver is None:
                return
            # No delay on fast path
            logger.debug(f'[LED Sim   ] _write_command_fast({command})')

    def _close_driver(self):
        self.driver = None

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------
    def color2ch(self, color):
        return {'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3, 'PC': 4, 'DF': 5}.get(color, 3)

    def ch2color(self, channel):
        return {0: 'Blue', 1: 'Green', 2: 'Red', 3: 'BF', 4: 'PC', 5: 'DF'}.get(channel, 'BF')

    # ------------------------------------------------------------------
    # LED control
    # ------------------------------------------------------------------
    def leds_enable(self):
        self._enabled = True
        self.exchange_command('LEDS_ENT')

    def leds_disable(self):
        self._enabled = False
        for color in self.led_ma:
            self.led_ma[color] = -1
        for ch in self._channel_states:
            self._channel_states[ch] = 0
        self.exchange_command('LEDS_ENF')

    def get_status(self):
        on_channels = [ch for ch, ma in self._channel_states.items() if ma > 0]
        if on_channels:
            status_str = ' '.join(f'LED{ch}:{self._channel_states[ch]}mA' for ch in on_channels)
            return f'RE: STATUS {status_str}'
        return 'RE: STATUS ALL_OFF'

    def wait_until_on(self):
        status = self.get_status()
        while "STATUS" not in status:
            status = self.get_status()

    def get_led_ma(self, color):
        return self.led_ma.get(color, -1)

    def is_led_on(self, color) -> bool:
        return self.led_ma.get(color, -1) > 0

    def get_led_state(self, color) -> dict:
        return {
            'enabled': self.is_led_on(color),
            'illumination': self.get_led_ma(color),
        }

    def get_led_states(self) -> dict:
        return {color: self.get_led_state(color) for color in self.led_ma}

    def led_on(self, channel, mA, block=False):
        color = self.ch2color(channel)
        self.led_ma[color] = mA
        self._channel_states[channel] = mA
        self.exchange_command(f'LED{int(channel)}_{int(mA)}')

    def led_off(self, channel):
        color = self.ch2color(channel)
        self.led_ma[color] = -1
        self._channel_states[channel] = 0
        self.exchange_command(f'LED{int(channel)}_OFF')

    def led_on_fast(self, channel, mA):
        color = self.ch2color(channel)
        self.led_ma[color] = mA
        self._channel_states[channel] = mA
        self._write_command_fast(f'LED{int(channel)}_{int(mA)}')

    def led_off_fast(self, channel):
        color = self.ch2color(channel)
        self.led_ma[color] = -1
        self._channel_states[channel] = 0
        self._write_command_fast(f'LED{int(channel)}_OFF')

    def leds_off(self):
        for color in self.led_ma:
            self.led_ma[color] = -1
        for ch in self._channel_states:
            self._channel_states[ch] = 0
        self.exchange_command('LEDS_OFF')

    def leds_off_fast(self):
        for color in self.led_ma:
            self.led_ma[color] = -1
        for ch in self._channel_states:
            self._channel_states[ch] = 0
        self._write_command_fast('LEDS_OFF')
