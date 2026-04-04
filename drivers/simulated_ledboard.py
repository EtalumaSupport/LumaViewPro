# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated LED Board — drop-in replacement for LEDBoard.

No serial hardware required. Tracks LED state, returns realistic responses,
and supports configurable delays to simulate real timing.

Timing modes:
  'fast'      — zero delays (for tests)
  'realistic' — serial delays matching real hardware (~12ms per command)

Failure injection (for testing error recovery):
  fail_after=N      — disconnect after N commands (simulates USB cable pull)
  fail_on={'LEDS_ENT'} — return None for specific commands (simulates timeout)
"""

import threading
import time
from lvp_logger import logger


class SimulatedLEDBoard:

    TIMING_INSTANT = {'delay': 0.0}     # Zero delay — for unit tests only
    TIMING_FAST = {'delay': 0.001}      # 1ms minimum — nothing returns instantly
    TIMING_REALISTIC = {'delay': 0.012}  # ~12ms per exchange (1ms flush + 10ms write + 1ms read)

    def __init__(self, delay: float = 0.0, timing: str = 'fast',
                 firmware_version: str = '2.0.1',
                 protocol_version: str = 'legacy',  # v3.0 STUB: 'legacy' or 'v3'
                 fail_after: int | None = None,
                 fail_on: set | None = None,
                 **kwargs):
        logger.info('[LED Sim   ] SimulatedLEDBoard.__init__()')
        self.found = True
        self._lock = threading.RLock()
        self.port = '/dev/simulated_led'
        self.baudrate = 115200
        self.driver = True  # truthy sentinel — not a real serial port
        self._delay = delay
        self.firmware_version = firmware_version  # Configurable for testing old firmware paths
        self.protocol_version = protocol_version  # v3.0 STUB: for future v3.0 simulation testing

        # Failure injection
        self._fail_after = fail_after          # disconnect after N commands
        self._fail_on = fail_on or set()       # return None for these commands
        self._cmd_count = 0

        # Apply timing preset (overrides delay if preset given)
        self.set_timing_mode(timing)
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

    def set_timing_mode(self, mode: str):
        """Switch timing mode: 'instant', 'fast', or 'realistic'."""
        presets = {
            'instant': self.TIMING_INSTANT,
            'fast': self.TIMING_FAST,
            'realistic': self.TIMING_REALISTIC,
        }
        if mode not in presets:
            raise ValueError(f"Unknown timing mode: {mode!r}. Use 'instant', 'fast', or 'realistic'.")
        preset = presets[mode]
        self._delay = preset['delay']
        self._timing_mode = mode

    @property
    def is_v2(self) -> bool:
        """True if firmware is v2.0 or later."""
        if self.firmware_version is None:
            return False
        try:
            major = int(self.firmware_version.split('.')[0])
            return major >= 2
        except (ValueError, IndexError):
            return False

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

    def exchange_command(self, command, response_numlines=1, timeout=None):
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return None
            if self.driver is None:
                return None

            # Failure injection: disconnect after N commands
            self._cmd_count += 1
            if self._fail_after is not None and self._cmd_count > self._fail_after:
                logger.warning(f'[LED Sim   ] INJECTED FAILURE: disconnect after {self._fail_after} commands')
                self.driver = None
                self.found = False
                return None

            # Failure injection: fail on specific commands
            cmd_word = command.strip().split('_')[0] if command else ''
            if command.strip() in self._fail_on:
                logger.warning(f'[LED Sim   ] INJECTED FAILURE: timeout on {command.strip()}')
                return None

            self._sim_delay()
            response = f"RE: {command}"
            logger.debug(f'[LED Sim   ] exchange_command({command}) -> {response}')
            return response

    def exchange_multiline(self, command, timeout=60, end_markers=None):
        """Simulated multi-line response."""
        return self.exchange_command(command)

    def _write_command_fast(self, command: str):
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return
            if self.driver is None:
                return

            # Failure injection (same as exchange_command)
            self._cmd_count += 1
            if self._fail_after is not None and self._cmd_count > self._fail_after:
                logger.warning(f'[LED Sim   ] INJECTED FAILURE: disconnect after {self._fail_after} commands')
                self.driver = None
                self.found = False
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

    # ------------------------------------------------------------------
    # Engineering mode and diagnostics (match LEDBoard API)
    # ------------------------------------------------------------------
    def enter_engineering_mode(self, timeout=5.0):
        """Simulated engineering mode entry — always succeeds."""
        logger.info('[LED Sim   ] enter_engineering_mode()')
        return True

    def exit_engineering_mode(self):
        """Simulated engineering mode exit."""
        logger.info('[LED Sim   ] exit_engineering_mode()')
        return 'Q'

    def selftest(self, timeout=180):
        """Simulated SELFTEST — returns fake result lines."""
        lines = []
        for ch in range(6):
            ma = self._channel_states.get(ch, 0)
            lines.append(f'LED{ch}: 0.1mA OK  1mA OK  10mA OK  100mA OK  500mA OK')
        lines.append('SELFTEST Complete')
        return lines

    def get_info(self):
        """Simulated INFO — returns version dict."""
        return {
            'raw': f'Simulated LED Board v{self.firmware_version}',
            'version': self.firmware_version,
        }

    def detect_firmware_version(self):
        """No-op for simulator — version is set at construction."""
        pass

    def read_led_current(self, channel):
        """Simulated ADC feedback — returns the set current for the channel, or None if not v2."""
        if not self.is_v2:
            return None
        ch = int(channel)
        if ch in self._channel_states:
            return float(self._channel_states[ch])
        return 0.0

    # ------------------------------------------------------------------
    # Raw REPL stubs (match SerialBoard API surface)
    # ------------------------------------------------------------------
    def enter_raw_repl(self):
        return True

    def exit_raw_repl(self):
        pass

    def repl_exec(self, code, timeout=10):
        return (b'', b'')

    def repl_list_files(self):
        return []

    def repl_read_file(self, filename, verify=True):
        return None

    def repl_write_file(self, filename, data):
        return True

    def verify_firmware_running(self, timeout=10):
        return 'Simulated firmware running'
