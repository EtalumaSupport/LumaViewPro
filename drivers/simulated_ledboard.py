# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated LED Board -- drop-in replacement for LEDBoard.

No serial hardware required. Tracks LED state, returns realistic responses,
and supports configurable delays to simulate real timing.

Timing modes:
  'fast'      -- zero delays (for tests)
  'realistic' -- serial delays matching real hardware (~12ms per command)

Failure injection (for testing error recovery):
  fail_after=N      -- disconnect after N commands (simulates USB cable pull)
  fail_on={'LEDS_ENT'} -- return None for specific commands (simulates timeout)
"""

import logging
import threading
import time
from typing import ClassVar
from lvp_logger import logger
from drivers.registry import led_registry

# SIM-SERIAL-LOG: emit the same serial.log line shape that the real
# SerialBoard does (`{label} {command} -> {resp} ({elapsed_ms}ms)`).
# Without this, sim runs leave serial.log empty and the operator can't
# inspect "what is LVP actually sending? are there duplicates? what's
# the timing?" -- which is the whole reason for running in sim mode.
_serial_log = logging.getLogger('LVP.serial')


@led_registry.register('sim', priority=100, is_simulator=True)
class SimulatedLEDBoard:
    TIMING_INSTANT: ClassVar[dict] = {'delay': 0.0}  # Zero delay -- for unit tests only
    TIMING_FAST: ClassVar[dict] = {'delay': 0.001}  # 1ms minimum -- nothing returns instantly
    TIMING_REALISTIC: ClassVar[dict] = {
        'delay': 0.012
    }  # ~12ms per exchange (1ms flush + 10ms write + 1ms read)

    _COLOR_TO_CH: ClassVar[dict] = {
        'Blue': 0,
        'Green': 1,
        'Red': 2,
        'BF': 3,
        'PC': 4,
        'DF': 5,
    }
    _CH_TO_COLOR: ClassVar[dict] = {v: k for k, v in _COLOR_TO_CH.items()}

    def __init__(
        self,
        delay: float = 0.0,
        timing: str = 'fast',
        firmware_version: str = '2.0.1',
        protocol_version: str = 'legacy',  # v3.0 STUB: 'legacy' or 'v3'
        supports_firmware_stim: bool = False,
        fail_after: int | None = None,
        fail_on: set | None = None,
        **kwargs,
    ):
        logger.info('[LED Sim   ] SimulatedLEDBoard.__init__()')
        self.found = True
        self._lock = threading.RLock()
        self.port = '/dev/simulated_led'
        self.baudrate = 115200
        self.driver = True  # truthy sentinel -- not a real serial port
        self._delay = delay
        self.firmware_version = firmware_version  # Configurable for testing old firmware paths
        self.protocol_version = protocol_version  # v3.0 STUB: for future v3.0 simulation testing
        self._supports_firmware_stim = supports_firmware_stim

        # Failure injection
        self._fail_after = fail_after  # disconnect after N commands
        self._fail_on = fail_on or set()  # return None for these commands
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
        self._channel_states = dict.fromkeys(range(6), 0)  # channel -> mA

    def set_timing_mode(self, mode: str) -> None:
        """Switch timing mode: 'instant', 'fast', or 'realistic'.

        Args:
            mode: One of ``'instant'``, ``'fast'``, ``'realistic'``.

        Raises:
            ValueError: ``mode`` is not a known preset.
        """
        presets = {
            'instant': self.TIMING_INSTANT,
            'fast': self.TIMING_FAST,
            'realistic': self.TIMING_REALISTIC,
        }
        if mode not in presets:
            raise ValueError(
                f"Unknown timing mode: {mode!r}. Use 'instant', 'fast', or 'realistic'."
            )
        preset = presets[mode]
        self._delay = preset['delay']
        self._timing_mode = mode

    @property
    def is_v2(self) -> bool:
        """True if firmware is v2.0 or later.

        Returns:
            bool: True when the parsed major version is >= 2.
        """
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
    def connect(self) -> None:
        """Mark the simulated board as connected (idempotent)."""
        with self._lock:
            self.driver = True
            logger.info('[LED Sim   ] SimulatedLEDBoard.connect()')

    def disconnect(self) -> None:
        """Mark the simulated board as disconnected and clear LED state."""
        with self._lock:
            self.driver = None
            self.port = None
            for ch in self._channel_states:
                self._channel_states[ch] = 0
            logger.info('[LED Sim   ] SimulatedLEDBoard.disconnect()')

    def is_connected(self) -> bool:
        """Whether the simulated board is currently connected.

        Returns:
            bool: True when ``connect()`` has been called and ``disconnect()`` has not.
        """
        return self.driver is not None

    # ------------------------------------------------------------------
    # Serial simulation
    # ------------------------------------------------------------------
    def _sim_delay(self):
        if self._delay > 0:
            time.sleep(self._delay)

    def exchange_command(self, command, response_numlines=1, timeout=None):
        """Exchange a single command with the simulated firmware.

        Honors failure injection (``fail_after`` / ``fail_on``) so callers
        can exercise disconnect / timeout paths without real hardware.

        Args:
            command: Command string to dispatch.
            response_numlines: Accepted for API parity; ignored by the simulator.
            timeout: Accepted for API parity; ignored by the simulator.

        Returns:
            Response string ``f'RE: {command}'``, or None on injected failure.
        """
        with self._lock:
            t_start = time.monotonic()
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
                logger.warning(
                    f'[LED Sim   ] INJECTED FAILURE: disconnect after {self._fail_after} commands'
                )
                self.driver = None
                self.found = False
                _serial_log.warning(
                    f'[LED Sim] {command} -> INJECTED DISCONNECT '
                    f'({(time.monotonic() - t_start) * 1000:.1f}ms)'
                )
                return None

            # Failure injection: fail on specific commands
            if command.strip() in self._fail_on:
                logger.warning(f'[LED Sim   ] INJECTED FAILURE: timeout on {command.strip()}')
                _serial_log.warning(
                    f'[LED Sim] {command} -> INJECTED TIMEOUT '
                    f'({(time.monotonic() - t_start) * 1000:.1f}ms)'
                )
                return None

            self._sim_delay()
            response = f'RE: {command}'
            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.debug(f'[LED Sim   ] exchange_command({command}) -> {response}')
            resp_repr = repr(response)
            if len(resp_repr) > 200:
                resp_repr = resp_repr[:200] + '...'
            _serial_log.info(f'[LED Sim] {command} -> {resp_repr} ({elapsed_ms:.1f}ms)')
            return response

    def exchange_multiline(self, command, timeout=60, end_markers=None):
        """Simulated multi-line response.

        Args:
            command: Command string to dispatch.
            timeout: Accepted for API parity; ignored by the simulator.
            end_markers: Accepted for API parity; ignored by the simulator.

        Returns:
            Response string from ``exchange_command()``.
        """
        return self.exchange_command(command)

    def _write_command_fast(self, command: str):
        with self._lock:
            t_start = time.monotonic()
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
                logger.warning(
                    f'[LED Sim   ] INJECTED FAILURE: disconnect after {self._fail_after} commands'
                )
                self.driver = None
                self.found = False
                _serial_log.warning(
                    f'[LED Sim] {command} -> INJECTED DISCONNECT (write_fast) '
                    f'({(time.monotonic() - t_start) * 1000:.1f}ms)'
                )
                return

            # No delay on fast path
            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.debug(f'[LED Sim   ] _write_command_fast({command})')
            # SIM-SERIAL-LOG: write-only path -> mark as TX with no response
            _serial_log.info(f'[LED Sim] {command} -> TX (write_fast, {elapsed_ms:.1f}ms)')

    def _close_driver(self):
        self.driver = None

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # LED control
    # ------------------------------------------------------------------
    def leds_enable(self) -> None:
        """Enable all LED channels (master enable). Sends ``LEDS_ENT``."""
        self._enabled = True
        self.exchange_command('LEDS_ENT')

    def leds_disable(self) -> None:
        """Disable all LED channels (master disable) and clear cached state."""
        self._enabled = False
        for color in self.led_ma:
            self.led_ma[color] = -1
        for ch in self._channel_states:
            self._channel_states[ch] = 0
        self.exchange_command('LEDS_ENF')

    def supports_firmware_stim(self) -> bool:
        """Return the configured firmware-STIM-support flag.

        Test-injectable: pass ``supports_firmware_stim=True`` to the
        constructor to simulate v3.0.8+ firmware; defaults to False so
        existing sim runs match pre-v3.0.8 behavior.
        """
        return self._supports_firmware_stim

    def get_status(self) -> str:
        """Return a synthetic STATUS line describing currently-on channels.

        Note: real LED firmware does not implement STATUS; the simulator
        provides a useful response for tests that exercise the call.

        Returns:
            str: ``'RE: STATUS LEDx:ymA ...'`` or ``'RE: STATUS ALL_OFF'``.
        """
        on_channels = [ch for ch, ma in self._channel_states.items() if ma > 0]
        if on_channels:
            status_str = ' '.join(f'LED{ch}:{self._channel_states[ch]}mA' for ch in on_channels)
            return f'RE: STATUS {status_str}'
        return 'RE: STATUS ALL_OFF'

    def wait_until_on(self) -> None:
        """Block until ``get_status()`` reports a STATUS line.

        The simulator returns a STATUS line on the first call, so this
        is effectively non-blocking; provided for API parity.
        """
        status = self.get_status()
        while 'STATUS' not in status:
            status = self.get_status()

    # State-query methods have been retired; see ledboard.py for
    # rationale.

    def led_on(self, channel: int, mA: int, block: bool = False) -> None:
        """Turn on the LED on a channel at a given current.

        Args:
            channel: Channel number (0-5).
            mA: Drive current in milliamps.
            block: Accepted for API parity; ignored by the simulator.
        """
        color = self.ch2color(channel)
        self.led_ma[color] = mA
        self._channel_states[channel] = mA
        self.exchange_command(f'LED{int(channel)}_{int(mA)}')

    def led_off(self, channel: int) -> None:
        """Turn off the LED on a channel.

        Args:
            channel: Channel number (0-5).
        """
        color = self.ch2color(channel)
        self.led_ma[color] = -1
        self._channel_states[channel] = 0
        self.exchange_command(f'LED{int(channel)}_OFF')

    def led_on_fast(self, channel: int, mA: int) -> None:
        """Fast write-only version of led_on for time-critical toggling.

        Args:
            channel: Channel number (0-5).
            mA: Drive current in milliamps.
        """
        color = self.ch2color(channel)
        self.led_ma[color] = mA
        self._channel_states[channel] = mA
        self._write_command_fast(f'LED{int(channel)}_{int(mA)}')

    def led_off_fast(self, channel: int) -> None:
        """Fast write-only version of led_off for time-critical toggling.

        Args:
            channel: Channel number (0-5).
        """
        color = self.ch2color(channel)
        self.led_ma[color] = -1
        self._channel_states[channel] = 0
        self._write_command_fast(f'LED{int(channel)}_OFF')

    def leds_off(self) -> None:
        """Turn off every LED channel. Clears the cached per-channel state."""
        for color in self.led_ma:
            self.led_ma[color] = -1
        for ch in self._channel_states:
            self._channel_states[ch] = 0
        self.exchange_command('LEDS_OFF')

    def leds_off_fast(self) -> None:
        """Fast write-only version of leds_off."""
        for color in self.led_ma:
            self.led_ma[color] = -1
        for ch in self._channel_states:
            self._channel_states[ch] = 0
        self._write_command_fast('LEDS_OFF')

    # ------------------------------------------------------------------
    # Engineering mode and diagnostics (match LEDBoard API)
    # ------------------------------------------------------------------
    def enter_engineering_mode(self, timeout: float = 5.0) -> bool:
        """Simulated engineering mode entry -- always succeeds.

        Returns:
            bool: Always True; sim never reproduces the
                no-response / no-Y/N-prompt failure modes that the
                real LEDBoard.enter_engineering_mode raises HardwareError on.
        """
        logger.info('[LED Sim   ] enter_engineering_mode()')
        return True

    def exit_engineering_mode(self) -> str:
        """Simulated engineering mode exit.

        Returns:
            str: Always ``'Q'`` (mirrors the firmware Q command).
        """
        logger.info('[LED Sim   ] exit_engineering_mode()')
        return 'Q'

    def selftest(self, timeout: float = 180) -> list:
        """Simulated SELFTEST -- returns fake result lines.

        Args:
            timeout: Accepted for API parity; ignored by the simulator.

        Returns:
            list: One line per channel plus a final ``'SELFTEST Complete'``.
        """
        lines = []
        for ch in range(6):
            lines.append(f'LED{ch}: 0.1mA OK  1mA OK  10mA OK  100mA OK  500mA OK')
        lines.append('SELFTEST Complete')
        return lines

    def get_info(self) -> dict:
        """Simulated INFO -- returns version dict.

        Returns:
            dict: ``{'raw': str, 'version': str}``.
        """
        return {
            'raw': f'Simulated LED Board v{self.firmware_version}',
            'version': self.firmware_version,
        }

    def detect_firmware_version(self) -> None:
        """No-op for simulator -- version is set at construction."""
        pass

    def read_led_current(self, channel: int) -> float | None:
        """Simulated ADC feedback -- returns the set current for the channel.

        Args:
            channel: Channel number (0-5).

        Returns:
            float | None: Set current in mA, 0.0 for unknown channels,
                or None when firmware is too old (not v2+).
        """
        if not self.is_v2:
            return None
        ch = int(channel)
        if ch in self._channel_states:
            return float(self._channel_states[ch])
        return 0.0

    # ------------------------------------------------------------------
    # Raw REPL stubs (match SerialBoard API surface)
    # ------------------------------------------------------------------
    def enter_raw_repl(self) -> bool:
        """Stub: simulated raw REPL entry always succeeds.

        Returns:
            bool: Always True.
        """
        return True

    def exit_raw_repl(self) -> None:
        """Stub: simulated raw REPL exit is a no-op."""
        pass

    def repl_exec(self, code: str, timeout: int = 10) -> tuple[bytes, bytes]:
        """Stub: pretend to execute code in raw REPL.

        Args:
            code: Source code to run (accepted but unused).
            timeout: Execution timeout (accepted but unused).

        Returns:
            tuple[bytes, bytes]: Empty ``(stdout, stderr)`` tuple.
        """
        return (b'', b'')

    def repl_list_files(self) -> list:
        """Stub: simulator has no on-board filesystem.

        Returns:
            list: Always empty.
        """
        return []

    def repl_read_file(self, filename: str, verify: bool = True):
        """Stub: simulator has no on-board filesystem.

        Args:
            filename: File name (accepted but unused).
            verify: SHA256 verify flag (accepted but unused).

        Returns:
            None: Always.
        """
        return None

    def repl_write_file(self, filename: str, data) -> bool:
        """Stub: simulator pretends every write succeeds.

        Args:
            filename: File name (accepted but unused).
            data: File contents (accepted but unused).

        Returns:
            bool: Always True.
        """
        return True

    def verify_firmware_running(self, timeout: int = 10) -> str:
        """Stub: simulator firmware always reports running.

        Args:
            timeout: Maximum wait time in seconds (accepted but unused).

        Returns:
            str: Always ``'Simulated firmware running'``.
        """
        return 'Simulated firmware running'
