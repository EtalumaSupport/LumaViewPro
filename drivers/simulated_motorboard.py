# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated Motor Board -- drop-in replacement for MotorBoard.

No serial hardware required. Tracks axis positions, simulates homing
and movement, and supports configurable delays.

Timing modes:
  'fast'      -- instant movement, zero delays (for tests)
  'realistic' -- serial delays and speed-limited movement matching real hardware

Failure injection (for testing error recovery):
  fail_after=N      -- disconnect after N commands (simulates USB cable pull)
  fail_on={'ZHOME'} -- return None for specific commands (simulates timeout)
"""

import logging
import pathlib
import threading
import time
from typing import ClassVar
from lvp_logger import logger
from drivers.exceptions import HardwareError
from drivers.motorconfig import MotorConfig
from drivers.registry import motor_registry

# SIM-SERIAL-LOG: emit the same serial.log line shape that the real
# SerialBoard does (`{label} {command} -> {resp} ({elapsed_ms}ms)`).
# See drivers/simulated_ledboard.py for rationale.
_serial_log = logging.getLogger('LVP.serial')


@motor_registry.register('sim', priority=100, is_simulator=True)
class SimulatedMotorBoard:
    # Axis speeds in usteps/sec (realistic values for Etaluma hardware)
    AXIS_SPEEDS: ClassVar[dict] = {
        'X': 20157 * 50,  # ~50 mm/s
        'Y': 20157 * 50,  # ~50 mm/s
        'Z': 170666 * 5,  # ~5 mm/s
        'T': 80000,  # ~90 deg/s
    }

    # Homing durations in seconds (realistic)
    HOMING_DURATIONS: ClassVar[dict] = {
        'XYZ': 3.0,
        'Z': 1.5,
        'T': 1.0,
    }

    # Timing presets
    TIMING_INSTANT: ClassVar[dict] = {
        'cmd_delay': 0.0,
        'move_delay': 0.0,
        'simulate_move_duration': False,  # Truly instant -- for unit tests only
        'fast_move_duration': 0.0,
    }
    TIMING_FAST: ClassVar[dict] = {
        'cmd_delay': 0.001,  # 1ms minimum -- nothing returns instantly
        'move_delay': 0.0,
        'simulate_move_duration': True,  # Simulates brief move duration
        'fast_move_duration': 0.003,  # 3ms per move in fast mode
    }
    TIMING_REALISTIC: ClassVar[dict] = {
        'cmd_delay': 0.003,  # ~3ms serial round-trip
        'move_delay': 0.0,  # homing uses HOMING_DURATIONS instead
        'simulate_move_duration': True,
        'fast_move_duration': 0.0,  # Use ramp-calculated duration
    }

    def __init__(
        self,
        model: str = 'LS850',
        serial_number: str = 'SIM-001',
        move_delay: float = 0.0,
        cmd_delay: float = 0.0,
        timing: str = 'fast',
        firmware_version: str = '2.0.1',
        protocol_version: str = 'legacy',  # v3.0 STUB: 'legacy' or 'v3'
        motorconfig_defaults_file: pathlib.Path | None = None,
        fail_after: int | None = None,
        fail_on: set | None = None,
        **kwargs,
    ):
        logger.info('[XYZ Sim   ] SimulatedMotorBoard.__init__()')

        # Failure injection
        self._fail_after = fail_after  # disconnect after N commands
        self._fail_on = fail_on or set()  # return None for these commands
        self._cmd_count = 0

        # Load hardware config (same defaults as real MotorBoard)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path('data/motorconfig_defaults.json')
        self.motorconfig = MotorConfig(defaults_file=motorconfig_defaults_file)

        self.found = True
        self.overshoot = False
        self.backlash = self.motorconfig.antibacklash_um('Z')
        self._has_turret = model.endswith('T')
        self.initial_homing_complete = False
        self.initial_t_homing_complete = False
        self.port = '/dev/simulated_motor'
        self.thread_lock = threading.RLock()
        self.driver = True  # truthy sentinel
        self._fullinfo = {'model': model, 'serial_number': serial_number}
        self._connect_fails = 0
        self._cmd_delay = cmd_delay
        self._move_delay = move_delay
        self._simulate_move_duration = False
        self.firmware_version = firmware_version  # Configurable for testing old firmware paths
        self.protocol_version = protocol_version  # v3.0 STUB: for future v3.0 simulation testing

        # Apply timing preset (overrides cmd_delay/move_delay if preset given)
        self.set_timing_mode(timing)

        # Internal position state (in usteps)
        self._actual = {'X': 0, 'Y': 0, 'Z': 0, 'T': 0}
        self._target = {'X': 0, 'Y': 0, 'Z': 0, 'T': 0}
        self._homed = {'X': False, 'Y': False, 'Z': False, 'T': False}

        # Move timing state
        self._move_start_pos = {'X': 0, 'Y': 0, 'Z': 0, 'T': 0}
        self._move_start_time = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'T': 0.0}
        self._move_end_time = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'T': 0.0}

        # Re-apply timing mode after all state is initialized
        self.set_timing_mode(timing)

        self.axes_config = {
            'Z': {
                'limits': {'min': 0.0, 'max': self.motorconfig.travel_limit_um('Z')},
                'move_func': self.z_um2ustep,
            },
            'X': {
                'limits': {'min': 0.0, 'max': self.motorconfig.travel_limit_um('X')},
                'move_func': self.xy_um2ustep,
            },
            'Y': {
                'limits': {'min': 0.0, 'max': self.motorconfig.travel_limit_um('Y')},
                'move_func': self.xy_um2ustep,
            },
            'T': {'move_func': self.t_pos2ustep},
        }

    def set_timing_mode(self, mode: str) -> None:
        """Switch timing mode: 'instant', 'fast', or 'realistic'.

        instant: zero delays, truly instant moves. For unit tests only.
        fast: 1ms command delay, 3ms move duration. Default for --simulate.
        realistic: 3ms command delay, TMC5072 ramp-calculated move durations.

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
        self._cmd_delay = preset['cmd_delay']
        self._move_delay = preset['move_delay']
        self._simulate_move_duration = preset['simulate_move_duration']
        self._fast_move_duration = preset.get('fast_move_duration', 0.0)
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
        with self.thread_lock:
            self.driver = True
            self._connect_fails = 0
            logger.info('[XYZ Sim   ] SimulatedMotorBoard.connect()')

    def disconnect(self) -> None:
        """Mark the simulated board as disconnected."""
        with self.thread_lock:
            self.driver = None
            logger.info('[XYZ Sim   ] SimulatedMotorBoard.disconnect()')

    def is_connected(self) -> bool:
        """Whether the simulated board is currently connected.

        Returns:
            bool: True when ``connect()`` has been called and ``disconnect()`` has not.
        """
        return self.driver is not None

    def motor_stop(self) -> bool:
        """Simulator answers True (sim firmware always supports STOP).
        Mirrors the production MotorBoard method so
        Lumascope.stop_motion works identically against the simulator.

        Returns:
            bool: Always True.
        """
        return True

    def supports_motor_stop(self) -> bool:
        """Sim firmware supports every command family."""
        return True

    def supports_fan(self) -> bool:
        """Sim firmware supports every command family."""
        return True

    def supports_diagnostics(self) -> bool:
        """Sim firmware supports every command family."""
        return True

    def _close_driver(self):
        self.driver = None

    # ------------------------------------------------------------------
    # Serial simulation
    # ------------------------------------------------------------------
    # Fast SPI register reads -- return in ~100-200us on real hardware.
    # Matched by startswith(), so 'STATUS_RZ' matches 'STATUS_R'.
    _FAST_PREFIXES = ('STATUS_R', 'TARGET_R', 'ACTUAL_R', 'VOLTAGE', 'CURRENT')

    def _sim_delay(self, command: str = ''):
        if self._cmd_delay <= 0:
            return
        # Fast register reads get a much shorter delay than general commands
        if command and command.startswith(self._FAST_PREFIXES):
            time.sleep(self._cmd_delay * 0.1)  # ~0.1ms for status reads
        else:
            time.sleep(self._cmd_delay)

    def exchange_command(self, command, response_numlines=1, timeout=None):
        """Exchange a single command with the simulated firmware.

        Honors failure injection (``fail_after`` / ``fail_on``) so callers
        can exercise disconnect / timeout paths without real hardware.

        Args:
            command: Command string to dispatch.
            response_numlines: 1 returns a single string; otherwise
                returns a single-element list (mirrors SerialBoard).
            timeout: Accepted for API parity; ignored by the simulator.

        Returns:
            Response string (or single-element list, or None on injected
            failure).
        """
        with self.thread_lock:
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
                    f'[XYZ Sim   ] INJECTED FAILURE: disconnect after {self._fail_after} commands'
                )
                self.driver = None
                self.found = False
                _serial_log.warning(
                    f'[XYZ Sim] {command} -> INJECTED DISCONNECT '
                    f'({(time.monotonic() - t_start) * 1000:.1f}ms)'
                )
                return None

            # Failure injection: fail on specific commands
            cmd_word = command.strip().split()[0] if command else ''
            if cmd_word in self._fail_on:
                logger.warning(f'[XYZ Sim   ] INJECTED FAILURE: timeout on {cmd_word}')
                _serial_log.warning(
                    f'[XYZ Sim] {command} -> INJECTED TIMEOUT '
                    f'({(time.monotonic() - t_start) * 1000:.1f}ms)'
                )
                return None

            self._sim_delay(command)
            response = self._handle_command(command)
            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.debug(f'[XYZ Sim   ] exchange_command({command}) -> {response}')
            resp_repr = repr(response)
            if len(resp_repr) > 200:
                resp_repr = resp_repr[:200] + '...'
            _serial_log.info(f'[XYZ Sim] {command} -> {resp_repr} ({elapsed_ms:.1f}ms)')
            if response_numlines == 1:
                return response
            return [response]

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

    def _handle_command(self, command):
        cmd = command.strip()

        if cmd == 'INFO':
            return f'Etaluma Motor Controller {self._fullinfo["model"]} Firmware: SIMULATED'

        if cmd == 'FULLINFO':
            model = self._fullinfo['model']
            sn = self._fullinfo['serial_number']
            return f'Model: {model} Serial: {sn} Firmware: SIMULATED'

        if cmd == 'HOME':
            self._do_home('X', 'Y', 'Z')
            if self._has_turret:
                self._do_home('T')
            self.initial_homing_complete = True
            return 'XYZ home complete'

        if cmd == 'ZHOME':
            self._do_home('Z')
            return 'Z home successful'

        if cmd == 'THOME':
            self._do_home('T')
            self.initial_t_homing_complete = True
            return 'T home successful'

        if cmd == 'CENTER':
            mid_x = self.xy_um2ustep(self.motorconfig.travel_limit_um('X') / 2)
            mid_y = self.xy_um2ustep(self.motorconfig.travel_limit_um('Y') / 2)
            self._actual['X'] = mid_x
            self._target['X'] = mid_x
            self._actual['Y'] = mid_y
            self._target['Y'] = mid_y
            return 'CENTER complete'

        # TARGET_W<axis><value>
        if cmd.startswith('TARGET_W'):
            axis = cmd[8]
            value = int(cmd[9:])
            if value >= 0x80000000:
                value -= 0x100000000
            self._move_start_pos[axis] = self._actual[axis]
            self._move_start_time[axis] = time.monotonic()
            self._target[axis] = value

            if self._simulate_move_duration:
                if self._fast_move_duration > 0:
                    # Fast mode: position updates instantly but target_status
                    # returns False for a brief period so the motion monitor
                    # can detect the MOVING->IDLE transition.
                    self._actual[axis] = value
                    self._move_end_time[axis] = time.monotonic() + self._fast_move_duration
                else:
                    # Realistic mode: trapezoidal ramp calculation with interpolation
                    duration = self._calc_move_duration(axis, self._move_start_pos[axis], value)
                    self._move_end_time[axis] = time.monotonic() + duration
            else:
                self._actual[axis] = value
                self._move_end_time[axis] = 0.0
            return str(value)

        # TARGET_R<axis>
        if cmd.startswith('TARGET_R'):
            axis = cmd[8]
            return str(self._target.get(axis, 0))

        # ACTUAL_R<axis>
        if cmd.startswith('ACTUAL_R'):
            axis = cmd[8]
            self._update_actual(axis)
            return str(self._actual.get(axis, 0))

        # STATUS_R<axis>
        if cmd.startswith('STATUS_R'):
            axis = cmd[8]
            self._update_actual(axis)
            return str(self._make_status(axis))

        # SPI<axis>0x<addr><payload>
        if cmd.startswith('SPI'):
            return 'SPI OK'

        # Acceleration limits
        if cmd.startswith('AMAX') or cmd.startswith('DMAX'):
            return '30000'

        # Tech-support diagnostic commands. The Python-level helpers
        # (get_current / get_motordetect / read_status)
        # already return realistic shapes -- but tech_support_report
        # talks to the board via raw exchange_command (LV-24 layer), so
        # the raw-text branches need to mirror the same content.
        if cmd == 'VOLTAGE':
            return '24V=OK 5V=N/A 3V3=N/A 1V2=N/A'
        if cmd == 'FANSPEED':
            return 'FANSPEED 1500 RPM'
        if cmd == 'MOTORDETECT':
            return (
                'X: detected=True configured=True\n'
                'Y: detected=True configured=True\n'
                'Z: detected=True configured=True\n'
                'T: detected=True configured=True'
            )
        if cmd == 'CURRENT':
            return (
                'X: CS_ACTUAL=0 IRUN=10 IHOLD=3 SG_RESULT=0\n'
                'Y: CS_ACTUAL=0 IRUN=10 IHOLD=3 SG_RESULT=0\n'
                'Z: CS_ACTUAL=0 IRUN=17 IHOLD=4 SG_RESULT=0\n'
                'T: CS_ACTUAL=0 IRUN=5  IHOLD=7 SG_RESULT=0'
            )
        if cmd.startswith('DRVSTAT_'):
            axis = cmd[len('DRVSTAT_') :]
            return f'{axis}: DRV_STATUS=0x80000000 (standstill)'

        # Fan speed setter -- `FAN:<value>` (real firmware accepts a duty
        # cycle 0-100). Tech-support pulses it 50 -> 0 to verify the
        # tachometer responds; sim just acks. (Read side is FANSPEED.)
        if cmd.startswith('FAN:'):
            return f'FAN OK ({cmd[4:]})'

        # Emergency stop. Real firmware halts all motion and resets
        # target=actual on every axis; sim mirrors that.
        if cmd == 'STOP':
            for ax in ('X', 'Y', 'Z', 'T'):
                self._target[ax] = self._actual[ax]
            return 'STOP OK'

        return f'ERROR: unknown command {cmd}'

    def _update_actual(self, axis):
        """Update actual position based on elapsed time.

        In realistic mode, interpolates position along the move trajectory.
        In fast mode, snaps to target after the brief delay.
        """
        if not self._simulate_move_duration:
            return
        now = time.monotonic()
        end = self._move_end_time.get(axis, 0.0)
        if now >= end:
            self._actual[axis] = self._target[axis]
        elif self._fast_move_duration > 0:
            # Fast mode: don't interpolate, just wait for end time
            pass
        else:
            # Realistic mode: linear interpolation (close enough for simulation)
            start = self._move_start_pos.get(axis, self._actual[axis])
            target = self._target[axis]
            start_t = self._move_start_time.get(axis, now)
            duration = end - start_t
            if duration > 0:
                frac = min(1.0, (now - start_t) / duration)
                self._actual[axis] = int(start + frac * (target - start))

    def _calc_move_duration(self, axis, start_usteps, target_usteps) -> float:
        """Calculate move duration using TMC5072 trapezoidal ramp parameters.

        Uses the same ramp_params as the position predictor in lumascope_api.
        Returns duration in seconds.
        """
        distance_usteps = abs(target_usteps - start_usteps)
        if distance_usteps == 0:
            return 0.0

        try:
            ramp = self.motorconfig.ramp_params_usteps(axis)
        except Exception:
            # Fallback to simple speed-based calculation
            speed = self.AXIS_SPEEDS.get(axis, self.AXIS_SPEEDS['X'])
            return distance_usteps / speed if speed > 0 else 0.0

        # TMC5072 register -> real units conversion
        fclk = 16_000_000
        vel_factor = fclk / (2**24)
        acc_factor = fclk**2 / (512 * 2**24)

        vmax = ramp['vmax'] * vel_factor  # usteps/sec
        amax = ramp['amax'] * acc_factor  # usteps/sec^2
        dmax = ramp['dmax'] * acc_factor  # usteps/sec^2

        if vmax <= 0 or amax <= 0 or dmax <= 0:
            speed = self.AXIS_SPEEDS.get(axis, self.AXIS_SPEEDS['X'])
            return distance_usteps / speed if speed > 0 else 0.0

        t_accel = vmax / amax
        t_decel = vmax / dmax
        s_accel = 0.5 * amax * t_accel * t_accel
        s_decel = 0.5 * dmax * t_decel * t_decel

        if distance_usteps <= (s_accel + s_decel):
            # Triangular profile
            import math

            t_peak = math.sqrt(2.0 * distance_usteps / (amax + amax * amax / dmax))
            v_peak = amax * t_peak
            return t_peak + v_peak / dmax
        else:
            # Full trapezoidal
            t_cruise = (distance_usteps - s_accel - s_decel) / vmax
            return t_accel + t_cruise + t_decel

    def _do_home(self, *axes):
        if self._simulate_move_duration:
            key = ''.join(sorted(axes))
            duration = self.HOMING_DURATIONS.get(key, self.HOMING_DURATIONS.get('XYZ', 3.0))
            time.sleep(duration)
        elif self._move_delay > 0:
            time.sleep(self._move_delay)
        for axis in axes:
            self._actual[axis] = 0
            self._target[axis] = 0
            self._homed[axis] = True
            self._move_end_time[axis] = 0.0

    def _make_status(self, axis):
        status = 0
        # Bit 0: home reference (status_stop_left)
        if self._homed.get(axis, False) and self._actual.get(axis, 0) == 0:
            status |= 1 << 0
        # Bit 9: position_reached -- only True when move duration has elapsed
        # AND actual == target. In fast mode, actual is set instantly but
        # target_status waits for the brief delay before reporting reached.
        end_time = self._move_end_time.get(axis, 0.0)
        at_target = self._actual.get(axis, 0) == self._target.get(axis, 0)
        time_elapsed = time.monotonic() >= end_time
        if at_target and time_elapsed:
            status |= 1 << 9
        return status

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------
    def infomation(self) -> None:
        """Send INFO; logs response via exchange_command (typo preserved for API parity)."""
        self.exchange_command('INFO')

    def fullinfo(self) -> dict:
        """Send FULLINFO and return the parsed model + serial-number dict.

        Returns:
            dict: ``{'model': str, 'serial_number': str}``.
        """
        info = self.exchange_command('FULLINFO')
        info_parts = info.split()
        model = info_parts[info_parts.index('Model:') + 1]
        if model.endswith('T'):
            self._has_turret = True
        serial_number = info_parts[info_parts.index('Serial:') + 1]
        return {'model': model, 'serial_number': serial_number}

    def get_microscope_model(self) -> str:
        """Return the configured microscope model string.

        Returns:
            str: Model identifier (e.g. ``'LS850'``).
        """
        return self._fullinfo['model']

    def get_serial_number(self) -> str:
        """Return the configured serial number string.

        Returns:
            str: Serial number from the simulated FULLINFO.
        """
        return self._fullinfo['serial_number']

    # ------------------------------------------------------------------
    # Conversion functions (identical to real MotorBoard)
    # ------------------------------------------------------------------
    def z_ustep2um(self, ustep: int) -> float:
        """Convert Z-axis microsteps to micrometers.

        Args:
            ustep: Microstep count.

        Returns:
            float: Position in micrometers.
        """
        return ustep * 1000 / self.motorconfig.usteps_per_mm('Z')

    def z_um2ustep(self, um: float) -> int:
        """Convert Z-axis micrometers to microsteps.

        Args:
            um: Position in micrometers.

        Returns:
            int: Microstep count (truncated toward zero).
        """
        return int(self.motorconfig.usteps_per_mm('Z') * um / 1000)

    def xy_ustep2um(self, ustep: int) -> float:
        """Convert XY microsteps to micrometers.

        Args:
            ustep: Microstep count.

        Returns:
            float: Position in micrometers.
        """
        return ustep * 1000 / self.motorconfig.usteps_per_mm('X')

    def xy_um2ustep(self, um: float) -> int:
        """Convert XY micrometers to microsteps.

        Args:
            um: Position in micrometers.

        Returns:
            int: Microstep count (truncated toward zero).
        """
        return int(self.motorconfig.usteps_per_mm('X') * um / 1000)

    def t_ustep2deg(self, ustep: int) -> float:
        """Convert turret microsteps to degrees.

        Args:
            ustep: Microstep count.

        Returns:
            float: Rotation in degrees.
        """
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        return 90.0 / usteps_per_90deg * ustep

    def t_ustep2pos(self, ustep: int) -> int:
        """Convert turret microsteps to a 1-based position number.

        Args:
            ustep: Microstep count.

        Returns:
            int: Turret position (1, 2, 3, 4 ...).
        """
        return int(self.t_ustep2deg(ustep) / 90) + 1

    def t_deg2ustep(self, degrees: float) -> int:
        """Convert turret degrees to microsteps.

        Args:
            degrees: Rotation in degrees.

        Returns:
            int: Microstep count (truncated toward zero).
        """
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        return int(degrees * usteps_per_90deg / 90.0)

    def t_pos2ustep(self, position: int) -> int:
        """Convert turret position (1-based) to microsteps.

        Args:
            position: Turret position (1-based).

        Returns:
            int: Microstep count for that position. Falls back to
                90-degree spacing when motorconfig has no entry.
        """
        usteps = self.motorconfig.turret_position_usteps(position)
        if usteps == 0 and position > 1:
            return self.t_deg2ustep(90 * (position - 1))
        return usteps

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------
    def zhome(self) -> bool:
        """Simulated Z homing. Mirrors `MotorBoard.zhome` contract.

        Raises:
            HardwareError: Simulated no-response or firmware-error path.
        """
        resp = self.exchange_command('ZHOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.zhome() -> {resp}')
        if resp is None:
            raise HardwareError('zhome(): no response from motor board (timeout or disconnect)')
        if 'successful' in resp.lower() or 'complete' in resp.lower():
            return True
        raise HardwareError(f'zhome(): firmware error: {resp}')

    def home(self) -> bool:
        """Simulated full home. Mirrors `MotorBoard.home` contract.

        Raises:
            HardwareError: Simulated no-response or firmware-error path.
        """
        resp = self.exchange_command('HOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.home() -> {resp}')
        if resp is None:
            raise HardwareError('home(): no response from motor board (timeout or disconnect)')
        if 'XYZ home complete' in resp:
            self.initial_homing_complete = True
            return True
        # Match real MotorBoard partial-home semantics so tests cover both.
        if ('not present' in resp) and ('X' in resp or 'Y' in resp):
            self.initial_homing_complete = True
            return True
        raise HardwareError(f'home(): firmware error: {resp}')

    def has_homed(self) -> bool:
        """Whether the simulated board has completed its initial XYZ home.

        Returns:
            bool: True if ``home()`` previously succeeded.
        """
        return self.initial_homing_complete

    def xycenter(self) -> None:
        """Move the simulated XY stage to centre (sends CENTER)."""
        self.exchange_command('CENTER')

    def thome(self) -> bool:
        """Simulated turret home. Mirrors `MotorBoard.thome` contract.

        Raises:
            HardwareError: Simulated no-response or firmware-error path.
        """
        resp = self.exchange_command('THOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.thome() -> {resp}')
        if resp is None:
            raise HardwareError('thome(): no response from motor board (timeout or disconnect)')
        if 'T home successful' in resp:
            self.initial_t_homing_complete = True
            return True
        if 'not present' in resp.lower():
            return True
        raise HardwareError(f'thome(): firmware error: {resp}')

    def has_turret(self) -> bool:
        """Whether the simulated board has a turret installed.

        Returns:
            bool: True when the model name ends in ``T``.
        """
        return self._has_turret

    def has_thomed(self) -> bool:
        """Whether the simulated board has completed turret homing.

        Returns:
            bool: True if either XYZ or T homing has completed.
        """
        return self.initial_homing_complete or self.initial_t_homing_complete

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move(self, axis: str, steps: int) -> None:
        """Move an axis to an absolute microstep position.

        Mirrors the production ``MotorBoard.move`` contract, including
        the raise on an unanswered target write -- without it a test that
        injects a dead board would watch the simulator report a move that
        never happened, which is the exact defect the production raise
        exists to surface.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            steps: Target absolute microstep position. Negative values
                are wrapped to two's-complement so the simulated firmware
                accepts them as unsigned 32-bit integers.

        Raises:
            HardwareError: Simulated no-response to the target write.
        """
        if steps < 0:
            steps += 0x100000000
        if self.exchange_command(f'TARGET_W{axis}{steps}') is None:
            raise HardwareError(
                f'move({axis}, {steps}): no response to the target write '
                f'(timeout or disconnect); the move did not happen'
            )

    def target_pos(self, axis: str) -> float | int:
        """Get the target position of an axis in user units.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            float | int: Microns for X/Y/Z, 1-based position for T, 0
                on read failure or unknown axis.
        """
        try:
            response = self.exchange_command(f'TARGET_R{axis}')
            position = int(response)
        except Exception:
            position = 0

        if axis == 'Z':
            return self.z_ustep2um(position)
        elif axis in ('X', 'Y'):
            return self.xy_ustep2um(position)
        elif axis == 'T':
            return self.t_ustep2pos(position)
        return 0

    def current_pos(self, axis: str) -> float | int:
        """Get the current position of an axis in user units.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            float | int: Microns for X/Y/Z, 1-based position for T, 0
                on read failure or unknown axis.
        """
        try:
            response = self.exchange_command(f'ACTUAL_R{axis}')
            position = int(response)
        except Exception:
            position = 0

        if axis == 'Z':
            return self.z_ustep2um(position)
        elif axis in ('X', 'Y'):
            return self.xy_ustep2um(position)
        elif axis == 'T':
            return self.t_ustep2pos(position)
        return 0

    def move_abs_pos(
        self, axis: str, pos: float, overshoot_enabled: bool = True, ignore_limits: bool = False
    ) -> None:
        """Move an axis to an absolute position in user units.

        Mirrors the production ``MotorBoard.move_abs_pos`` contract,
        including Z backlash overshoot when ``overshoot_enabled`` is True.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            pos: Target absolute position. Microns for X/Y/Z, 1-based
                position for T.
            overshoot_enabled: When True, apply Z backlash compensation
                if the target is sufficiently below the current position.
            ignore_limits: When True, skip the configured min/max
                clamping.

        Raises:
            Exception: ``axis`` is not in ``axes_config``.
        """
        if axis not in self.axes_config:
            raise Exception(f'Unsupported axis ({axis})')

        axis_config = self.axes_config[axis]
        if 'limits' in axis_config and not ignore_limits:
            limits = axis_config['limits']
            pos = max(pos, limits['min'])
            pos = min(pos, limits['max'])

        steps = axis_config['move_func'](pos)

        if overshoot_enabled and axis == 'Z':
            current = self.current_pos('Z')
            if current > pos and pos > (self.backlash + 50):
                self.overshoot = True
                overshoot = self.z_um2ustep(pos - self.backlash)
                overshoot = max(1, overshoot)
                self.move(axis, overshoot)
                while not self.target_status('Z'):
                    time.sleep(0.001)
                self.overshoot = False

        self.move(axis, steps)

    def move_rel_pos(self, axis: str, um: float, overshoot_enabled: bool = False) -> None:
        """Move an axis by a relative offset in user units.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').
            um: Offset to apply. Microns for X/Y/Z, position-count
                offset for T.
            overshoot_enabled: When True, apply Z backlash compensation
                during the underlying absolute move.
        """
        pos = self.target_pos(axis)
        self.move_abs_pos(axis, pos + um, overshoot_enabled=overshoot_enabled)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def home_status(self, axis: str) -> bool:
        """Return True if the axis is at the home position.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            bool: True when the simulated STATUS_R bit is set; False on
                read failure (does not raise, unlike production).
        """
        try:
            data = int(self.exchange_command(f'STATUS_R{axis}'))
            bits = format(data, 'b').zfill(32)
            return bits[31] == '1'
        except Exception:
            return False

    def target_status(self, axis: str) -> bool:
        """Return True if the axis has reached its target position.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            bool: True when current == target; False on read failure
                (does not raise, unlike production).
        """
        try:
            data = int(self.exchange_command(f'STATUS_R{axis}'))
            bits = format(data, 'b').zfill(32)
            return bits[22] == '1'
        except Exception:
            return False

    def reference_status(self, axis: str) -> int:
        """Read the raw STATUS register for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int: 32-bit register value, or 0 on read failure (does not
                raise, unlike production).
        """
        try:
            return int(self.exchange_command(f'STATUS_R{axis}'))
        except Exception:
            return 0

    def limit_switch_status(self, axis: str) -> tuple[int, int]:
        """Read the left + right limit switch state for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            tuple[int, int]: ``(left, right)`` where each value is 1
                (engaged), 0 (clear), or -1 on read failure.
        """
        try:
            resp = self.reference_status(axis)
            left = 1 if (resp & (1 << 0)) else 0
            right = 1 if (resp & (1 << 1)) else 0
        except Exception:
            left, right = -1, -1
        return left, right

    # ------------------------------------------------------------------
    # Acceleration (stubs)
    # ------------------------------------------------------------------
    def acceleration_limit(self, axis: str, parameter: str) -> int:
        """Stub: return the simulated firmware's fixed AMAX/DMAX value.

        Args:
            axis: Axis letter ('X' or 'Y').
            parameter: ``'acceleration'`` or ``'deceleration'``.

        Returns:
            int: Always 30000 (matches simulated AMAX/DMAX response).
        """
        return 30000

    def acceleration_limits(self) -> dict:
        """Stub: return acceleration + deceleration limits for X / Y.

        Returns:
            dict: ``{axis: {parameter: 30000}}`` for X and Y.
        """
        return {
            'X': {'acceleration': 30000, 'deceleration': 30000},
            'Y': {'acceleration': 30000, 'deceleration': 30000},
        }

    def set_acceleration_limit(self, axis: str, parameter: str, val_pct: int) -> None:
        """Stub: log the requested acceleration limit (no state change).

        Args:
            axis: Axis letter ('X' or 'Y').
            parameter: ``'acceleration'`` or ``'deceleration'``.
            val_pct: Percentage of the maximum (1-100, inclusive).
        """
        logger.info(f'[XYZ Sim   ] set_acceleration_limit({axis}, {parameter}, {val_pct}%)')

    def set_acceleration_limits(self, val_pct: int) -> None:
        """Stub: log the requested global acceleration limit.

        Args:
            val_pct: Percentage of the maximum (1-100, inclusive).
        """
        logger.info(f'[XYZ Sim   ] set_acceleration_limits({val_pct}%)')

    # ------------------------------------------------------------------
    # SPI (stubs)
    # ------------------------------------------------------------------
    def spi_read(self, axis: str, addr: int) -> str:
        """Stub: return a fixed acknowledgement for SPI reads.

        Args:
            axis: Motor axis ('X', 'Y', 'Z', 'T').
            addr: SPI register address.

        Returns:
            str: Always ``'SPI OK'``.
        """
        return 'SPI OK'

    def spi_write(self, axis: str, addr: int, payload: str) -> str:
        """Stub: return a fixed acknowledgement for SPI writes.

        Args:
            axis: Motor axis ('X', 'Y', 'Z', 'T').
            addr: SPI register address.
            payload: Value to write.

        Returns:
            str: Always ``'SPI OK'``.
        """
        return 'SPI OK'

    def set_precision_mode(self, axis: str, enabled: bool) -> None:
        """Stub: precision mode is a no-op in the simulator.

        Args:
            axis: Axis letter (accepted but unused).
            enabled: Precision-mode flag (accepted but unused).
        """
        pass  # No-op for simulator

    # ------------------------------------------------------------------
    # Firmware (stubs)
    # ------------------------------------------------------------------
    def check_firmware(self) -> None:
        """Stub: firmware check is a no-op in the simulator."""
        pass

    def update_firmware(self) -> bool:
        """Stub: firmware update always succeeds in the simulator.

        Returns:
            bool: Always True.
        """
        return True

    def get_firmware_URL(self, owner: str, repo: str, path: str) -> str:
        """Stub: simulator does not fetch firmware from GitHub.

        Args:
            owner: GitHub repository owner (unused).
            repo: GitHub repository name (unused).
            path: Relative path inside the repo (unused).

        Returns:
            str: Always an empty string.
        """
        return ''

    def get_latest_firmware(self, firmware_url: str, auth_token: str) -> dict:
        """Stub: no-op firmware fetch in the simulator.

        Args:
            firmware_url: URL to the firmware blob (unused).
            auth_token: GitHub auth token (unused).

        Returns:
            dict: Always an empty dict.
        """
        return {}

    def firmware_is_up_to_date(self) -> bool:
        """Stub: simulator firmware is always up to date.

        Returns:
            bool: Always True.
        """
        return True

    def get_current_firmware(self) -> str:
        """Return a fixed simulated firmware identification string.

        Returns:
            str: ``'Etaluma Motor Controller <model> Firmware: SIMULATED'``.
        """
        return f'Etaluma Motor Controller {self._fullinfo["model"]} Firmware: SIMULATED'

    def get_axes_config(self) -> dict:
        """Return the per-axis config (limits + unit-conversion func).

        Returns:
            dict: Axis-letter-keyed configuration dict.
        """
        return self.axes_config

    def get_axis_limits(self, axis: str) -> dict | None:
        """Return the configured min/max travel limits for an axis.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            dict: ``{'min': float, 'max': float}`` in axis user units,
                or ``None`` if the axis has no configured limits.

        Raises:
            Exception: ``axis`` is not a supported axis at all.
        """
        if axis not in self.axes_config:
            raise Exception(f'Unsupported axis ({axis})')
        if 'limits' not in self.axes_config[axis]:
            return None
        return self.axes_config[axis]['limits']

    # ------------------------------------------------------------------
    # New MotorBoard methods (2026-03-13)
    # ------------------------------------------------------------------
    def detect_present_axes(self) -> list:
        """Return the list of axes present on the simulated board.

        Returns:
            list: Axis letters present (e.g. ``['X', 'Y', 'Z', 'T']``).
        """
        axes = ['Z']  # Z always present
        if self._fullinfo.get('model', '').startswith('LS85'):
            axes = ['X', 'Y', 'Z']
        model = self._fullinfo.get('model', '')
        if model.endswith('T'):
            axes.append('T')
        return axes

    def current_pos_steps(self, axis: str) -> int:
        """Get current position in raw microsteps.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int: Microstep position (0 if axis is unknown).
        """
        with self.thread_lock:
            return self._actual.get(axis, 0)

    def target_pos_steps(self, axis: str) -> int:
        """Get target position in raw microsteps.

        Args:
            axis: Axis letter ('X', 'Y', 'Z', 'T').

        Returns:
            int: Microstep target (0 if axis is unknown).
        """
        with self.thread_lock:
            return self._target.get(axis, 0)

    # ------------------------------------------------------------------
    # Diagnostic commands (match MotorBoard API surface)
    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        """Simulated CONFIG -- returns motorconfig data.

        Returns:
            dict: Fixed-shape dict with ``Serial Number`` + ``Axis Present``.
        """
        return {
            'Serial Number': self._fullinfo.get('serial_number', 'SIM-0000'),
            'Axis Present': {'X': 1, 'Y': 1, 'Z': 1, 'T': 0},
        }

    def get_drvstat(self, axis: str | None = None) -> list:
        """Simulated DRVSTAT -- returns fake driver status.

        Args:
            axis: Optional single axis to query. If None, returns all four.

        Returns:
            list: One dict per axis with ``raw``, ``SG``, ``CS`` keys.
        """
        axes = [axis] if axis else ['X', 'Y', 'Z', 'T']
        return [
            {
                'axis': a,
                'raw': '0x00000000',
                'SG': 0,
                'CS': 0,
                'raw_line': f'{a}: raw=0x00000000 SG=0 CS=0',
            }
            for a in axes
        ]

    def get_motordetect(self) -> list:
        """Simulated MOTORDETECT.

        Returns:
            list: One dict per axis reporting detected + configured True.
        """
        return [
            {
                'axis': a,
                'detected': True,
                'configured': True,
                'raw_line': f'{a}: detected=True configured=True',
            }
            for a in ['X', 'Y', 'Z', 'T']
        ]

    def get_current(self) -> list:
        """Simulated CURRENT.

        Returns:
            list: One dict per axis with ``CS_ACTUAL``, ``IRUN``,
                ``IHOLD``, ``SG_RESULT`` keys.
        """
        return [
            {
                'axis': a,
                'CS_ACTUAL': 0,
                'IRUN': 10,
                'IHOLD': 3,
                'SG_RESULT': 0,
                'raw_line': f'{a}: CS_ACTUAL=0 IRUN=10 IHOLD=3 SG_RESULT=0',
            }
            for a in ['X', 'Y', 'Z', 'T']
        ]

    def read_drv_status(self, axis: str) -> int | None:
        """Simulated TMC5072 DRV_STATUS register -- returns 0 (no fault flags)."""
        axis = axis.upper()
        if axis not in ('X', 'Y', 'Z', 'T'):
            raise ValueError(f'Invalid axis: {axis!r}')
        return 0

    def read_fanspeed(self) -> int | None:
        """Simulated fan tachometer -- returns nominal RPM."""
        return 1200

    def set_fan_duty(self, duty_pct: int) -> bool:
        """Simulated fan PWM duty -- accepts any valid 0..100 setting."""
        if not 0 <= duty_pct <= 100:
            raise ValueError(f'Fan duty must be 0..100, got {duty_pct}')
        return True

    def wait_for_position(self, axis: str, timeout: float = 5.0) -> bool:
        """Simulated wait -- position is always reached instantly.

        Args:
            axis: Axis letter (accepted but unused).
            timeout: Maximum wait time in seconds (accepted but unused).

        Returns:
            bool: Always True.
        """
        return True

    def read_status(self, axis: str) -> int:
        """Simulated STATUS register -- bit 9 (position_reached) set.

        Args:
            axis: Axis letter (accepted but unused).

        Returns:
            int: Always ``0x200``.
        """
        return 0x200  # bit 9 = position_reached

    def detect_firmware_version(self) -> None:
        """No-op for simulator -- version is set at construction."""
        pass

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
