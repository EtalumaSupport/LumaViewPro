# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated Motor Board — drop-in replacement for MotorBoard.

No serial hardware required. Tracks axis positions, simulates homing
and movement, and supports configurable delays.

Timing modes:
  'fast'      — instant movement, zero delays (for tests)
  'realistic' — serial delays and speed-limited movement matching real hardware

Failure injection (for testing error recovery):
  fail_after=N      — disconnect after N commands (simulates USB cable pull)
  fail_on={'ZHOME'} — return None for specific commands (simulates timeout)
"""

import pathlib
import threading
import time
from lvp_logger import logger
from modules.motorconfig import MotorConfig


class SimulatedMotorBoard:

    # Axis speeds in usteps/sec (realistic values for Etaluma hardware)
    AXIS_SPEEDS = {
        'X': 20157 * 50,   # ~50 mm/s
        'Y': 20157 * 50,   # ~50 mm/s
        'Z': 170667 * 5,   # ~5 mm/s
        'T': 80000,         # ~90 deg/s
    }

    # Homing durations in seconds (realistic)
    HOMING_DURATIONS = {
        'XYZ': 3.0,
        'Z': 1.5,
        'T': 1.0,
    }

    # Timing presets
    TIMING_FAST = {
        'cmd_delay': 0.0,
        'move_delay': 0.0,
        'simulate_move_duration': False,
    }
    TIMING_REALISTIC = {
        'cmd_delay': 0.003,       # ~3ms serial round-trip
        'move_delay': 0.0,        # homing uses HOMING_DURATIONS instead
        'simulate_move_duration': True,
    }

    def __init__(self, model: str = 'LS850', serial_number: str = 'SIM-001',
                 move_delay: float = 0.0, cmd_delay: float = 0.0,
                 timing: str = 'fast', firmware_version: str = '2.0.1',
                 protocol_version: str = 'legacy',  # v3.0 STUB: 'legacy' or 'v3'
                 motorconfig_defaults_file: pathlib.Path | None = None,
                 fail_after: int | None = None,
                 fail_on: set | None = None,
                 **kwargs):
        logger.info('[XYZ Sim   ] SimulatedMotorBoard.__init__()')

        # Failure injection
        self._fail_after = fail_after          # disconnect after N commands
        self._fail_on = fail_on or set()       # return None for these commands
        self._cmd_count = 0

        # Load hardware config (same defaults as real MotorBoard)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path("data/motorconfig_defaults.json")
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

        # Move completion times (for realistic timing)
        self._move_end_time = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'T': 0.0}

        self.axes_config = {
            'Z': {
                'limits': {'min': 0., 'max': self.motorconfig.travel_limit_um('Z')},
                'move_func': self.z_um2ustep
            },
            'X': {
                'limits': {'min': 0., 'max': self.motorconfig.travel_limit_um('X')},
                'move_func': self.xy_um2ustep
            },
            'Y': {
                'limits': {'min': 0., 'max': self.motorconfig.travel_limit_um('Y')},
                'move_func': self.xy_um2ustep
            },
            'T': {
                'move_func': self.t_pos2ustep
            }
        }

    def set_timing_mode(self, mode: str):
        """Switch timing mode: 'fast' or 'realistic'."""
        if mode == 'realistic':
            preset = self.TIMING_REALISTIC
        elif mode == 'fast':
            preset = self.TIMING_FAST
        else:
            raise ValueError(f"Unknown timing mode: {mode!r}. Use 'fast' or 'realistic'.")
        self._cmd_delay = preset['cmd_delay']
        self._move_delay = preset['move_delay']
        self._simulate_move_duration = preset['simulate_move_duration']
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
        with self.thread_lock:
            self.driver = True
            self._connect_fails = 0
            logger.info('[XYZ Sim   ] SimulatedMotorBoard.connect()')

    def disconnect(self):
        with self.thread_lock:
            self.driver = None
            logger.info('[XYZ Sim   ] SimulatedMotorBoard.disconnect()')

    def is_connected(self) -> bool:
        return self.driver is not None

    def _close_driver(self):
        self.driver = None

    # ------------------------------------------------------------------
    # Serial simulation
    # ------------------------------------------------------------------
    def _sim_delay(self):
        if self._cmd_delay > 0:
            time.sleep(self._cmd_delay)

    def exchange_command(self, command, response_numlines=1, timeout=None):
        with self.thread_lock:
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
                logger.warning(f'[XYZ Sim   ] INJECTED FAILURE: disconnect after {self._fail_after} commands')
                self.driver = None
                self.found = False
                return None

            # Failure injection: fail on specific commands
            cmd_word = command.strip().split()[0] if command else ''
            if cmd_word in self._fail_on:
                logger.warning(f'[XYZ Sim   ] INJECTED FAILURE: timeout on {cmd_word}')
                return None

            self._sim_delay()
            response = self._handle_command(command)
            logger.debug(f'[XYZ Sim   ] exchange_command({command}) -> {response}')
            if response_numlines == 1:
                return response
            return [response]

    def exchange_multiline(self, command, timeout=60, end_markers=None):
        """Simulated multi-line response."""
        return self.exchange_command(command)

    def _handle_command(self, command):
        cmd = command.strip()

        if cmd == 'INFO':
            return f"Etaluma Motor Controller {self._fullinfo['model']} Firmware: SIMULATED"

        if cmd == 'FULLINFO':
            model = self._fullinfo['model']
            sn = self._fullinfo['serial_number']
            return f"Model: {model} Serial: {sn} Firmware: SIMULATED"

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
            old_target = self._actual[axis]
            self._target[axis] = value

            if self._simulate_move_duration:
                distance = abs(value - old_target)
                speed = self.AXIS_SPEEDS.get(axis, self.AXIS_SPEEDS['X'])
                duration = distance / speed if speed > 0 else 0
                self._move_end_time[axis] = time.monotonic() + duration
            else:
                self._actual[axis] = value  # instant move
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

        return f'ERROR: unknown command {cmd}'

    def _update_actual(self, axis):
        """Update actual position based on elapsed time (realistic mode only)."""
        if not self._simulate_move_duration:
            return
        now = time.monotonic()
        end = self._move_end_time.get(axis, 0.0)
        if now >= end:
            self._actual[axis] = self._target[axis]

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
            status |= (1 << 0)
        # Bit 9: position_reached (target == actual)
        if self._actual.get(axis, 0) == self._target.get(axis, 0):
            status |= (1 << 9)
        return status

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------
    def infomation(self):
        self.exchange_command('INFO')

    def fullinfo(self):
        info = self.exchange_command('FULLINFO')
        info_parts = info.split()
        model = info_parts[info_parts.index("Model:") + 1]
        if model.endswith('T'):
            self._has_turret = True
        serial_number = info_parts[info_parts.index("Serial:") + 1]
        return {"model": model, "serial_number": serial_number}

    def get_microscope_model(self):
        return self._fullinfo['model']

    # ------------------------------------------------------------------
    # Conversion functions (identical to real MotorBoard)
    # ------------------------------------------------------------------
    def z_ustep2um(self, ustep):
        return ustep * 1000 / self.motorconfig.usteps_per_mm('Z')

    def z_um2ustep(self, um):
        return int(self.motorconfig.usteps_per_mm('Z') * um / 1000)

    def xy_ustep2um(self, ustep):
        return ustep * 1000 / self.motorconfig.usteps_per_mm('X')

    def xy_um2ustep(self, um):
        return int(self.motorconfig.usteps_per_mm('X') * um / 1000)

    def t_ustep2deg(self, ustep):
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        return 90.0 / usteps_per_90deg * ustep

    def t_ustep2pos(self, ustep):
        return int(self.t_ustep2deg(ustep) / 90) + 1

    def t_deg2ustep(self, degrees):
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        return int(degrees * usteps_per_90deg / 90.0)

    def t_pos2ustep(self, position):
        usteps = self.motorconfig.turret_position_usteps(position)
        if usteps == 0 and position > 1:
            return self.t_deg2ustep(90 * (position - 1))
        return usteps

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------
    def zhome(self):
        resp = self.exchange_command('ZHOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.zhome() -> {resp}')
        if resp is None:
            return False
        return 'successful' in resp.lower() or 'complete' in resp.lower()

    def xyhome(self):
        resp = self.exchange_command('HOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.xyhome() -> {resp}')
        if resp is None:
            return False
        if 'XYZ home complete' in resp:
            self.initial_homing_complete = True
            return True
        return False

    def has_xyhomed(self):
        return self.initial_homing_complete

    def xycenter(self):
        self.exchange_command('CENTER')

    def thome(self):
        resp = self.exchange_command('THOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.thome() -> {resp}')
        if resp is None:
            return False
        if 'T home successful' in resp:
            self.initial_t_homing_complete = True
            return True
        if 'not present' in resp.lower():
            return True
        return False

    def has_turret(self) -> bool:
        return self._has_turret

    def has_thomed(self):
        return self.initial_homing_complete or self.initial_t_homing_complete

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move(self, axis, steps):
        if steps < 0:
            steps += 0x100000000
        self.exchange_command(f'TARGET_W{axis}{steps}')

    def target_pos(self, axis):
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

    def current_pos(self, axis):
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

    def move_abs_pos(self, axis, pos, overshoot_enabled: bool = True, ignore_limits: bool = False):
        if axis not in self.axes_config:
            raise Exception(f"Unsupported axis ({axis})")

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

    def move_rel_pos(self, axis, um, overshoot_enabled: bool = False):
        pos = self.target_pos(axis)
        self.move_abs_pos(axis, pos + um, overshoot_enabled=overshoot_enabled)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def home_status(self, axis):
        try:
            data = int(self.exchange_command(f'STATUS_R{axis}'))
            bits = format(data, 'b').zfill(32)
            return bits[31] == '1'
        except Exception:
            return False

    def target_status(self, axis):
        try:
            data = int(self.exchange_command(f'STATUS_R{axis}'))
            bits = format(data, 'b').zfill(32)
            return bits[22] == '1'
        except Exception:
            return False

    def reference_status(self, axis):
        try:
            return int(self.exchange_command(f'STATUS_R{axis}'))
        except Exception:
            return 0

    def limit_switch_status(self, axis):
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
        return 30000

    def acceleration_limits(self) -> dict:
        return {
            'X': {'acceleration': 30000, 'deceleration': 30000},
            'Y': {'acceleration': 30000, 'deceleration': 30000},
        }

    def set_acceleration_limit(self, axis: str, parameter: str, val_pct: int):
        logger.info(f'[XYZ Sim   ] set_acceleration_limit({axis}, {parameter}, {val_pct}%)')

    def set_acceleration_limits(self, val_pct):
        logger.info(f'[XYZ Sim   ] set_acceleration_limits({val_pct}%)')

    # ------------------------------------------------------------------
    # SPI (stubs)
    # ------------------------------------------------------------------
    def spi_read(self, axis: str, addr: int) -> str:
        return 'SPI OK'

    def spi_write(self, axis: str, addr: int, payload: str) -> str:
        return 'SPI OK'

    def set_precision_mode(self, axis: str, enabled: bool):
        pass  # No-op for simulator

    # ------------------------------------------------------------------
    # Firmware (stubs)
    # ------------------------------------------------------------------
    def check_firmware(self):
        pass

    def update_firmware(self):
        return True

    def get_firmware_URL(self, owner, repo, path):
        # Stub: no real URL construction needed in simulator
        return ''

    def get_latest_firmware(self, firmware_url, auth_token):
        return {}

    def firmware_is_up_to_date(self):
        return True

    def get_current_firmware(self):
        return f"Etaluma Motor Controller {self._fullinfo['model']} Firmware: SIMULATED"

    def get_axes_config(self):
        return self.axes_config

    def get_axis_limits(self, axis: str):
        if axis not in self.axes_config:
            raise Exception(f"Unsupported axis ({axis})")
        if 'limits' not in self.axes_config[axis]:
            raise Exception(f"Axis {axis} does not have defined limits")
        return self.axes_config[axis]['limits']

    # ------------------------------------------------------------------
    # New MotorBoard methods (2026-03-13)
    # ------------------------------------------------------------------
    def detect_present_axes(self):
        """Return list of axes present on this board."""
        axes = ['Z']  # Z always present
        if self._fullinfo.get('model', '').startswith('LS85'):
            axes = ['X', 'Y', 'Z']
        model = self._fullinfo.get('model', '')
        if model.endswith('T'):
            axes.append('T')
        return axes

    def current_pos_steps(self, axis):
        """Get current position in raw microsteps."""
        with self.thread_lock:
            return self._actual.get(axis, 0)

    def target_pos_steps(self, axis):
        """Get target position in raw microsteps."""
        with self.thread_lock:
            return self._target.get(axis, 0)

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
