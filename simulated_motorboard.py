"""
Simulated Motor Board — drop-in replacement for MotorBoard.

No serial hardware required. Tracks axis positions, simulates homing
and movement, and supports configurable delays.
"""

import threading
import time
from lvp_logger import logger


class SimulatedMotorBoard:

    # Conversion constants (same as real MotorBoard)
    Z_USTEP_PER_MM = 170667
    XY_USTEP_PER_MM = 20157
    T_USTEP_PER_DEG = 80000.0 / 90.0

    def __init__(self, model: str = 'LS720T', serial_number: str = 'SIM-001',
                 move_delay: float = 0.01, cmd_delay: float = 0.001, **kwargs):
        logger.info('[XYZ Sim   ] SimulatedMotorBoard.__init__()')

        self.found = True
        self.overshoot = False
        self.backlash = 25
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

        # Internal position state (in usteps)
        self._actual = {'X': 0, 'Y': 0, 'Z': 0, 'T': 0}
        self._target = {'X': 0, 'Y': 0, 'Z': 0, 'T': 0}
        self._homed = {'X': False, 'Y': False, 'Z': False, 'T': False}

        self.axes_config = {
            'Z': {
                'limits': {'min': 0., 'max': 14000.},
                'move_func': self.z_um2ustep
            },
            'X': {
                'limits': {'min': 0., 'max': 120000.},
                'move_func': self.xy_um2ustep
            },
            'Y': {
                'limits': {'min': 0., 'max': 80000.},
                'move_func': self.xy_um2ustep
            },
            'T': {
                'move_func': self.t_pos2ustep
            }
        }

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

    def exchange_command(self, command, response_numlines=1):
        with self.thread_lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return None
            if self.driver is None:
                return None

            self._sim_delay()
            response = self._handle_command(command)
            logger.debug(f'[XYZ Sim   ] exchange_command({command}) -> {response}')
            if response_numlines == 1:
                return response
            return [response]

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
            mid_x = self.xy_um2ustep(60000)
            mid_y = self.xy_um2ustep(40000)
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
            self._target[axis] = value
            self._actual[axis] = value  # instant move in simulation
            return str(value)

        # TARGET_R<axis>
        if cmd.startswith('TARGET_R'):
            axis = cmd[8]
            return str(self._target.get(axis, 0))

        # ACTUAL_R<axis>
        if cmd.startswith('ACTUAL_R'):
            axis = cmd[8]
            return str(self._actual.get(axis, 0))

        # STATUS_R<axis>
        if cmd.startswith('STATUS_R'):
            axis = cmd[8]
            return str(self._make_status(axis))

        # SPI<axis>0x<addr><payload>
        if cmd.startswith('SPI'):
            return 'SPI OK'

        # Acceleration limits
        if cmd.startswith('AMAX') or cmd.startswith('DMAX'):
            return '30000'

        return f'ERROR: unknown command {cmd}'

    def _do_home(self, *axes):
        if self._move_delay > 0:
            time.sleep(self._move_delay)
        for axis in axes:
            self._actual[axis] = 0
            self._target[axis] = 0
            self._homed[axis] = True

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
        return ustep * 1000 / self.Z_USTEP_PER_MM

    def z_um2ustep(self, um):
        return int(self.Z_USTEP_PER_MM * um / 1000)

    def xy_ustep2um(self, ustep):
        return ustep * 1000 / self.XY_USTEP_PER_MM

    def xy_um2ustep(self, um):
        return int(self.XY_USTEP_PER_MM * um / 1000)

    def t_ustep2deg(self, ustep):
        return 90.0 / 80000.0 * ustep

    def t_ustep2pos(self, ustep):
        return int(self.t_ustep2deg(ustep) / 90) + 1

    def t_deg2ustep(self, degrees):
        return int(degrees * 80000.0 / 90.0)

    def t_pos2ustep(self, position):
        return self.t_deg2ustep(90 * (position - 1))

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------
    def zhome(self):
        resp = self.exchange_command('ZHOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.zhome() -> {resp}')

    def xyhome(self):
        resp = self.exchange_command('HOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.xyhome() -> {resp}')
        if resp is not None and 'XYZ home complete' in resp:
            self.initial_homing_complete = True

    def has_xyhomed(self):
        return self.initial_homing_complete

    def xycenter(self):
        self.exchange_command('CENTER')

    def thome(self):
        resp = self.exchange_command('THOME')
        logger.info(f'[XYZ Sim   ] SimulatedMotorBoard.thome() -> {resp}')
        if resp is not None and 'T home successful' in resp:
            self.initial_t_homing_complete = True

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

    # ------------------------------------------------------------------
    # Firmware (stubs)
    # ------------------------------------------------------------------
    def check_firmware(self):
        pass

    def update_firmware(self):
        return True

    def get_firmware_URL(self, owner, repo, path):
        return f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

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
