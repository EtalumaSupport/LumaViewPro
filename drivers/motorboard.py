#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib
import threading
import time
import lvp_logger
from lvp_logger import logger

from drivers.serialboard import SerialBoard
from modules.exceptions import HardwareError
from modules.motorconfig import MotorConfig


class MotorBoard(SerialBoard):

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, motorconfig_defaults_file: pathlib.Path | None = None, **kwargs):
        self._state_lock = threading.Lock()
        self.overshoot = False
        self._has_turret = False
        self.initial_homing_complete = False
        self.initial_t_homing_complete = False
        self._fullinfo = None
        self._connect_fails = 0

        # Load hardware config (per-unit values from motorconfig.json, with defaults fallback)
        if motorconfig_defaults_file is None:
            motorconfig_defaults_file = pathlib.Path("data/motorconfig_defaults.json")
        self.motorconfig = MotorConfig(defaults_file=motorconfig_defaults_file)
        self.backlash = self.motorconfig.antibacklash_um('Z')

        super().__init__(vid=0x2E8A, pid=0x0005, label='[XYZ Class ]',
                         timeout=30, write_timeout=5)

        # Backward-compatible alias for lock name
        self.thread_lock = self._lock

        self.axes_config = {
            'Z': {
                'limits': {
                    'min': 0.,
                    'max': self.motorconfig.travel_limit_um('Z'),
                },
                'move_func': self.z_um2ustep
            },
            'X': {
                'limits': {
                    'min': 0.,
                    'max': self.motorconfig.travel_limit_um('X'),
                },
                'move_func': self.xy_um2ustep
            },
            'Y': {
                'limits': {
                    'min': 0.,
                    'max': self.motorconfig.travel_limit_um('Y'),
                },
                'move_func': self.xy_um2ustep
            },
            'T': {
                'move_func': self.t_pos2ustep
            }
        }

        try:
            self.connect()
        except Exception:
            logger.error('[XYZ Class ] Failed to connect to motor controller')
            raise

    def connect(self):
        """ Try to connect to the motor controller based on the known VID/PID"""
        # Note: _lock is an RLock (from SerialBoard), so re-entrant acquisition
        # by _open_serial, _reset_firmware, exchange_command etc. is safe.
        with self._lock:
            try:
                self._open_serial()

                # Motor-specific: close/open dance for port reset
                self.driver.close()
                self.driver.open()

                self._connect_fails = 0
                if lvp_logger.is_thread_paused():
                    lvp_logger.unpause_thread()

                self._reset_firmware()
                info = self.fullinfo()
                with self._state_lock:
                    self._fullinfo = info
                logger.info('[XYZ Class ] Connected to motor controller')
            except Exception as e:
                self._close_driver()
                self._connect_fails += 1
                if self._connect_fails >= 10:
                    logger.critical(f'[XYZ Class ] MotorBoard.connect() failed 10 times, pausing thread logs')
                    lvp_logger.pause_thread()
                logger.error(f'[XYZ Class ] MotorBoard.connect() failed: {e}')


    # v3.0 STUB: Motor command builders for JSON Lines protocol
    # When v3.0 is active, commands will use structured JSON format:
    #   {"cmd": "HOME", "axes": ["X", "Y", "Z"]}
    #   {"cmd": "MOVE", "axis": "Z", "target": 12345}
    #   {"cmd": "STATUS", "axis": "Z"}
    #   {"cmd": "SPI", "axis": "Z", "addr": "0x6A", "payload": "0x00"}
    # Currently all commands use the legacy text format.

    # Firmware 1-14-2023 commands include
    # 'QUIT'
    # 'INFO'
    # 'HOME'
    # 'ZHOME'
    # 'THOME'
    # 'ACTUAL_R'
    # 'ACTUAL_W'
    # 'TARGET_R'
    # 'TARGET_W'
    # 'STATUS_R'
    # 'SPI'

    #----------------------------------------------------------
    # Informational Functions
    #----------------------------------------------------------
    def fullinfo(self):
        info = self.exchange_command("FULLINFO")
        logger.info('[XYZ Class ] MotorBoard.fullinfo(): %s', info, extra={'force_error': True})
        if info is None:
            logger.error('[XYZ Class ] FULLINFO returned None — board disconnected?')
            return {"model": "unknown", "serial_number": "unknown"}
        try:
            parts = info.split()
            model = parts[parts.index("Model:") + 1]
            if model[-1] == "T":
                with self._state_lock:
                    self._has_turret = True
            serial_number = parts[parts.index("Serial:") + 1]
        except (ValueError, IndexError) as e:
            logger.error(f'[XYZ Class ] Failed to parse FULLINFO response: {info!r} ({e})')
            return {"model": "unknown", "serial_number": "unknown"}
        return {
            "model": model,
            "serial_number": serial_number
        }


    def get_microscope_model(self):
        with self._state_lock:
            info = self._fullinfo
        return info['model']

    def detect_present_axes(self):
        """Detect which axes are present on this board.

        Parses FULLINFO response for 'X present: True' etc.
        Returns list of axis letters, e.g. ['X', 'Y', 'Z', 'T'] or ['Z', 'T'].
        """
        resp = self.exchange_command('FULLINFO')
        if resp is None:
            return []
        axes = []
        for axis in ('X', 'Y', 'Z', 'T'):
            if f'{axis} present: True' in resp or f'{axis} present:True' in resp:
                axes.append(axis)
        return axes

    def current_pos_steps(self, axis):
        """Get current position in raw microsteps (no unit conversion).

        Returns int or None on failure.
        """
        try:
            response = self.exchange_command('ACTUAL_R' + axis)
            if response is None:
                return None
            return int(response)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] current_pos_steps({axis}) failed: {e}')
            return None

    def target_pos_steps(self, axis):
        """Get target position in raw microsteps (no unit conversion).

        Returns int or None on failure.
        """
        try:
            response = self.exchange_command('TARGET_R' + axis)
            if response is None:
                return None
            return int(response)
        except (ValueError, TypeError) as e:
            logger.warning(f'[XYZ Class ] target_pos_steps({axis}) failed: {e}')
            return None

    #----------------------------------------------------------
    # Acceleration control functions
    #----------------------------------------------------------

    # Get single acceleration limit for a specific axis and parameter
    def acceleration_limit(self, axis: str, parameter: str) -> int:
        if not self._acceleration_validate_inputs(axis=axis, parameter=parameter):
            return 0

        parameter_map = {
            'acceleration': 'A',
            'deceleration': 'D'
        }

        parameter_char = parameter_map[parameter]
        command = f"{parameter_char}MAX{axis}"
        DEFAULT_ACCELERATION_LIMIT = 30000
        using_default = False
        try:
            resp = self.exchange_command(command)

            # In case firmware doesn't support retrieving the acceleration limits
            if resp is None or resp.startswith("ERROR"):
                raise ValueError(f"Firmware returned ERROR for {command}")

            # Extra protection for now in case motorboard responds with a different string that doesnt start with ERROR
            if not resp.isdigit():
                raise ValueError(f"Non-numeric response for {command}: {resp}")

        except Exception:
            resp = DEFAULT_ACCELERATION_LIMIT
            using_default = True

        if using_default:
            logger.debug(f'[XYZ Class ] MotorBoard.acceleration_limit({command}): firmware does not support, using default {DEFAULT_ACCELERATION_LIMIT}')
        else:
            logger.info(f'[XYZ Class ] MotorBoard.acceleration_limit({command}): {resp}')

        return int(resp)


    def _acceleration_validate_inputs(self, axis: str, parameter: str):
        config = self._acceleration_supported_info()
        if axis not in config['axes']:
            raise NotImplementedError(f"Support for acceleration limit on axis {axis} not implemented")

        if parameter not in config['parameters']:
            raise NotImplementedError(f"Support for acceleration limit parameter {parameter} not implemented.")

        return True


    def _acceleration_supported_info(self):
        return {
            'axes': ('X','Y'),
            'parameters': ('acceleration', 'deceleration')
        }

    # Get all acceleration limits for all axes and parameters
    def acceleration_limits(self) -> dict[str, dict[str, int]]:
        limits = {}
        config = self._acceleration_supported_info()
        for axis in config['axes']:
            limits[axis] = {}
            for parameter in config['parameters']:
                limits[axis][parameter] = self.acceleration_limit(axis=axis, parameter=parameter)

        return limits


    # Sets the percentage acceleration/deceleration limit (of max) for a single axis/parameter
    def set_acceleration_limit(self, axis: str, parameter: str, val_pct: int):
        if not self._acceleration_validate_inputs(axis=axis, parameter=parameter):
            return

        if (val_pct < 1) or (val_pct > 100):
            raise ValueError(f"Acceleration limit of {val_pct}% is out of bounds. Must be between 1 and 100.")

        limit = self.acceleration_limit(axis=axis, parameter=parameter)
        setpoint = round(limit*(val_pct/100))

        SPI_ADDRS = {
            'X': {
                'acceleration': 0x26,
                'deceleration': 0x28,
            },
            'Y': {
                'acceleration': 0x46,
                'deceleration': 0x48,
            },
        }

        self.spi_write(
            axis=axis,
            addr=SPI_ADDRS[axis][parameter],
            payload=setpoint
        )
        logger.info(f"[XYZ Class ] MotorBoard.set_acceleration_limit({axis}, {parameter}, {val_pct}%)")


    # Sets the percentage acceleration/deceleration (of max) for all supported axes/parameters
    def set_acceleration_limits(self, val_pct):
        config = self._acceleration_supported_info()
        for axis in config['axes']:
            for parameter in config['parameters']:
                self.set_acceleration_limit(axis=axis, parameter=parameter, val_pct=val_pct)

    #----------------------------------------------------------
    # SPI-direct related functions
    #----------------------------------------------------------
    def spi_read(self, axis: str, addr: int) -> str:
        # Add a dummy payload of "00" to the end in order for the firmware to not error out on a read.
        # It is expecting a payload.
        command = f"SPI{axis}0x{addr:02x}00"
        resp = self.exchange_command(command)
        logger.debug(f"[XYZ Class ] MotorBoard.spi_read({axis}, 0x{addr:02x}): {command} -> {resp}")
        return resp


    def spi_write(self, axis: str, addr: int, payload: str) -> str:
        WRITE_OFFSET = 0x80
        write_addr = addr + WRITE_OFFSET
        command = f"SPI{axis}0x{write_addr:02x}{payload}"
        resp = self.exchange_command(command)
        logger.debug(f"[XYZ Class ] MotorBoard.spi_write({axis}, 0x{addr:02x}): {command} -> {resp}")
        return resp


    #----------------------------------------------------------
    # Precision mode — controls motor stop accuracy
    #----------------------------------------------------------

    # TMC5072 VSTOP register addresses per axis.
    # VSTOP sets the velocity threshold for declaring "stopped" —
    # lower = more accurate final position, slightly slower settle.
    _VSTOP_ADDR = {
        'X': 0x2B,  # VSTOP_M1 on XY chip
        'Y': 0x4B,  # VSTOP_M2 on XY chip
        'Z': 0x4B,  # VSTOP_M2 on ZT chip
        'T': 0x2B,  # VSTOP_M1 on ZT chip
    }
    _VSTOP_NORMAL = 1000    # factory default — fast but overshoots
    _VSTOP_PRECISION = 100  # accurate stop position

    def set_precision_mode(self, axis: str, enabled: bool):
        """Set motor precision mode for an axis.

        Precision mode uses a lower VSTOP threshold so the motor fully
        decelerates before reporting target reached. Use for autofocus
        fine passes and any measurement that needs accurate positioning.

        Normal mode uses a higher VSTOP for faster moves where overshoot
        is acceptable (coarse AF pass, user jogging, homing approach).

        Args:
            axis: Axis name ("X", "Y", "Z", "T").
            enabled: True for precise positioning, False for speed.
        """
        if axis not in self._VSTOP_ADDR:
            logger.warning(f'[XYZ Class ] set_precision_mode: invalid axis {axis}')
            return
        vstop = self._VSTOP_PRECISION if enabled else self._VSTOP_NORMAL
        addr = self._VSTOP_ADDR[axis]
        self.spi_write(axis, addr, str(vstop))
        logger.info(f'[XYZ Class ] {axis} precision_mode={enabled} (VSTOP={vstop})')

    #----------------------------------------------------------
    # Z (Focus) Functions
    # Stock actuator = 0.30 mm pitch.  (1 rev/0.30 mm) x (200 steps/rev) x (256 usteps/step) = 170667 ustep/mm
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        usteps_per_mm = self.motorconfig.usteps_per_mm('Z')
        um = (ustep * 1000 / usteps_per_mm)
        return um

    def z_um2ustep(self, um):
        usteps_per_mm = self.motorconfig.usteps_per_mm('Z')
        ustep = int((usteps_per_mm * um) / 1000)
        return ustep

    def zhome(self):
        """Home the objective. Returns True on success, False on failure."""
        resp = self.exchange_command('ZHOME')
        logger.info(f'[XYZ Class ] MotorBoard.zhome() -> {resp}')
        if resp is None:
            logger.error('[XYZ Class ] zhome(): no response (timeout or disconnect)')
            return False
        success = 'successful' in resp.lower() or 'complete' in resp.lower()
        if not success:
            logger.error(f'[XYZ Class ] zhome() failed: {resp}')
        return success

    #----------------------------------------------------------
    # XY Stage Functions
    # Stock actuator = 2.54mm pitch.  (1 rev/2.540 mm) x (200 steps/rev) x (256 usteps/step) = 20157 ustep/mm
    #----------------------------------------------------------

    def xy_ustep2um(self, ustep):
        usteps_per_mm = self.motorconfig.usteps_per_mm('X')
        um = (ustep * 1000 / usteps_per_mm)
        return um

    def xy_um2ustep(self, um):
        usteps_per_mm = self.motorconfig.usteps_per_mm('X')
        ustep = int((usteps_per_mm * um) / 1000)
        return ustep

    def xyhome(self):
        """Home the stage (also homes objective first). Returns True on success."""
        resp = self.exchange_command('HOME')
        logger.info(f'[XYZ Class ] MotorBoard.xyhome() -> {resp}', extra={'force_error': True})
        if resp is None:
            logger.error('[XYZ Class ] xyhome(): no response (timeout or disconnect)')
            return False
        if 'XYZ home complete' in resp:
            with self._state_lock:
                self.initial_homing_complete = True
            return True
        logger.error(f'[XYZ Class ] xyhome() failed: {resp}')
        return False

    def has_xyhomed(self):
        with self._state_lock:
            return self.initial_homing_complete

    def xycenter(self):
        """ Home the stage which also homes the objective first """
        logger.info('[XYZ Class ] MotorBoard.xycenter()')
        self.exchange_command('CENTER')

    #----------------------------------------------------------
    # T (Turret) Functions
    #----------------------------------------------------------
    def t_ustep2deg(self, ustep):
        # T config value is usteps per 90 degrees (one turret position)
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        degrees = 90.0 / usteps_per_90deg * ustep
        return degrees

    def t_ustep2pos(self, ustep):
        return int(self.t_ustep2deg(ustep=ustep)/90)+1

    def t_deg2ustep(self, degrees):
        usteps_per_90deg = self.motorconfig.usteps_per_mm('T')
        ustep = int(degrees * usteps_per_90deg / 90.0)
        return ustep

    def t_pos2ustep(self, position):
        """Convert turret position (1-based) to microsteps.
        Uses motorconfig turret positions if available, falls back to 90-degree spacing."""
        usteps = self.motorconfig.turret_position_usteps(position)
        if usteps == 0 and position > 1:
            # Fallback: evenly-spaced positions
            return self.t_deg2ustep(degrees=90*(position-1))
        return usteps

    def thome(self):
        """Home the turret. Returns True on success."""
        resp = self.exchange_command('THOME')
        logger.info(f'[XYZ Class ] MotorBoard.thome() -> {resp}', extra={'force_error': True})
        if resp is None:
            logger.error('[XYZ Class ] thome(): no response (timeout or disconnect)')
            return False
        if 'T home successful' in resp:
            with self._state_lock:
                self.initial_t_homing_complete = True
            return True
        # "T not present" is not a failure — board just doesn't have a turret
        if 'not present' in resp.lower():
            return True
        logger.error(f'[XYZ Class ] thome() failed: {resp}')
        return False

    def has_turret(self) -> bool:
        with self._state_lock:
            return self._has_turret

    def has_thomed(self):
        # Note: When the motorboard firmware performs an XYZ homing, it also
        # does a T homing if a turret is present
        with self._state_lock:
            return self.initial_homing_complete or self.initial_t_homing_complete

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------

    def move(self, axis, steps):
        """ Move the axis to an absolute position (in usteps)
        compared to Home """
        # logger.info('move', axis, steps)

        # logger.info('def move(self, axis, steps)', axis, steps)
        if steps < 0:
            steps += 0x100000000 # twos compliment
        #print(f"Axis: {axis} steps: {steps}")
        self.exchange_command('TARGET_W' + axis + str(steps))

        # target_pos = int(self.exchange_command('TARGET_R' + axis))
        # desired_target = steps

        # while int(target_pos) != desired_target:
        #     self.exchange_command('TARGET_W' + axis + str(steps))
        #     time.sleep(0.005)
        #     target_pos = int(self.exchange_command('TARGET_R' + axis))

    # Get target position
    def target_pos(self, axis):
        """ Get the target position of an axis"""

        try:
            response = self.exchange_command('TARGET_R' + axis)
            position = int(response)
        except Exception as e:
            logger.warning(f'[XYZ Class ] target_pos({axis}) failed: {e}')
            position = 0

        if axis == 'Z':
            um = self.z_ustep2um(position)
            return um
        elif (axis == 'X') or (axis == 'Y'):
            um = self.xy_ustep2um(position)
            return um
        elif axis == 'T':
            return self.t_ustep2pos(position)
        else:
            return 0

    # Get current position (in um or position for Turret)
    def current_pos(self, axis):
        """Get current position (in um) of axis"""

        try:
            response = self.exchange_command('ACTUAL_R' + axis)
            position = int(response)
        except Exception as e:
            logger.warning(f'[XYZ Class ] current_pos({axis}) failed: {e}')
            position = 0

        if axis == 'Z':
            um = self.z_ustep2um(position)
            return um
        elif (axis == 'X') or (axis == 'Y'):
            um = self.xy_ustep2um(position)
            return um
        elif axis == 'T':
            return self.t_ustep2pos(position)
        else:
            return 0

    # Move to absolute position (in um or degrees for Turret)
    def move_abs_pos(self, axis, pos, overshoot_enabled: bool=True, ignore_limits: bool=False):
        """ Move to absolute position (in um) of axis"""
        # logger.info('move_abs_pos', axis, pos)
        AXES_CONFIG = self.axes_config

        if axis not in AXES_CONFIG:
            raise HardwareError(f"Unsupported axis ({axis})")

        axis_config = AXES_CONFIG[axis]

        if ('limits' in axis_config) and (not ignore_limits):
            axis_limits = axis_config['limits']
            pos = max(pos, axis_limits['min'])
            pos = min(pos, axis_limits['max'])

        steps = axis_config['move_func'](pos)

        if overshoot_enabled and (axis=='Z'): # perform overshoot to always come from one direction
            # get current position
            current = self.current_pos('Z')

            # if the current position is above the new target position
            # and 50um above the height of the backlash
            if (current > pos) and (pos > (self.backlash+50)):
                # In process of overshoot
                with self._state_lock:
                    self.overshoot = True
                # First overshoot downwards
                overshoot = self.z_um2ustep(pos-self.backlash) # target minus backlash
                overshoot = max(1, overshoot)
                #self.SPI_write (self.chip_pin[axis], self.write_target[axis], overshoot)
                self.move(axis, overshoot)
                while not self.target_status('Z'):
                    time.sleep(0.001)
                # complete overshoot
                with self._state_lock:
                    self.overshoot = False

        self.move(axis, steps)

    # Move by relative distance (in um or degrees for Turret)
    def move_rel_pos(self, axis, um, overshoot_enabled: bool = False):
        """ Move by relative distance (in um for X, Y, Z or position for T) of axis """

        # Read target position in um
        pos = self.target_pos(axis)
        self.move_abs_pos(axis, pos+um, overshoot_enabled=overshoot_enabled)

    #----------------------------------------------------------
    # Ramp and Reference Switch Status Register
    #----------------------------------------------------------

    # return True if current and target position are at home.
    def home_status(self, axis):
        """ Return True if axis is in home position"""

        # logger.info('[XYZ Class ] MotorBoard.home_status('+axis+')')
        try:
            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            return bits[31] == '1'
        except Exception:
            logger.error('[XYZ Class ] MotorBoard.home_status('+axis+') inactive')
            raise

    # return True if current position and target position are the same
    def target_status(self, axis):
        """ Return True if axis is at target position"""

        # logger.info('[XYZ Class ] MotorBoard.target_status('+axis+')')
        try:
            payload = 'STATUS_R' + axis
            response = self.exchange_command(payload)
            if response is None:
                raise ValueError("STATUS_R returned None")
            data = int( response )
            bits = format(data, 'b').zfill(32)

            return bits[22] == '1'

        except Exception:
            logger.error('[XYZ Class ] MotorBoard.get_limit_status('+axis+') inactive')
            raise


    # Get all reference status register bits as 32 character string (32-> 0)
    def reference_status(self, axis):
        """ Get all reference status register bits as 32 character string (32-> 0) """
        try:

            data = int( self.exchange_command('STATUS_R' + axis) )
            # bits = format(data, 'b').zfill(32)

            # data is an integer that represents 4 bytes, or 32 bits,
            # largest bit first
            '''
            bit: 33222222222211111111110000000000
            bit: 10987654321098765432109876543210
            bit: ----------------------*-------**
            '''
            # logger.info(data)
            return data
        except Exception:
            logger.error('[XYZ Class ] MotorBoard.reference_status('+axis+') inactive')
            raise

    def limit_switch_status(self, axis):
        try:
            resp = self.reference_status(axis=axis)
            resp_int = int(resp)
            if resp_int & (1 << 0):
                left = 1
            else:
                left = 0

            if resp_int & (1 << 1):
                right = 1
            else:
                right = 0

        except Exception as e:
            logger.warning(f'[XYZ Class ] limit_switch_status({axis}) failed: {e}')
            left, right = -1, -1

        return left, right


    def get_current_firmware(self):
        """ Returns current version of firmware on Motorboard

            :return the string
                Etaluma Motor Controller Board <BOARD TYPE>
                Firmware:     <DATE>
        """
        response = self.exchange_command('INFO')
        if not response:
            logger.info('[XYZ Class ] MotorBoard not connected. Unable to check current firmware')
            return
        return response

    def get_axes_config(self):
        return self.axes_config

    def get_axis_limits(self, axis: str):
        AXES_CONFIG = self.axes_config
        if axis not in AXES_CONFIG:
            logger.error(f"[XYZ Class ] MotorBoard.get_axis_limits(): Unsupported axis ({axis})")
            raise HardwareError(f"Unsupported axis ({axis})")

        axis_config = AXES_CONFIG[axis]
        if 'limits' not in axis_config:
            logger.error(f"[XYZ Class ] MotorBoard.get_axis_limits(): No limits defined for axis ({axis})")
            raise HardwareError(f"Axis {axis} does not have defined limits")

        return axis_config['limits']
