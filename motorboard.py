#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import time
import lvp_logger
from lvp_logger import logger

from serialboard import SerialBoard


class MotorBoard(SerialBoard):

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        self.overshoot = False
        self.backlash = 25 # um of additional downlaod travel in z for drive hysterisis
        self._has_turret = False
        self.initial_homing_complete = False
        self.initial_t_homing_complete = False
        self._fullinfo = None
        self._connect_fails = 0

        super().__init__(vid=0x2E8A, pid=0x0005, label='[XYZ Class ]',
                         timeout=30, write_timeout=5)

        # Backward-compatible alias for lock name
        self.thread_lock = self._lock

        self.axes_config = {
            'Z': {
                'limits': {
                    'min': 0.,
                    'max': 14000.,
                },
                'move_func': self.z_um2ustep
            },
            'X': {
                'limits': {
                    'min': 0.,
                    'max': 120000.,
                },
                'move_func': self.xy_um2ustep
            },
            'Y': {
                'limits': {
                    'min': 0.,
                    'max': 80000.,
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
                self._fullinfo = self.fullinfo()
                logger.info('[XYZ Class ] Connected to motor controller')
            except Exception as e:
                self._close_driver()
                self._connect_fails += 1
                if self._connect_fails >= 10:
                    logger.critical(f'[XYZ Class ] MotorBoard.connect() failed 10 times, pausing thread logs')
                    lvp_logger.pause_thread()
                logger.error(f'[XYZ Class ] MotorBoard.connect() failed: {e}')


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
        info = self._fullinfo
        return info['model']

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
            if resp.startswith("ERROR"):
                raise ValueError(f"Firmware returned ERROR for {command}")

            # Extra protection for now in case motorboard responds with a different string that doesnt start with ERROR
            if not resp.isdigit():
                raise ValueError(f"Non-numeric response for {command}: {resp}")

        except Exception:
            resp = DEFAULT_ACCELERATION_LIMIT
            using_default = True

        using_default_str = "-> default" if using_default else ""
        logger.info(f'[XYZ Class ] MotorBoard.acceleration_limit({command}): {resp} {using_default_str}')

        return resp


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
    # Z (Focus) Functions
    # Stock actuator = 0.30 mm pitch.  (1 rev/0.30 mm) x (200 steps/rev) x (256 usteps/step) = 170667 ustep/mm
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        # logger.info('[XYZ Class ] MotorBoard.z_ustep2um('+str(ustep)+')')
        um = (ustep * 1000 / 170667)
        return um

    def z_um2ustep(self, um):
        # logger.info('[XYZ Class ] MotorBoard.z_um2ustep('+str(um)+')')
        ustep = int( (170667 * um) / 1000)
        return ustep

    def zhome(self):
        """ Home the objective """
        resp = self.exchange_command('ZHOME')
        logger.info(f'[XYZ Class ] MotorBoard.zhome() -> {resp}')

    #----------------------------------------------------------
    # XY Stage Functions
    # Stock actuator = 2.54mm pitch.  (1 rev/2.540 mm) x (200 steps/rev) x (256 usteps/step) = 20157 ustep/mm
    #----------------------------------------------------------

    def xy_ustep2um(self, ustep):
        # logger.info('[XYZ Class ] MotorBoard.xy_ustep2um('+str(ustep)+')')
        um = (ustep * 1000 / 20157)
        return um

    def xy_um2ustep(self, um):
        # logger.info('[XYZ Class ] MotorBoard.xy_um2ustep('+str(um)+')')
        ustep = int( (20157 * um) / 1000)
        return ustep

    def xyhome(self):
        """ Home the stage which also homes the objective first """
        resp = self.exchange_command('HOME')
        logger.info(f'[XYZ Class ] MotorBoard.xyhome() -> {resp}', extra={'force_error': True})
        if (resp is not None) and ('XYZ home complete' in resp):
            self.initial_homing_complete = True

    def has_xyhomed(self):
        return self.initial_homing_complete

    def xycenter(self):
        """ Home the stage which also homes the objective first """
        logger.info('[XYZ Class ] MotorBoard.xycenter()')
        self.exchange_command('CENTER')

    #----------------------------------------------------------
    # T (Turret) Functions
    #----------------------------------------------------------
    def t_ustep2deg(self, ustep):
        # logger.info('[XYZ Class ] MotorBoard.t_ustep2deg('+str(ustep)+')')
        degrees = 90./80000. * ustep # needs correct value
        return degrees

    def t_ustep2pos(self, ustep):
        return int(self.t_ustep2deg(ustep=ustep)/90)+1

    def t_deg2ustep(self, degrees):
        # logger.info('[XYZ Class ] MotorBoard.t_ustep2deg('+str(um)+')')
        ustep = int( degrees * 80000./90.) # needs correct value
        #print("ustep: ",ustep)
        return ustep

    def t_pos2ustep(self, position):
        return self.t_deg2ustep(degrees=90*(position-1))

    def thome(self):
        """ Home the turret, need to test if functional in hardware"""
        resp = self.exchange_command('THOME')
        logger.info(f'[XYZ Class ] MotorBoard.thome() -> {resp}', extra={'force_error': True})
        if (resp is not None) and ('T home successful' in resp):
            self.initial_t_homing_complete = True

    def has_turret(self) -> bool:
        return self._has_turret

    def has_thomed(self):
        # Note: When the motorboard firmware performs an XYZ homing, it also
        # does a T homing if a turret is present
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
        except Exception:
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
        except Exception:
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
            raise Exception(f"Unsupported axis ({axis})")

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
                self.overshoot = True
                # First overshoot downwards
                overshoot = self.z_um2ustep(pos-self.backlash) # target minus backlash
                overshoot = max(1, overshoot)
                #self.SPI_write (self.chip_pin[axis], self.write_target[axis], overshoot)
                self.move(axis, overshoot)
                while not self.target_status('Z'):
                    time.sleep(0.001)
                # complete overshoot
                self.overshoot = False

        self.move(axis, steps)

    # Move by relative distance (in um or degrees for Turret)
    def move_rel_pos(self, axis, um, overshoot_enabled: bool = False):
        """ Move by relative distance (in um for X, Y, Z or position for T) of axis """

        # Read target position in um
        pos = self.target_pos(axis)
        self.move_abs_pos(axis, pos+um, overshoot_enabled=overshoot_enabled)
        logger.info('[XYZ Class ] MotorBoard.move_rel_pos('+axis+','+str(um)+') succeeded')

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

        except Exception:
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
            raise Exception(f"Unsupported axis ({axis})")

        axis_config = AXES_CONFIG[axis]
        if 'limits' not in axis_config:
            logger.error(f"[XYZ Class ] MotorBoard.get_axis_limits(): No limits defined for axis ({axis})")
            raise Exception(f"Axis {axis} does not have defined limits")

        return axis_config['limits']
