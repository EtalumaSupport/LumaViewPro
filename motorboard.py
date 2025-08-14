#!/usr/bin/python3

'''
MIT License

Copyright (c) 2024 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.

AUTHORS:
Kevin Peter Hickerson, The Earthineering Company
Anna Iwaniec Hickerson, Keck Graduate Institute
Gerard Decker, The Earthineering Company
'''

import time
from requests.structures import CaseInsensitiveDict
import serial
import serial.tools.list_ports as list_ports
from lvp_logger import logger

import threading

class MotorBoard:

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        logger.info('[XYZ Class ] MotorBoard.__init__()')
        ports = list_ports.comports(include_links = True)
        self.found = False
        self.overshoot = False
        self.backlash = 25 # um of additional downlaod travel in z for drive hysterisis
        self._has_turret = False
        self.initial_homing_complete = False
        self.initial_t_homing_complete = False
        self.port = None
        self.thread_lock = threading.RLock()

        for port in ports:
            if (port.vid == 0x2E8A) and (port.pid == 0x0005):
                logger.info(f'[XYZ Class ] Motor Controller at {port.device}')
                self.port = port.device
                self.found = True
                break

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=None # seconds
        self.write_timeout=None # seconds
        self.driver = False
        self._fullinfo = None
        try:
            logger.info('[XYZ Class ] Found motor controller and about to establish connection.')
            self.connect()
        except:
            logger.error('[XYZ Class ] Found motor controller but unable to establish connection.')
            raise

    def connect(self):
        """ Try to connect to the motor controller based on the known VID/PID"""
        # Lock to ensure mutex
        with self.thread_lock:
            try:
                logger.info('[XYZ Class ] Found motor controller and about to create driver.')
                if self.port is not None:
                    self.driver = serial.Serial(port=self.port,
                                                baudrate=self.baudrate,
                                                bytesize=self.bytesize,
                                                parity=self.parity,
                                                stopbits=self.stopbits,
                                                timeout=self.timeout,
                                                write_timeout=self.write_timeout)
                else:
                    raise ValueError("No port found for motor controller")
                
                self.driver.close()
                #print([comport.device for comport in serial.tools.list_ports.comports()])
                self.driver.open()

                logger.info('[XYZ Class ] MotorBoard.connect() succeeded')

                # After powering on the scope, the first command seems to be ignored.
                # This is to ensure the following commands are followed
                # Dev 2023-MAY-16 the above 2 comments are suspect - doesn't seem to matter
                #Sometimes the firmware fails to start (or the port has a \x00 left in the buffer), this forces MicroPython to reset, and the normal firmware just complains
                self.driver.write(b'\x04\n')
                logger.debug('[XYZ Class ] MotorBoard.connect() port initial state: %r'%self.driver.readline())
                # Fullinfo checks to see if it has a turret, so call that here
                self._fullinfo = self.fullinfo()
            except Exception as e:
                self.driver = False
                logger.error(f'[XYZ Class ] MotorBoard.connect() failed: {e}')

    def disconnect(self):
        logger.info('[XYZ Class ] Disconnecting from motor controller...')
        try:
            if self.driver is not None and self.driver is not False:
                self.driver.close()
                self.driver = None
                logger.info('[XYZ Class ] MotorBoard.disconnect() succeeded')
            else:
                logger.info('[XYZ Class ] MotorBoard.disconnect() failed: Motor controller not connected')
        except Exception as e:
            logger.error(f'[XYZ Class ] MotorBoard.disconnect() failed: {e}')

    #----------------------------------------------------------
    # Define Communication
    #----------------------------------------------------------
    def exchange_command(self, command, response_numlines=1):
        """ Exchange command through serial to SPI to the motor boards
        This should NOT be used in a script. It is intended for other functions to access"""
        with self.thread_lock:
            stream = command.encode('utf-8')+b"\n"
            #print(stream)

            if not self.driver:
                try:
                    self.connect()
                except:
                    return
            try:
                self.driver.write(stream)
                #if (command)=='HOME': # ESW to increase homing reliability
                #    CRLF = command.encode('utf-8')+b"\r\n"
                #    self.driver.write(CRLF)
                resp_lines = [self.driver.readline() for _ in range(response_numlines)]
                response = [r.decode("utf-8","ignore").strip() for r in resp_lines]
                if response_numlines == 1:
                    if '\r' in response[0].strip():
                        response[0] = response[0].rsplit('\r')[-1]
                    response = response[0]
                logger.debug('[XYZ Class ] MotorBoard.exchange_command('+command+') %r'%response)

            except serial.SerialTimeoutException:
                self.driver = False
                logger.error('[XYZ Class ] MotorBoard.exchange_command('+command+') Serial Timeout Occurred')
                response = None

            except:
                self.driver = False
                logger.error('[XYZ Class ] MotorBoard.exchange_command('+command+') failed')
                response = None
            
            return response


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
    def infomation(self):
        self.exchange_command('INFO')

    def fullinfo(self):
        info = self.exchange_command("FULLINFO")
        logger.info('[XYZ Class ] MotorBoard.fullinfo(): %s', info)
        info = info.split()
        model = info[info.index("Model:")+1]
        if model[-1] == "T":
            self._has_turret = True

        serial_number = info[info.index("Serial:")+1]

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
                raise

            # Extra protection for now in case motorboard responds with a different string that doesnt start with ERROR
            if not resp.isdigit():
                raise

        except:
            resp = DEFAULT_ACCELERATION_LIMIT
            using_default = True

        using_default_str = "-> default" if using_default else ""
        logger.info(f'[XYZ Class ] MotorBoard.acceleration_limit({command}): {resp} {using_default_str}')
        
        # TODO parse response value out of response string once implemented
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
        logger.info(f'[XYZ Class ] MotorBoard.xyhome() -> {resp}')
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
        logger.info(f'[XYZ Class ] MotorBoard.thome() -> {resp}')
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
        except:
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
        except:
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

        AXES_CONFIG = {
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

        if axis not in AXES_CONFIG:
            raise Exception(f"Unsupported axis ({axis})")
        
        axis_config = AXES_CONFIG[axis]

        if ('limits' in axis_config) and (ignore_limits == False):
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

            if bits[31] == '1':
                return True
            else:
                return False
        except:
            logger.error('[XYZ Class ] MotorBoard.home_status('+axis+') inactive')
            raise

    # return True if current position and target position are the same
    def target_status(self, axis):
        """ Return True if axis is at target position"""

        # logger.info('[XYZ Class ] MotorBoard.target_status('+axis+')')
        try:
            #logger.warning(f"AXIS PARAM: ====={axis}=====")
            payload = 'STATUS_R' + axis
            #logger.warning(f"Sending payload to motorboard: {payload}=====")
            response = self.exchange_command(payload)
            #logger.warning(f"Response: {response}")
            if response is None:
                raise
            data = int( response )
            bits = format(data, 'b').zfill(32)

            if bits[22] == '1':
                return True
            else:
                return False
  
        except:
            logger.error('[XYZ Class ] MotorBoard.get_limit_status('+axis+') inactive')
            raise
            #return False


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
        except:
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

        except:
            left, right = -1, -1

        return left, right


    #-------------------------------------------------------------------------------
    #                         FIRMWARE HANDLING 
    #
    # Last Modified: 5/24/2023
    # 
    # TODO: Implement firmware version comparing (firmware_is_up_to_date() function)
    # TODO: Add GUI Controls for Firmware Handling 
    # TODO: Eventually move firmware handling to separate .py file (Firmware Class)
    #--------------------------------------------------------------------------------

    def check_firmware(self):
        """ Checks and updates motorboard firmware if out of date """

        # Ensure Motorboard is connected and found
        if not self.found or not self.driver:
            logger.warning(f'[XYZ Class] Cannot perform firmware update. Motorboard not connected or found')
            return
        
        # If firmware is outdated, attempt to update firmware. 
        if not self.firmware_is_up_to_date():
            logger.info(f'[XYZ Class] Motorboard is out of date. Installing new firmware...')

            if self.update_firmware():
                logger.info(f'[XYZ Class] Succesfully updated Motorboard firmware')
            else:
                logger.warning(f'[XYZ Class] Failed to update Motorboard firmware')
        else:
            logger.info(f'[XYZ Class] Motorboard firmware is already up to date')

    def update_firmware(self):
        """ Performs the firmware update on the motorboard

            :return a boolean true if firmware update was successful, false otherwise

        """
        # Ensure Motorboard is connected and found
        if not self.found or not self.driver:
            logger.warning(f'[XYZ Class] Cannot perform firmware update. Motorboard not connected or found')
            return False
        
        # Obtain the latest firmware files from the firmware repository
        FIRMWARE_URL = self.get_firmware_URL('EtalumaSupport/Firmware', 'Firmware', 'Motor Controller/LVP current functional/')
        AUTH_TOKEN = None # Insert authentication token here

        latest_firmware = self.get_latest_firmware(FIRMWARE_URL, AUTH_TOKEN)
        if not latest_firmware:
            logger.warning(f'[XYZ Class] Failed to get latest firmware from remote repository')
            return False

        try:
        
            # Attempts to overwrite current files on PICO with new firmware files
            file_manager = ampy.files.Files(self.device)
            for file_name in latest_firmware.keys():
                file_manager.put(latest_firmware[file_name])
                logger.info(f'[XYZ Class] Succesfully upload firmware file {file_name} to Motorboard')

            return True
        except:
            logger.error(f'[XYZ Class] Failed to upload new Firmware files to Motorboard')
            raise


    def get_firmware_URL(self, owner, repo, path):
        """ Generates a GitHub API URL to make get requests from
        
            :param string owner: the owner of the GitHub Repo
            :param string repo: the title of the repo
            :param path: the path to firmware files
            :return the GitHub URL string

        """
        return f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

    def get_latest_firmware(self, firmware_url, auth_token):
        """ Retrieves the latest firmware files from a GitHub repository

            :param string firmware_url: the URL to the Github API containing the Firmware
            :param string auth_token: an authentification token to access the repo if it is private
            :return firmware_files dictionary of the format ['FILE NAME' : <FILE CONTENT>]

        """
        firmware_files = {}

        # Authentication token needed if repo is private
        headers = CaseInsensitiveDict()
        if auth_token:
            headers["Authorization"] = f'Bearer {auth_token}'   

        # Make GET request from firmware URL 
        response = requests.get(firmware_url, headers=headers, stream = True)

        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f'[XYZ Class] Succesfully reached firmware repository at: {firmware_url}')

            # Parse the response 
            contents = response.json()

            # Download each file in the folder
            for item in contents:
                file_name = item['name']
                download_url = item['download_url']
                if not download_url:
                    continue

                # Send a GET request to download the file with authentication
                file_response = requests.get(download_url, headers=headers, stream=True)
                
                # Check if the file was downloaded successfully
                if file_response.status_code == 200:
                    firmware_files[file_name] = file_response
                    logger.info(f'[XYZ Class] Succesfully download firmware file: {file_name}')
                else:
                    logger.warning(f'[XYZ Class] Failed to download firmware file: {file_name}')
        else:
            logger.warning(f'[XYZ Class] Failed to reach GitHub API. STATUS CODE: {response.status_code}')

        return firmware_files

    def firmware_is_up_to_date(self):
        """ Checks if current firmware is out of date

            :return a boolean true if firmware is up to date, false otherwise
        """
        # TO BE IMPLEMENTED
        # Need Eric to add VERSION.txt or alternative to firmware to allow for easy version comparison
        return True
        
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

'''
# signed 32 bit hex to dec
if value >=  0x80000000:
    value -= 0x10000000
logger.info(int(value))

# signed dec to 32 bit hex
value = -200000
if value < 0:
    value = 4294967296+value
logger.info(hex(value))
'''
