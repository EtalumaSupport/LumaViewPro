#!/usr/bin/python3

'''
MIT License

Copyright (c) 2023 Etaluma, Inc.

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

MODIFIED:
March 16, 2023
'''

#import threading
#import queue
import time
import serial
import serial.tools.list_ports as list_ports

class MotorBoard:

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        print('[XYZ Class ] MotorBoard.__init__()')
        ports = list_ports.comports(include_links = True)
        self.found = False
        self.overshoot = False
        self.backlash = 25 # um of additional downlaod travel in z for drive hysterisis

        for port in ports:
            if (port.vid == 0x2E8A) and (port.pid == 0x0005):
                print('[XYZ Class ] Motor Controller at', port.device)
                self.port = port.device
                self.found = True
                break

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=0.01 # seconds
        self.write_timeout=0.01 # seconds
        self.driver = False
        try:
            print('[XYZ Class ] Found motor controller and about to establish connection.')
            self.connect()
        except:
            print('[LED Class ] Found motor controller but unable to establish connection.')
            raise

    def connect(self):
        """ Try to connect to the motor controller based on the known VID/PID"""
        try:
            print('[XYZ Class ] Found motor controller and about to create driver.')
            self.driver = serial.Serial(port=self.port,
                                        baudrate=self.baudrate,
                                        bytesize=self.bytesize,
                                        parity=self.parity,
                                        stopbits=self.stopbits,
                                        timeout=self.timeout,
                                        write_timeout=self.write_timeout)
            self.driver.close()
            self.driver.open()
            
            print('[XYZ Class ] MotorBoard.connect() succeeded')

            self.xyhome()
            
        except:
            self.driver = False
            print('[XYZ Class ] MotorBoard.connect() failed')

    #----------------------------------------------------------
    # Define Communication
    #----------------------------------------------------------
    def exchange_command(self, command):
        """ Exchange command through serial to SPI to the motor boards
        This should NOT be used in a script. It is intended for other functions to access"""

        stream = command.encode('utf-8')+b"\r\n"

        if self.driver != False:
            try:
                self.driver.close()
                self.driver.open()
                self.driver.write(stream)
                response = self.driver.readline()
                response = response.decode("utf-8","ignore")

                # (too often) print('[XYZ Class ] MotorBoard.exchange_command('+command+') succeeded')
                return response[:-2]

            except serial.SerialTimeoutException:
                self.driver = False
                print('[LED Class ] LEDBoard.exchange_command('+command+') Serial Timeout Occurred')

            except:
                self.driver = False
                print('[LED Class ] LEDBoard.exchange_command('+command+') failed')

        else:
            try:
                self.connect()
            except:
                return

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
    # Z (Focus) Functions
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        # print('[XYZ Class ] MotorBoard.z_ustep2um('+str(ustep)+')')
        um = 0.00586 * ustep # 0.00586 um/ustep Olympus Z
        return um

    def z_um2ustep(self, um):
        # print('[XYZ Class ] MotorBoard.z_um2ustep('+str(um)+')')       
        ustep = int( um / 0.00586 ) # 0.00586 um/ustep Olympus Z
        return ustep

    def zhome(self):
        """ Home the objective """
        print('[XYZ Class ] MotorBoard.zhome()')        
        self.exchange_command('ZHOME')

    #----------------------------------------------------------
    # XY Stage Functions
    #----------------------------------------------------------
    def xy_ustep2um(self, ustep):
        # print('[XYZ Class ] MotorBoard.xy_ustep2um('+str(ustep)+')')
        um = 0.0496 * ustep # 0.0496 um/ustep
        return um

    def xy_um2ustep(self, um):
        # print('[XYZ Class ] MotorBoard.xy_um2ustep('+str(um)+')')
        ustep = int( um / 0.0496) # 0.0496 um/ustep
        return ustep

    def xyhome(self):
        """ Home the stage which also homes the objective first """
        print('[XYZ Class ] MotorBoard.xyhome()')   
        if self.found:
            self.exchange_command('HOME')

    def xycenter(self):
        """ Home the stage which also homes the objective first """
        print('[XYZ Class ] MotorBoard.xycenter()')
        self.exchange_command('CENTER')
            
    #----------------------------------------------------------
    # T (Turret) Functions
    #----------------------------------------------------------
    def t_ustep2deg(self, ustep):
        # print('[XYZ Class ] MotorBoard.t_ustep2deg('+str(ustep)+')')
        um = 1. * ustep # needs correct value
        return um

    def t_deg2ustep(self, um):
        # print('[XYZ Class ] MotorBoard.t_ustep2deg('+str(um)+')')
        ustep = int( um / 1.) # needs correct value
        return ustep

    def thome(self):
        """ Home the turret, not yet functional in hardware"""
        print('[XYZ Class ] MotorBoard.thome()')
        self.exchange_command('THOME')

    def tmove(self, degrees):
        """ Move the turret, not yet functional in hardware"""
        print('[XYZ Class ] MotorBoard.thome()')
        steps = self.t_deg2ustep(degrees)
        self.move('T', steps)

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------
 
    def move(self, axis, steps):
        """ Move the axis to an absolute position (in usteps)
        compared to Home """
        # print('move', axis, steps)

        # print('def move(self, axis, steps)', axis, steps)
        if steps < 0:
            steps += 0x100000000 # twos compliment
        self.exchange_command('TARGET_W' + axis + str(steps))

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
        else:
            um = self.xy_ustep2um(position)

        return um

    # Get current position (in um)
    def current_pos(self, axis):
        """Get current position (in um) of axis"""
        
        try:
            response = self.exchange_command('ACTUAL_R' + axis)
            position = int(response)
        except:
            position = 0

        if axis == 'Z':
            um = self.z_ustep2um(position)
        else:
            um = self.xy_ustep2um(position)

        return um
 
    # Move to absolute position (in um)
    def move_abs_pos(self, axis, pos):
        """ Move to absolute position (in um) of axis"""
        # print('move_abs_pos', axis, pos)
        if axis == 'Z': # Z bound between 0 to 14mm
            if pos < 0:
                pos = 0.
            elif pos > 14000:
                pos = 14000.
            steps = self.z_um2ustep(pos)
        elif axis == 'X': # X bound 0 to 120mm
            if pos < 0:
                pos = 0.
            elif pos > 120000:
                pos = 120000.
            steps = self.xy_um2ustep(pos)
        elif axis == 'Y': # y bound 0 to 80mm
            if pos < 0:
                pos = 0.
            elif pos > 80000:
                pos = 80000.
            steps = self.xy_um2ustep(pos)

        if axis=='Z': # perform overshoot to always come from one direction
            # get current position
            current = self.current_pos('Z')

            # if the current position is above the new target position
            if current > pos:
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

    # Move by relative distance (in um)
    def move_rel_pos(self, axis, um):
        """ Move by relative distance (in um) of axis """

        # Read target position in um
        pos = self.target_pos(axis)
        self.move_abs_pos(axis, pos+um)
        print('[XYZ Class ] MotorBoard.move_rel_pos('+axis+','+str(um)+') succeeded')
 
    #----------------------------------------------------------
    # Ramp and Reference Switch Status Register
    #----------------------------------------------------------

    # return True if current and target position are at home.
    def home_status(self, axis):
        """ Return True if axis is in home position"""

        # print('[XYZ Class ] MotorBoard.home_status('+axis+')')      
        try:
            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            if bits[31] == '1':
                return True
            else:
                return False
        except:
            print('[XYZ Class ] MotorBoard.home_status('+axis+') inactive')        
            return False

    # return True if current position and target position are the same
    def target_status(self, axis):
        """ Return True if axis is at target position"""

        # print('[XYZ Class ] MotorBoard.target_status('+axis+')')
        try:
            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            if bits[22] == '1':
                return True
            else:
                return False
  
        except:
            print('[XYZ Class ] MotorBoard.get_limit_status('+axis+') inactive')
            return False


    # Get all reference status register bits as 32 character string (32-> 0)
    def reference_status(self, axis):
        """ Get all reference status register bits as 32 character string (32-> 0) """
        try:

            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            # data is an integer that represents 4 bytes, or 32 bits,
            # largest bit first
            '''
            bit: 33222222222211111111110000000000
            bit: 10987654321098765432109876543210
            bit: ----------------------*-------**
            '''
            print(data)
            return data
        except:
            print('[XYZ Class ] MotorBoard.reference_status('+axis+') inactive')
            return False


'''
# signed 32 bit hex to dec
if value >=  0x80000000:
    value -= 0x10000000
print(int(value))

# signed dec to 32 bit hex
value = -200000
if value < 0:
    value = 4294967296+value
print(hex(value))
'''