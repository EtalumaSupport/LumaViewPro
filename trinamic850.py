#!/usr/bin/python3

'''
MIT License

Copyright (c) 2020 Etaluma, Inc.

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
January 22, 2023
'''

#import threading
#import queue
import time
import serial
import serial.tools.list_ports as list_ports

class TrinamicBoard:

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        ports = list_ports.comports(include_links = True)
        self.message = 'TrinamicBoard.__init__()'
        self.found = False
        self.overshoot = False
        self.backlash = 25 # um of additional downlaod travel in z for drive hysterisis

        for port in ports:
            if (port.vid == 0x2E8A) and (port.pid == 0x0005):
                print('Motor Controller at', port.device)
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
            self.connect()
        except:
            print('Found device but motor controller unable establish connection.')
            raise

    # def __del__(self):
    #     if self.driver != False:
    #         self.driver.close()

    def connect(self):
        """ Try to connect to the motor controller based on the known VID/PID"""
        try:
            self.driver = serial.Serial(port=self.port,
                                        baudrate=self.baudrate,
                                        bytesize=self.bytesize,
                                        parity=self.parity,
                                        stopbits=self.stopbits,
                                        timeout=self.timeout,
                                        write_timeout=self.write_timeout)
            self.driver.close()
            self.driver.open()
            
            response = self.exchange_command('INFO')
            self.message = 'TrinamicBoard.connect()\n' + response
            print('TrinamicBoard.connect()\n' + response)
        except:
            self.driver = False
            self.message = 'TrinamicBoard.connect() failed'
            print('TrinamicBoard.connect() failed')
            #raise

    #----------------------------------------------------------
    # Define Communication
    #----------------------------------------------------------
    def exchange_command(self, command):
        """ Exchange command through serial to SPI to the Trinamic boards
        This should NOT be used in a script. It is intended for other functions to access"""

        if self.driver != False:
            try:
                self.driver.close()
                self.driver.open()
                self.driver.write(command.encode('utf-8')+b"\r\n")
                response = self.driver.readline()
                response = response.decode("utf-8","ignore")
                self.message = 'TrinamicBoard.exchange_command('+command+') succeeded'
                return response[:-2]                

            except serial.SerialTimeoutException:
                self.message = 'TrinamicBoard.exchange_command('+command+') serial timeout occurred'
                raise IOError('Trinamic board timeout occured.')
        else:
            try:
                self.connect()
            except:
                self.message = 'TrinamicBoard.exchange_command('+command+') Unable to connect to board'
                raise IOError('Unable to connect to Trinamic board.')

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

    
    # def SPI_put (self, pin: int, address: int, data: int):
    #     if self.buffer == False:
    #         self.buffer = queue.Queue(1024)  # could add buffer size to settings someday
        
    #     if self.buffer.full():
    #        print("Trinamic SPI write buffer size is "+self.buffer.qsize()+" and is full.")
    #        return
        
    #     self.buffer.put({
    #             pin:     pin,
    #             address: address,
    #             data:    data
    #         })

    # def SPI_get (self):
    #     if self.buffer == False:
    #         return
        
    #     cmd = self.buffer.get()
    #     self.SPI_write(cmd.pin, cmd.address, cmd.data)

    #----------------------------------------------------------
    # Z (Focus) Functions
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        self.message = 'TrinamicBoard.z_ustep2um('+str(ustep)+')'            
        um = 0.00586 * ustep # 0.00586 um/ustep Olympus Z
        return um

    def z_um2ustep(self, um):
        self.message = 'TrinamicBoard.z_um2ustep('+str(um)+')'            
        ustep = int( um / 0.00586 ) # 0.00586 um/ustep Olympus Z
        return ustep

    def zhome(self):
        """ Home the objective """
        self.message = 'TrinamicBoard.zhome()'            
        if self.found:
            self.exchange_command('ZHOME')

    #----------------------------------------------------------
    # XY Stage Functions
    #----------------------------------------------------------
    def xy_ustep2um(self, ustep):
        self.message = 'TrinamicBoard.xy_ustep2um('+str(ustep)+')'            
        um = 0.0496 * ustep # 0.0496 um/ustep
        return um

    def xy_um2ustep(self, um):
        self.message = 'TrinamicBoard.xy_um2ustep('+str(um)+')'            
        ustep = int( um / 0.0496) # 0.0496 um/ustep
        return ustep

    def xyhome(self):
        """ Home the stage which also homes the objective first """
        self.message = 'TrinamicBoard.xyhome()'            
        if self.found:
            self.exchange_command('HOME')

    #----------------------------------------------------------
    # T (Turret) Functions
    #----------------------------------------------------------
    def t_ustep2deg(self, ustep):
        self.message = 'TrinamicBoard.t_ustep2deg('+str(ustep)+')'            
        um = 1. * ustep # needs correct value
        return um

    def t_deg2ustep(self, um):
        self.message = 'TrinamicBoard.t_ustep2deg('+str(um)+')'            
        ustep = int( um / 1.) # needs correct value
        return ustep

    def thome(self):
        """ Home the turret, not yet functional in hardware"""
        self.message = 'TrinamicBoard.thome()'            
        if self.found:
            self.exchange_command('THOME')

    def tmove(self, degrees):
        """ Move the turret, not yet functional in hardware"""
        self.message = 'TrinamicBoard.thome()'
        steps = self.t_deg2ustep(degrees)
        if self.found:
            self.move('T', steps)

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------
 
    def move(self, axis, steps):
        """ Move the axis to an absolute position (in usteps)
        compared to Home """

        if steps < 0:
            steps += 0x100000000 # twos compliment
        self.exchange_command('TARGET_W' + axis + str(steps))

    # Get target position
    def target_pos(self, axis):
        """ Get the target position of an axis"""

        if self.found:
            try:
                responce = self.exchange_command('TARGET_R' + axis)
                position = int(responce)
            except ValueError:
                return 0 # short term fix
                raise IOError(f"Expected target position from motor controller. Repsonded with '{responce}'")
            except:
                raise

            if axis == 'Z':
                um = self.z_ustep2um(position)
            else:
                um = self.xy_ustep2um(position)

            self.message = 'TrinamicBoard.target_pos('+axis+') succeeded'            
            return um
        else:
            self.message = 'TrinamicBoard.target_pos('+axis+') inactive'            
            return 0

    # Get current position (in um)
    def current_pos(self, axis):
        """Get current position (in um) of axis"""
        
        if self.found:
            try:
                responce = self.exchange_command('ACTUAL_R' + axis)
                position = int(responce)
            except ValueError:
                return 0 #short term fix
                raise IOError(f"Expected current position from motor controller. Repsonded with '{responce}'")
            except:
                raise

            if axis == 'Z':
                um = self.z_ustep2um(position)
            else:
                um = self.xy_ustep2um(position)

            self.message = 'TrinamicBoard.current_pos('+axis+') succeeded'            
            return um
        else:
            self.message = 'TrinamicBoard.current_pos('+axis+') inactive'            
            return 0

    # Move to absolute position (in um)
    def move_abs_pos(self, axis, pos):
        """ Move to absolute position (in um) of axis"""

        if self.found:
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
            self.message = 'TrinamicBoard.move_abs_pos('+axis+','+str(pos)+') succeeded'
        else:
            self.message = 'TrinamicBoard.move_abs_pos('+axis+','+str(pos)+') inactive'

    # Move by relative distance (in um)
    def move_rel_pos(self, axis, um):
        """ Move by relative distance (in um) of axis """

        if self.found:
            # Read target position in um
            pos = self.target_pos(axis)
            self.move_abs_pos(axis, pos+um)
            self.message = 'TrinamicBoard.move_rel_pos('+axis+','+str(um)+') succeeded'
        else:
            self.message = 'TrinamicBoard.move_rel_pos('+axis+','+str(um)+') inactive'

    #----------------------------------------------------------
    # Ramp and Reference Switch Status Register
    #----------------------------------------------------------

    # return True if current and target position are at home.
    def home_status(self, axis):
        """ Return True if axis is in home position"""

        self.message = 'TrinamicBoard.home_status('+axis+')'            
        if self.found:
            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            if bits[31] == '1':
                return True
            else:
                return False
        else:
            self.message = 'TrinamicBoard.home_status('+axis+') inactive'            
            return False

    # return True if current position and target position are the same
    def target_status(self, axis):
        """ Return True if axis is at target position"""

        self.message = 'TrinamicBoard.target_status('+axis+')'  
        
        if self.found:
            data = int( self.exchange_command('STATUS_R' + axis) )
            bits = format(data, 'b').zfill(32)

            if bits[22] == '1':
                return True
            else:
                return False
  
        else:
            self.message = 'TrinamicBoard.get_limit_status('+axis+') inactive'            
            return False


    # Get all reference status register bits as 32 character string (32-> 0)
    def reference_status(self, axis):
        """ Get all reference status register bits as 32 character string (32-> 0) """
        if self.found:

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
        else:
            self.message = 'TrinamicBoard.reference_status('+axis+') inactive'            
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