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
October 8, 2022
'''

from mcp2210 import Mcp2210, Mcp2210GpioDesignation, Mcp2210GpioDirection
import struct    # For making c style data structures, and send them through the mcp chip
import sched
import time
import os

platform = os.sys.platform 

if platform == 'win32': # Windows
    import pywinusb.hid as hid
else: # Not Windows (linux, MacOS)
    import usb.core # Access to USB functionality  from pyusb available at https://pyusb.github.io/pyusb
    import usb.util

class TrinamicBoard:

    chip_pin = {
        'X': 0,
        'Y': 0,
        'Z': 1,
        'T': 1
    }
    read_actual = {
        'X': 0x21,
        'Y': 0x41,
        'Z': 0x41,
        'T': 0x21
    }
    write_actual = {
        'X': 0xA1,
        'Y': 0xC1,
        'Z': 0xC1,
        'T': 0xA1
    }
    read_target = {
        'X': 0x2D,
        'Y': 0x4D,
        'Z': 0x4D,
        'T': 0x2D
    }
    write_target = {
        'X': 0xAD,
        'Y': 0xCD,
        'Z': 0xCD,
        'T': 0xAD
    }
    limit_status = { # reference switch status
        # bit 0, left status (1=active)
        # bit 1, right status (1=active)
        'X': 0x35,
        'Y': 0x55,
        'Z': 0x55,
        'T': 0x35
    }

    #----------------------------------------------------------
    # Initialize connection through SPI
    #----------------------------------------------------------
    def __init__(self, **kwargs):

        self.found = False
        self.mssg = 'TrinamicBoard.__init__()'
        self.overshoot = False

        try:
            if platform == 'win32':
                devices = hid.core.find_all_hid_devices()
                for device in devices:
                    if device.vendor_id == 0x04D8 and device.product_id == 0x00DE:
                        SPI_serial = device.serial_number
            else:
                device = usb.core.find(idVendor=0x04d8, idProduct=0x00de)  # find the mcp2210 USB device, idVendor=0x04d8, idProduct=0x00de is defaulted in the chip
                SPI_serial = usb.util.get_string(device, device.iSerialNumber ) # find the serial number of the MCP device
            
            print("Found Trinamic board with SPI serial number ", SPI_serial)

            self.chip = Mcp2210(SPI_serial)  # the serial number of the chip. Can be determined by using dmesg after plugging in the device
            print ("Found Device", self.chip)

            # set all GPIO lines on the MCP2210 as General GPIO lines
            # Tri-State all GPIO lines on the MCP2210
            for i in range(9):
                self.chip.set_gpio_designation(i, Mcp2210GpioDesignation.GPIO)
                self.chip.set_gpio_direction(i, Mcp2210GpioDirection.INPUT)

            self.configure()
            self.found = True
            self.mssg = 'TrinamicBoard.__init__() succeeded'            

        except OSError as err:
            print("OS error: {0}".format(err))
            print("Trinamic setup failed")
            self.mssg = 'TrinamicBoard.__init__() failed'            
            self.found = False


    #----------------------------------------------------------
    # Define SPI communication
    #----------------------------------------------------------

    def SPI_write (self, pin_number: int, addr: int, data: int):
        self.mssg = 'TrinamicBoard.SPI_write()'            
        # Set the correct pin as chip select. Only one chip select line at a time,
        # or else you will have data conflicts on receive.
        self.chip.set_gpio_designation(pin_number, Mcp2210GpioDesignation.CHIP_SELECT)

        # glue the address and data lines together.  make it big-endian.
        tx_data = struct.pack('>BI', addr, data)

        # squirt the config word down to the correct chip,
        # put anything coming back from the previous write cycle into rx_data
        rx_data = self.chip.spi_exchange(tx_data, cs_pin_number=pin_number)

        # turn the Chip Select pin back to a Tri-state input pin.
        self.chip.set_gpio_designation(pin_number, Mcp2210GpioDesignation.GPIO)

        # decode 40 bit returned squirt from trinamic TMC5072 chip
        spi_status = bin(rx_data[0])
        data = int.from_bytes(rx_data[1:5], byteorder='big', signed=True)

        return spi_status, data


    #----------------------------------------------------------
    # Import motor configurations
    #----------------------------------------------------------
    def configure(self):
        self.mssg = 'TrinamicBoard.configure()'            
        # Load settings for the xy motors from config file for 'XY' motor driver
        with open('./data/xymotorconfig.ini', 'r') as xyconfigFile:
            for line in xyconfigFile:                           # go through each line of the config file
                if 'writing' in line:                           # if the line has the word "writing" in it (all the right lines have this)
                    configLine = line.split()                   # split the line into an indexable list
                    addr = int(0x80 | int(configLine[0], 16))   # take the address field, format it, add 80 to it (to make it a write operation to the trinamic chip), throw it into the 'address' variable
                    data = int(configLine[2], 16)               # take the data field, format it, and put it into the 'data' variable
                    self.SPI_write(0, addr, data)                # call SPI Writer. right now were writing to chip 0
        print ("Successfully loaded parameters to XY motor driver from xymotorconfig.ini")

        # Load settings for the z motor
        with open('./data/ztmotorconfig.ini', 'r') as ztconfigFile:
            for line in ztconfigFile:
                if 'writing' in line:
                    configLine = line.split()
                    addr = int(0x80 | int(configLine[0], 16))
                    data = int(configLine[2], 16)
                    self.SPI_write(1, addr, data)
        print ("Successfully loaded parameters to ZT motor driver from zmotorconfig.ini")

    #----------------------------------------------------------
    # Z (Focus) Functions
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        self.mssg = 'TrinamicBoard.z_ustep2um('+str(ustep)+')'            
        um = 0.00586 * ustep # 0.00586 um/ustep Olympus Z
        return um

    def z_um2ustep(self, um):
        self.mssg = 'TrinamicBoard.z_um2ustep('+str(um)+')'            
        ustep = int( um / 0.00586 ) # 0.00586 um/ustep Olympus Z
        return ustep

    def zhome(self):
        self.mssg = 'TrinamicBoard.zhome()'            
        if self.found:
            # self.move_abs_pos('Z', -1000000)
            self.SPI_write(self.chip_pin['Z'], self.write_target['Z'], 4294967296-2000000000)

            # Schedule writing target and actual Z position to 0 after 4 seconds
            s = sched.scheduler()
            s.enter(3, 1, self.zhome_write)
            s.run()

    def zhome_write(self):
        self.mssg = 'TrinamicBoard.zhome_write()'            
        self.SPI_write (self.chip_pin['Z'], self.write_actual['Z'], 0x00000000)
        self.SPI_write (self.chip_pin['Z'], self.write_target['Z'], 0x00000000)

    # TODO: access the information of a limit switch
    def reference_status(self, axis):
        status, data = self.SPI_write (self.chip_pin[axis], self.limit_status[axis], 0x00000000)
        status, data = self.SPI_write (self.chip_pin[axis], self.limit_status[axis], 0x00000000)
        return data

    #----------------------------------------------------------
    # XY Stage Functions
    #----------------------------------------------------------
    def xy_ustep2um(self, ustep):
        self.mssg = 'TrinamicBoard.xy_ustep2um('+str(ustep)+')'            
        um = 0.0496 * ustep # 0.0496 um/ustep
        return um

    def xy_um2ustep(self, um):
        self.mssg = 'TrinamicBoard.xy_um2ustep('+str(um)+')'            
        ustep = int( um / 0.0496) # 0.0496 um/ustep
        return ustep

    def xyhome(self):
        self.mssg = 'TrinamicBoard.xyhome()'            
        if self.found:
            self.zhome()

            self.move_abs_pos('X', -1000000)
            self.move_abs_pos('Y', -1000000)

            # Schedule writing target and actual XY positions to 0 after 8 seconds
            s = sched.scheduler()
            s.enter(5, 2, self.xyhome_write)
            s.run()

    def xyhome_write(self):
        self.mssg = 'TrinamicBoard.xyhome_write()'            
        self.SPI_write(self.chip_pin['X'], self.write_actual['X'], 0x00000000)
        self.SPI_write(self.chip_pin['X'], self.write_target['X'], 0x00000000)

        self.SPI_write(self.chip_pin['Y'], self.write_actual['Y'], 0x00000000)
        self.SPI_write(self.chip_pin['Y'], self.write_target['Y'], 0x00000000)

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------
 
    # return True if current and target position are at home.
    def home_status(self, axis):
        self.mssg = 'TrinamicBoard.home_status('+axis+')'            
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
            status, target = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)

            self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
            status, actual = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)

            if (target == actual) and (target == 0):
                home_status = True
            else:
                home_status = False
            self.mssg = 'TrinamicBoard.home_status('+axis+') succeeded'            
            return home_status
        else:
            self.mssg = 'TrinamicBoard.home_status('+axis+') inactive'            
            return False

    # return True if current position and target position are the same
    def target_status(self, axis):
        self.mssg = 'TrinamicBoard.target_status('+axis+')'            
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
            status, target = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)

            self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
            status, actual = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)

            self.mssg = 'TrinamicBoard.target_status('+axis+') succeeded'            
            return (target == actual)
        else:
            self.mssg = 'TrinamicBoard.target_status('+axis+') inactive'            
            return False

    # Get reference (limit) switch status (1=active)
    def get_limit_status(self, axis):
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.limit_status[axis], 0x00000000)
            status, data = self.SPI_write(self.chip_pin[axis], self.limit_status[axis], 0x00000000)

            # print('raw:', data)
            # data = bin(data) # convert returned data to binary
            # print('bin':, data)
            left = False     # convert bit 0 to boolean
            right = False    # convert bit 1 to boolean
            return left, right    # return left and right limit switch status' 
        else:
            self.mssg = 'TrinamicBoard.get_limit_status('+axis+') inactive'            
            return 0

    # Get target position
    def target_pos(self, axis):
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
            status, pos = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
            if axis == 'Z':
                um = self.z_ustep2um(pos)
            else:
                um = self.xy_ustep2um(pos)

            # print('read_target:', um)
            self.mssg = 'TrinamicBoard.target_pos('+axis+') succeeded'            
            return um
        else:
            self.mssg = 'TrinamicBoard.target_pos('+axis+') inactive'            
            return 0

    # Get current position
    def current_pos(self, axis):
        if self.found:
                self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
                status, pos = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
                if axis == 'Z':
                    um = self.z_ustep2um(pos)
                else:
                    um = self.xy_ustep2um(pos)

                # print('read_actual:', um)
                self.mssg = 'TrinamicBoard.current_pos('+axis+') succeeded'            
                return um
        else:
            self.mssg = 'TrinamicBoard.current_pos('+axis+') inactive'            
            return 0

    # Move to absolute position (in um)
    def move_abs_pos(self, axis, pos):

        if self.found:
            
            if axis == 'Z':

                # don't let it move out of bounds
                if pos < 0:
                    pos = 0.
                elif pos > 12000:
                    pos = 12000.

                steps = self.z_um2ustep(pos)
            else:
                steps = self.xy_um2ustep(pos)

            # signed to unsigned 32_bit integer
            if steps < 0:
                steps = 4294967296+steps

            if axis=='Z': # perform overshoot to always come from one direction

                # get current position
                current = self.current_pos('Z')

                # if the current position is above the new target position
                if current > pos:
                    self.overshoot = True
                    # First overshoot downwards
                    overshoot = self.z_um2ustep(pos-30) # target minus 30 um
                    self.SPI_write (self.chip_pin[axis], self.write_target[axis], overshoot)
                    while not self.target_status('Z'):
                        time.sleep(0.05)
                    self.overshoot = False

            self.SPI_write (self.chip_pin[axis], self.write_target[axis], steps)
            self.mssg = 'TrinamicBoard.move_abs_pos('+axis+','+str(pos)+') succeeded'
        else:
            self.mssg = 'TrinamicBoard.move_abs_pos('+axis+','+str(pos)+') inactive'


    # Move by relative distance (in um)
    def move_rel_pos(self, axis, um):
        if self.found:
            # Read actual position in um
            pos = self.current_pos(axis)
 
            # Add relative motion and convert
            if axis == 'Z':
                pos = um+pos
                if pos < 0:
                    pos = 0.
                elif pos > 12000:
                    pos = 12000.     

                steps = self.z_um2ustep(pos)
            else:
                steps = self.xy_um2ustep(um+pos)

             # signed to unsigned 32_bit integer
            if steps < 0:
                steps = 4294967296+steps

            if axis=='Z': # perform overshoot to always come from one direction

                # if the movement is downward or backward
                if um < 0:
                    self.overshoot = True
                    # First overshoot downwards
                    overshoot = self.z_um2ustep(pos-30) # target minus 30 um
                    self.SPI_write (self.chip_pin[axis], self.write_target[axis], overshoot)
                    while not self.target_status('Z'):
                        time.sleep(0.05)
                    self.overshoot = False
                    
            # print('pos:', pos, 'um:', um, 'pos+um:', um+pos, 'steps:', steps)
            self.SPI_write (self.chip_pin[axis], self.write_target[axis], steps)
            self.mssg = 'TrinamicBoard.move_rel_pos('+axis+','+str(um)+') succeeded'
        else:
            self.mssg = 'TrinamicBoard.move_rel_pos('+axis+','+str(um)+') inactive'



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