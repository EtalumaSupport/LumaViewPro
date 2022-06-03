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
June 3, 2022
'''

from mcp2210 import Mcp2210, Mcp2210GpioDesignation, Mcp2210GpioDirection
import struct    # For making c style data structures, and send them through the mcp chip
import sched

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


    #----------------------------------------------------------
    # Initialize connection through SPI
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        # USB\VID_04D8&PID_00DE\0001006900
        self.chip = Mcp2210('0001006900')  # the serial number of the chip. Can be determined by using dmesg after plugging in the device
        print ("Found Device", self.chip)

        # set all GPIO lines on the MCP2210 as General GPIO lines
        # Tri-State all GPIO lines on the MCP2210
        for i in range(9):
            self.chip.set_gpio_designation(i, Mcp2210GpioDesignation.GPIO)
            self.chip.set_gpio_direction(i, Mcp2210GpioDirection.INPUT)

        self.configure()

    #----------------------------------------------------------
    # Define SPI communication
    #----------------------------------------------------------

    def SPI_write (self, pin_number: int, addr: int, data: int):
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
        # Load settings for the xy motors from config file for 'XY' motor driver
        with open('./data/xymotorconfig.ini', 'r') as xyconfigFile:
            for line in xyconfigFile:                           # go through each line of the config file
                if 'writing' in line:                           # if the line has the word "writing" in it (all the right lines have this)
                    configLine = line.split()                   # split the line into an indexable list
                    addr = int(0x80 | int(configLine[0], 16))   # take the address field, format it, add 80 to it (to make it a write operation to the trinamic chip), throw it into the 'address' variable
                    data = int(configLine[2], 16)               # take the data field, format it, and put it into the 'data' variable
                    self.SPI_write(0, addr, data)                # call SPI Writer. right now were writing to chip 0
        print ("Sucessfully loaded parameters to XY motor driver from xymotorconfig.ini")

        # Load settings for the z motor
        with open('./data/ztmotorconfig.ini', 'r') as ztconfigFile:
            for line in ztconfigFile:
                if 'writing' in line:
                    configLine = line.split()
                    addr = int(0x80 | int(configLine[0], 16))
                    data = int(configLine[2], 16)
                    self.SPI_write(1, addr, data)
        print ("Sucessfully loaded parameters to ZT motor driver from zmotorconfig.ini")

    #----------------------------------------------------------
    # Z (Focus) Functions
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        um = 0.0059 * ustep # 0.0059 um/ustep Olympus Z
        return um

    def z_um2ustep(self, um):
        ustep = int( um / 0.0059 ) # 0.0059 um/ustep Olympus Z
        return ustep

    def zhome(self):
        self.move_abs_pos('Z', -1000000)

        # Schedule writing target and actual Z position to 0 after 4 seconds
        s = sched.scheduler()
        zevent = s.enter(4, 1, self.zhome_write)
        s.run()

    def zhome_write(self):
        self.SPI_write (self.chip_pin['Z'], self.write_actual['Z'], 0x00000000)
        self.SPI_write (self.chip_pin['Z'], self.write_target['Z'], 0x00000000)

    #----------------------------------------------------------
    # XY Stage Functions
    #----------------------------------------------------------
    def xy_ustep2um(self, ustep):
        um = 0.0496 * ustep # 0.0496 um/ustep Heidstar XY
        return um

    def xy_um2ustep(self, um):
        ustep = int( um / 0.0496) # 0.0496 um/ustep Heidstar XY
        return ustep

    def xyhome(self):
        self.zhome()

        self.move_abs_pos('X', -1000000)
        self.move_abs_pos('Y', -1000000)

        # Schedule writing target and actual XY positions to 0 after 8 seconds
        s = sched.scheduler()
        xyevent = s.enter(8, 2, self.xyhome_write)
        s.run()

    def xyhome_write(self):
        self.SPI_write(self.chip_pin['X'], self.write_actual['X'], 0x00000000)
        self.SPI_write(self.chip_pin['X'], self.write_target['X'], 0x00000000)

        self.SPI_write(self.chip_pin['Y'], self.write_actual['Y'], 0x00000000)
        self.SPI_write(self.chip_pin['Y'], self.write_target['Y'], 0x00000000)

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------
    # # Get reference switch status (True -> reference is currently being saught,
    # #                              False -> reference is not currently being saught)
    # # Modified for for any target (TMC5072 does not have a proper limit status)
    # def limit_status(self, axis):
    #     self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
    #     status, target = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
    #
    #     self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
    #     status, actual = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
    #
    #     ref_status = not(target == actual)
    #     print('target status:', ref_status)
    #     return ref_status

    # return True if current and target position are at home.
    def home_status(self, axis):
        self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
        status, target = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)

        self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
        status, actual = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)

        if (target == actual) and (target == 0):
            home_status = True
        else:
            home_status = False

        # print(home_status)
        return home_status

    # return True if current position and target position are the same
    def target_status(self, axis):
        self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
        status, target = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)

        self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
        status, actual = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)

        #print(target == actual)
        return (target == actual)

    # For backward compatibility
    def limit_status(self, axis):
        print("This board does not support 'limit_status'")
        return False

    # Get target position
    def target_pos(self, axis):
        self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
        status, pos = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
        if axis == 'Z':
            um = self.z_ustep2um(pos)
        else:
            um = self.xy_ustep2um(pos)

        # print('read_target:', um)
        return um

    # Get current position
    def current_pos(self, axis):
        self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
        status, pos = self.SPI_write (self.chip_pin[axis], self.read_actual[axis], 0x00000000)
        if axis == 'Z':
            um = self.z_ustep2um(pos)
        else:
            um = self.xy_ustep2um(pos)

        # print('read_actual:', um)
        return um

    # Move to absolute position (in um)
    def move_abs_pos(self, axis, pos):
        if axis == 'Z':
            steps = self.z_um2ustep(pos)
        else:
            steps = self.xy_um2ustep(pos)
        # signed to unsigned 32_bit integer
        if steps < 0:
            steps = 4294967296+steps

        # print('steps:', steps, '\t pos:', pos)
        self.SPI_write (self.chip_pin[axis], self.write_target[axis], steps)

    # Move by relative distance (in um)
    def move_rel_pos(self, axis, um):
        # Read actual position in um
        pos = self.current_pos(axis)
        # Add relative motion and convert
        if axis == 'Z':
            steps = self.z_um2ustep(um+pos)
        else:
            steps = self.xy_um2ustep(um+pos)
        # signed to unsigned 32_bit integer
        if steps < 0:
            steps = 4294967296+steps

        # print('steps:', steps, '\t um:', um+pos)
        self.SPI_write (self.chip_pin[axis], self.write_target[axis], steps)



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
