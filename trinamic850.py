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
# import sched
import threading
import queue
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
    write_actual = { # Added 0x80 to the addressess for write access
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
    write_target = { # Added 0x80 to the addressess for write access
        'X': 0xAD,
        'Y': 0xCD,
        'Z': 0xCD,
        'T': 0xAD
    }
    ref_status = { # reference status register
        # bit 0, left status (1=active)
        # bit 1, right status (1=active)
        # bit 9, position reached (1 = True)
        # See section 6.2.2.2 for remaining bits
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
        self.message = 'TrinamicBoard.__init__()'
        self.overshoot = False
        self.backlash = 25 # um of additional downlaod travel in z for drive hysterisis

        try:
            if platform == 'win32':
                devices = hid.core.find_all_hid_devices()
                for device in devices:
                    if device.vendor_id == 0x04D8 and device.product_id == 0x00DE:
                        SPI_serial = device.serial_number
            else:
                device = usb.core.find(idVendor=0x04d8, idProduct=0x00de)  # find the mcp2210 USB device, idVendor=0x04d8, idProduct=0x00de is defaulted in the chip
                SPI_serial = usb.util.get_string(device, device.iSerialNumber ) # find the serial number of the MCP device
            
            self.chip = Mcp2210(SPI_serial)  # the serial number of the chip. Can be determined by using dmesg after plugging in the device
            print ("Found Device", self.chip)

            # set all GPIO lines on the MCP2210 as General GPIO lines
            # Tri-State all GPIO lines on the MCP2210
            for i in range(9):
                self.chip.set_gpio_designation(i, Mcp2210GpioDesignation.GPIO)
                self.chip.set_gpio_direction(i, Mcp2210GpioDirection.INPUT)

        except:
            self.message = 'TrinamicBoard.__init__() failed'         
            self.found = False
        
        else:
            self.configure()
            self.message = 'TrinamicBoard.__init__() succeeded'
            self.found = True       


    #----------------------------------------------------------
    # Define SPI communication
    #----------------------------------------------------------
    def SPI_put (self, pin: int, address: int, data: int, callback):
        if self.buffer == False:
            self.buffer = queue.Queue(1024)  # could add buffer size to settings someday
        
        if self.buffer.full():
           print("Trinamic SPI write buffer size is "+self.buffer.qsize()+" and is full.")
           return
        
        self.buffer.put({
                pin:      pin,
                address:  address,
                data:     data,
                callback: callback
            })


    def SPI_get (self):
        if self.buffer == False:
            return
        
        cmd = self.buffer.get()
        spi_status, data = self.SPI_write(cmd.pin, cmd.address, cmd.data)
        cmd.callback(spi_status = spi_status, data = data)


    def SPI_write (self, pin: int, address: int, data: int):
        try:   
            # Set the correct pin as chip select. Only one chip select line at a time,
            # or else you will have data conflicts on receive.
            self.chip.set_gpio_designation(pin, Mcp2210GpioDesignation.CHIP_SELECT)

            # glue the addressess and data lines together.  make it big-endian.
            tx_data = struct.pack('>BI', address, data)

            # squirt the config word down to the correct chip,
            # put anything coming back from the previous write cycle into rx_data
            rx_data = self.chip.spi_exchange(tx_data, cs_pin=pin)

            # turn the Chip Select pin back to a Tri-state input pin.
            self.chip.set_gpio_designation(pin, Mcp2210GpioDesignation.GPIO)

        except:
            self.message = 'TrinamicBoard.SPI_write() unsuccesful'
            self.found = False
            spi_status = 0
            data = 0
        
        else:
            # decode 40 bit returned squirt from trinamic TMC5072 chip
            self.message = 'TrinamicBoard.SPI_write() succesful'
            self.found = True
            spi_status = bin(rx_data[0])
            data = int.from_bytes(rx_data[1:5], byteorder='big', signed=True)

        return spi_status, data


    #----------------------------------------------------------
    # Import motor configurations
    #----------------------------------------------------------
    def configure(self):
        self.message = 'TrinamicBoard.configure()'            
        # Load settings for the xy motors from config file for 'XY' motor driver
        try:
            with open('./data/xymotorconfig.ini', 'r') as xyconfigFile:
                for line in xyconfigFile:                           # go through each line of the config file
                    if 'writing' in line:                           # if the line has the word "writing" in it (all the right lines have this)
                        configLine = line.split()                   # split the line into an indexable list
                        address = int(0x80 | int(configLine[0], 16))   # take the addressess field, format it, add 80 to it (to make it a write operation to the trinamic chip), throw it into the 'addressess' variable
                        data = int(configLine[2], 16)               # take the data field, format it, and put it into the 'data' variable
                        self.SPI_write(0, address, data)                # call SPI Writer. right now were writing to chip 0
        except:
            print ("Unable to load parameters to XY motor driver from file xymotorconfig.ini")
        else:
            print ("Successfully loaded parameters to XY motor driver from xymotorconfig.ini")

        # Load settings for the z motor
        try:
            with open('./data/ztmotorconfig.ini', 'r') as ztconfigFile:
                for line in ztconfigFile:
                    if 'writing' in line:
                        configLine = line.split()
                        address = int(0x80 | int(configLine[0], 16))
                        data = int(configLine[2], 16)
                        self.SPI_write(1, address, data)
        except:
            print ("Unable to load parameters to ZT motor driver from file ztmotorconfig.ini")
        else:
            print ("Successfully loaded parameters to ZT motor driver from ztmotorconfig.ini")

    def move(self, axis, steps):
        if steps < 0:
            steps += 0x100000000
        self.SPI_write(self.chip_pin[axis], self.write_target[axis], steps)
        time.sleep(0.1)

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
        self.message = 'TrinamicBoard.zhome()'            
        if self.found:
            # self.move_abs_pos('Z', -1000000)
            # self.SPI_write(self.chip_pin['Z'], self.write_target['Z'], 4294967296-2000000000)
            self.move('Z', -2000000000)

            # Start a thread to check if Z limit is active
            z_thread = threading.Thread(target=self.zhome_write)
            z_thread.start()

    def zhome_write(self):
        self.message = 'TrinamicBoard.zhome_write()'

        for i in range(100):
            time.sleep(0.1)       
            if self.home_status('Z'):
                self.SPI_write (self.chip_pin['Z'], self.write_actual['Z'], 0x00000000)
                self.SPI_write (self.chip_pin['Z'], self.write_target['Z'], 0x00000000)
                print('Z at Home')
                return

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

    def xyhome(self): # Must move zhome first but threading causes communication challenge
        self.message = 'TrinamicBoard.xyhome()'            
        if self.found:

            # self.move_abs_pos('X', -1000000)
            # self.move_abs_pos('Y', -1000000)
            #self.SPI_write(self.chip_pin['X'], self.write_target['X'], 4294967296-2000000000)
            #self.SPI_write(self.chip_pin['Y'], self.write_target['Y'], 4294967296-2000000000)
            #self.SPI_write(self.chip_pin['Z'], self.write_target['Z'], 4294967296-2000000000)
            self.move('X', -2000000000)
            self.move('Y', -2000000000)
            self.move('Z', -2000000000)

            # Start a thread to check if XY limits are active
            xy_thread = threading.Thread(target=self.xyhome_write) 
            xy_thread.start()

    def xyhome_write(self):
        self.message = 'TrinamicBoard.xyhome_write()'

        # Home Z first
        for i in range(100):
            time.sleep(0.1) # in a thread therefore should not be disruptive       
            if self.home_status('Z'):
                self.SPI_write (self.chip_pin['Z'], self.write_actual['Z'], 0x00000000)
                self.SPI_write (self.chip_pin['Z'], self.write_target['Z'], 0x00000000)
                print('Z at Home')
                break

        # Home XY second
        for i in range(200):
            time.sleep(0.1) # in a thread therefore should not be disruptive
            if self.home_status('X') and self.home_status('Y'):        
                self.SPI_write(self.chip_pin['X'], self.write_actual['X'], 0x00000000)
                self.SPI_write(self.chip_pin['X'], self.write_target['X'], 0x00000000)

                self.SPI_write(self.chip_pin['Y'], self.write_actual['Y'], 0x00000000)
                self.SPI_write(self.chip_pin['Y'], self.write_target['Y'], 0x00000000)
                print('XY at Home')
                break

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
        self.message = 'TrinamicBoard.thome()'            
        if self.found:

            # TODO enable home switch

            # move to home and beyond
            self.move_abs_pos('T', -1000000)

            # Start a thread to check if Z limit is active
            t_thread = threading.Thread(target=self.thome_write)
            t_thread.start()

    def thome_write(self):
        self.message = 'TrinamicBoard.thome_write()'            

        for i in range(100):
            time.sleep(0.1) # in a thread therefore should not be disruptive       
            if self.home_status('T'):
                self.SPI_write (self.chip_pin['T'], self.write_actual['T'], 0x00000000)
                self.SPI_write (self.chip_pin['T'], self.write_target['T'], 0x00000000)
                # TODO disable home switch
                return

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------
 
    # Get target position
    def target_pos(self, axis):
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
            status, pos = self.SPI_write (self.chip_pin[axis], self.read_target[axis], 0x00000000)
            if axis == 'Z':
                um = self.z_ustep2um(pos)
            else:
                um = self.xy_ustep2um(pos)

            self.message = 'TrinamicBoard.target_pos('+axis+') succeeded'            
            return um
        else:
            self.message = 'TrinamicBoard.target_pos('+axis+') inactive'            
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

            self.message = 'TrinamicBoard.current_pos('+axis+') succeeded'            
            return um
        else:
            self.message = 'TrinamicBoard.current_pos('+axis+') inactive'            
            return 0

    # Move to absolute position (in um)
    def move_abs_pos(self, axis, pos):
        if self.found:
            if axis == 'Z': # Z bound 0 to 14mm
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
            # signed to unsigned 32_bit integer
            if steps < 0:
                #steps = 4294967296+steps # TODO change to correct casting operation
                steps = 0x100000000+steps # TODO REMOVE this code block should no longer be nessesary and shoudl be removed after testing

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

            #self.SPI_write (self.chip_pin[axis], self.write_target[axis], steps)
            self.move(axis, steps)
            self.message = 'TrinamicBoard.move_abs_pos('+axis+','+str(pos)+') succeeded'
        else:
            self.message = 'TrinamicBoard.move_abs_pos('+axis+','+str(pos)+') inactive'


    # Move by relative distance (in um)
    def move_rel_pos(self, axis, um):
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
        self.message = 'TrinamicBoard.home_status('+axis+')'            
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.ref_status[axis], 0x00000000)
            status, data = self.SPI_write(self.chip_pin[axis], self.ref_status[axis], 0x00000000)

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

        self.message = 'TrinamicBoard.target_status('+axis+')'  
        
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.ref_status[axis], 0x00000000)
            status, data = self.SPI_write(self.chip_pin[axis], self.ref_status[axis], 0x00000000)

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
        if self.found:
            self.SPI_write (self.chip_pin[axis], self.ref_status[axis], 0x00000000)
            status, data = self.SPI_write(self.chip_pin[axis], self.ref_status[axis], 0x00000000)

            # data is an integer that represents 4 bytes, or 32 bits, largest bit first
            '''
            bit: 33222222222211111111110000000000
            bit: 10987654321098765432109876543210
            bit: ----------------------*-------**
            '''
            bits = format(data, 'b').zfill(32)
            return bits
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