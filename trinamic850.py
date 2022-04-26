from mcp2210 import Mcp2210, Mcp2210GpioDesignation, Mcp2210GpioDirection
import struct    # For making c style data structures, and send them through the mcp chip

class TrinamicBoard:

    axis_pin = {
        'X': 0,
        'Y': 0,
        'Z': 1,
        'T': 1
    }
    r_actual = {
        'X': 0x21,
        'Y': 0x41,
        'Z': 0x41,
        'T': 0x21
    }
    w_actual = {
        'X': 0xA1,
        'Y': 0xC1,
        'Z': 0xC1,
        'T': 0xA1
    }
    r_target = {
        'X': 0x2D,
        'Y': 0x4D,
        'Z': 0x4D,
        'T': 0x2D
    }
    w_target = {
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

        for i in range(9):
            self.chip.set_gpio_designation(i, Mcp2210GpioDesignation.GPIO) # set all GPIO lines on the mcp2210 as General GPIO lines
        for i in range(9):
            self.chip.set_gpio_direction(i, Mcp2210GpioDirection.INPUT) # Tri-State all GPIO lines on the mcp2210

        self.configure()

    #----------------------------------------------------------
    # Define SPI communication: Manny's handy dandy spi writer.
    #----------------------------------------------------------
    def SPIWRITE (self, pin_number: int, addr: int, data: int):
        # set the correct pin as chip select. only one chip select line at a time, or else you will have data conflicts on receive!!!
        self.chip.set_gpio_designation(pin_number, Mcp2210GpioDesignation.CHIP_SELECT)
        # glue the address and data lines together.  make it big-endian
        tx_data = struct.pack('>BI', addr, data)
        # squirt the config word down to the correct chip,
        # put anything coming back from the previous write cycle into rx_data
        rx_data = self.chip.spi_exchange(tx_data, cs_pin_number=pin_number)
        # turn the Chip Select pin back to a Tri-state input pin.
        self.chip.set_gpio_designation(pin_number, Mcp2210GpioDesignation.GPIO)
        return rx_data

    #----------------------------------------------------------
    # Import motor configurations
    #----------------------------------------------------------
    def configure(self):
        # Load settings for the xy motors
        with open('./data/xymotorconfig.ini', 'r') as xyconfigFile: # open config file for 'XY' motor driver
            for line in xyconfigFile: # go through each line of the config file
                if 'writing' in line: # if the line has the word "writing" in it (all the right lines have this)
                    configLine = line.split() #split the line into an indexable list
                    addr = int(0x80 | int(configLine[0], 16))#take the address field, format it, add 80 to it (to make it a write operation to the trinamic chip), throw it into the 'address' variable
                    data = int(configLine[2], 16) # take the data field, format it, and put it into the 'data' variable
                    self.SPIWRITE(0, addr, data) # call Manny's handy dandy spi writer. right now were writing to chip 0
        print ("sucessfully loaded parameters to XY motor driver from xymotorconfig.ini")

        # Load settings for the z motor
        with open('./data/ztmotorconfig.ini', 'r') as ztconfigFile: #same as before, but using the config file for 'ZT' motor driver
            for line in ztconfigFile:
                if 'writing' in line:
                    configLine = line.split()
                    addr = int(0x80 | int(configLine[0], 16))
                    data = int(configLine[2], 16)
                    self.SPIWRITE(1, addr, data)
        print ("sucessfully loaded parameters to XY motor driver from xymotorconfig.ini")

    #----------------------------------------------------------
    # Z (Focus) Functions
    #----------------------------------------------------------
    def z_ustep2um(self, ustep):
        pass

    def z_um2ustep(self, um):
        pass

    def zhome(self):
        pass

    #----------------------------------------------------------
    # XY Staage Functions
    #----------------------------------------------------------
    def xy_ustep2um(self, ustep):
        pass

    def xy_um2ustep(self, um):
        pass

    def xyhome(self):
        pass

    #----------------------------------------------------------
    # Motion Functions
    #----------------------------------------------------------
    # Get reference switch status (True -> reference is currently being saught,
    #                              False -> reference is not currently being saught)
    def limit_status(self, axis):
        pass

    # Get target position
    def target_pos(self, axis):
        rx = self.SPIWRITE (self.axis_pin[axis], self.r_target[axis], 0x00000000)
        print(list(rx))

    # Get current position
    def current_pos(self, axis):
        rx = self.SPIWRITE (self.axis_pin[axis], self.r_actual[axis], 0x00000000)
        print(list(rx))

    # Move to absolute position
    def move_abs_pos(self, axis, pos):
        pass

    # Move by relative distance
    def move_rel_pos(self, axis, um):
        # currently using steps
        steps = um
        # signed to unsigned 32_bit integer
        if steps < 0:
            steps = 4294967296+steps

        rx = self.SPIWRITE (self.axis_pin[axis], self.w_target[axis], steps)
        print(list(rx))
