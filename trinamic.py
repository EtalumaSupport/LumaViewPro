import serial
import serial.tools.list_ports as list_ports

class TrinamicBoard:
    # Trinamic 3230 board (preferred)
    # Trinamic 6110 board (now) Address
    z_microstep = 0.05489864865 # microns per microstep at 32x microstepping
                                # via Eric Weiner 6/24/2021
    xy_microstep = 0.15625      # microns per microstep at 32x microstepping

    addr = 255

    # Commmand set available in direct mode
    cmnd = {
        'ROR':1,    # Rotate Right
        'ROL':2,    # Rotate Left
        'MST':3,    # Motor Stop
        'MVP':4,    # Move to Position
        'SAP':5,    # Set Axis Parameter
        'GAP':6,    # Get Axis Parameter
        'STAP':7,   # Store Axis Parameter
        'RSAP':8,   # Restore Axis Parameter
        'SGP':9,    # Set Global Parameter
        'GGP':10,   # Get Global Parameter
        'STGP':11,  # Store Global Parameter
        'RSGP':12,  # Restore Global Parameter
        'RFS':13,   # Reference Search
        'SIO':14,   # Set Output
        'GIO':15,   # Get Input / Output
        'SCO':30,   # Set Coordinate
        'GCO':31,   # Get Coordinate
        'CCO':32    # Capture Coordinate
    }

    axis = {
        'X':0, # Left to Right is positive
        'Y':1, # Front to Back is positive
        'Z':2  # Down is positive
    }

    #----------------------------------------------------------
    # Set up Serial Port connection
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        # find all available serial ports
        ports = serial.tools.list_ports.comports(include_links = True)
        for port in ports:
            if (port.vid == 10812) and (port.pid == 256):
                print('Trinamic Motor Control Board identified at', port.device)
                self.port = port.device

        self.reset = 0
        self.baudrate = 9600
        self.bytesize = serial.EIGHTBITS
        self.parity = serial.PARITY_NONE
        self.stopbits = serial.STOPBITS_ONE
        self.timeout = 5 # seconds
        self.driver = True
        try:
            self.connect()
        except:
            print("Could not connect to Trinamic Motor Control Board")
            self.driver = False

    def __del__(self):
        if self.driver != False:
            self.driver.close()

    def connect(self):
        self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)

        # check user variable not stored in EEPROM
        self.reset = self.SendGram('GGP', 255, 'Z', 0) # This is bank 2 ('Z' is just to get at it)
        if self.reset == 0:
            print('Homing Z-axis')
            self.zhome()
            self.reset = self.SendGram('SGP', 255, 'Z', 1) # This is bank 2 ('Z' is just to get at it)

        self.driver.close()
        self.driver.open()

    #----------------------------------------------------------
    # Receive Datagram
    #----------------------------------------------------------
    def GetGram(self, verbose = True):

        # receive the datagram
        datagram = self.driver.read(9)

        checksum = 0
        for i in range(8):         # compare checksum
            checksum =  (checksum + datagram[i]) & 0xff

        if checksum != datagram[8]:
            print('Return Checksum Error')
            return False

        Address=datagram[0]
        Status=datagram[2]
        if Status != 100:
            print('Return Status Error')
        Value = int.from_bytes(datagram[4:8], byteorder='big', signed=True)
        if verbose == True:
            print('Status: ', Status)
            print('Value:  ', Value)
        return Value

    #----------------------------------------------------------
    # Send Datagram
    #----------------------------------------------------------
    def SendGram(self, Command, Type, Motor, Value):

        datagram = bytearray(9)
        datagram[0] = self.addr
        datagram[1] = self.cmnd[Command]
        datagram[2] = Type
        datagram[3] = self.axis[Motor]
        datagram[4] = (Value >> 24) & 0xff # shift by 24 bits i.e. divide by 2, 24 times
        datagram[5] = (Value >> 16) & 0xff # shift by 16 bits i.e. divide by 2, 16 times
        datagram[6] = (Value >> 8) & 0xff  # shift by 8 bits i.e. divide by 2, 8 times
        datagram[7] = Value & 0xff # bitwise and with 0xff to get last 8 byte

        for i in range(8):         # generate checksum
            datagram[8] =  (datagram[8] + datagram[i]) & 0xff

        if self.driver != False:
            self.driver.write(datagram)
            return self.GetGram(verbose = False)
        else:
            # print('Trinamic Motor Control Board is not connected')
            return False

    def z_ustep2um(self, ustep):
        um = float(ustep)*self.z_microstep
        return um

    def z_um2ustep(self, um):
        ustep = int(float(um)/self.z_microstep)
        return ustep

    def zhome(self):
        #----------------------------------------------------------
        # Z-Axis Initialization
        #----------------------------------------------------------
        self.SendGram('SAP', 6, 'Z', 16)      # Maximum current
        self.SendGram('SAP', 7, 'Z', 8)       # Standby current
        self.SendGram('SAP', 2, 'Z', 1000)    # Target Velocity
        self.SendGram('SAP', 5, 'Z', 500)     # Acceleration
        self.SendGram('SAP', 140, 'Z', 5)     # 32X Microstepping
        self.SendGram('SAP', 153, 'Z', 9)     # Ramp Divisor 9
        self.SendGram('SAP', 154, 'Z', 3)     # Pulse Divisor 3
        self.SendGram('SAP', 163, 'Z', 0)     # Constant TOff Mode (spreadcycle)


        # Parameters for Limit Switches
        #----------------------------------------------------------
        self.SendGram('SAP', 12, 'Z', 0)      # enable Right Limit switch
        self.SendGram('SAP', 13, 'Z', 0)      # enable Left Limit switch

        # Parameters for Homing
        #----------------------------------------------------------
        self.SendGram('SAP', 193, 'Z', 65)    # Search Right Stop Switch (Down)
        self.SendGram('SAP', 194, 'Z', 1000)  # Reference search speed
        self.SendGram('SAP', 195, 'Z', 10)    # Reference switch speed (was 10X less than search speed in LumaView)

        # Start the Trinamic Homing Procedure
        #----------------------------------------------------------
        self.SendGram('RFS', 0, 'Z', 0)       # Home to the Right Limit switch (Down)

    # DEBUG: What was this for?
    #     self.zhome_event = Clock.schedule_interval(self.check_zhome, 0.05)
    #
    # # Test if reference function is complete (z homing)
    # def check_zhome(self, dt):
    #     value = self.SendGram('RFS', 2, 'Z', 0)
    #     if value == 0:
    #         Clock.unschedule(self.zhome_event)


    def xy_ustep2um(self, ustep):
        um = float(ustep)*self.xy_microstep
        return um

    def xy_um2ustep(self, um):
        ustep = int(um/self.xy_microstep)
        return ustep

    def xyhome(self):

        self.zhome()
        #----------------------------------------------------------
        # X-Axis Initialization
        #----------------------------------------------------------
        self.SendGram('SAP', 6, 'X', 16)     # Maximum current
        self.SendGram('SAP', 7, 'X', 8)      # standby current
        self.SendGram('SAP', 2, 'X', 1000)   # Target Velocity
        self.SendGram('SAP', 5, 'X', 500)    # Acceleration
        self.SendGram('SAP', 140, 'X', 5)    # 32X Microstepping
        self.SendGram('SAP', 153, 'X', 9)    # Ramp Divisor 9
        self.SendGram('SAP', 154, 'X', 3)    # Pulse Divisor 3
        self.SendGram('SAP', 163, 'X', 0)    # Constant TOff Mode (spreadcycle)

        # Parameters for Limit Switches
        #----------------------------------------------------------
        self.SendGram('SAP', 12, 'X', 0)     # enable Right Limit switch
        self.SendGram('SAP', 13, 'X', 0)     # enable Left Limit switch

        # Parameters for Homing
        #----------------------------------------------------------
        # 'Set Axis Parameter', 'Reference Search Mode', 'X-axis', '1+64 = Search Right Stop Switch Only'
        self.SendGram('SAP', 193, 'X', 65)   # Search Right Stop switch Only
        self.SendGram('SAP', 194, 'X', 1000) # Reference search speed
        self.SendGram('SAP', 195, 'X', 10)   # Reference switch speed (was 10X less than search speed in LumaView)

        # Start the Trinamic Homing Procedure
        # ----------------------------------------------------------
        self.SendGram('RFS', 0, 'X', 0)      # Home to the Right Limit switch (Right)

        #----------------------------------------------------------
        # Y-Axis Initialization
        #----------------------------------------------------------
        self.SendGram('SAP', 6, 'Y', 16)    # Maximum current
        self.SendGram('SAP', 7, 'Y', 8)     # Standby current
        self.SendGram('SAP', 2, 'Y', 1000)  # Target Velocity
        self.SendGram('SAP', 5, 'Y', 500)   # Acceleration
        self.SendGram('SAP', 140, 'Y', 5)   # 32X Microstepping
        self.SendGram('SAP', 153, 'Y', 9)   # Ramp Divisor 9
        self.SendGram('SAP', 154, 'Y', 3)   # Pulse Divisor 3
        self.SendGram('SAP', 163, 'Y', 0)   # Constant TOff Mode (spreadcycle)

        # Parameters for Limit Switches
        #----------------------------------------------------------
        self.SendGram('SAP', 12, 'Y', 0)    # enable Right Limit switch
        self.SendGram('SAP', 13, 'Y', 0)    # enable Left Limit switch

        # Parameters for Homing
        #----------------------------------------------------------
        self.SendGram('SAP', 193, 'Y', 65)   # Search Right Stop switch Only (Back)
        self.SendGram('SAP', 194, 'Y', 1000) # Reference search speed
        self.SendGram('SAP', 195, 'Y', 10)   # Reference switch speed (was 10X less than search speed in LumaView)

        # Start the Trinamic Homing Procedure
        #----------------------------------------------------------
        self.SendGram('RFS', 0, 'Y', 0)      # Home to the Right Limit switch (Back)

    # DEBUG: What was this for?
    #     self.xyhome_event = Clock.schedule_interval(self.check_xyhome, 0.05)
    #
    # # Test if reference function is complete (xy homing)
    # def check_xyhome(self, dt):
    #     x = self.SendGram('RFS', 2, 'X', 0)
    #     y = self.SendGram('RFS', 2, 'Y', 0)
    #     if (x == 0) and (y == 0):
    #         Clock.unschedule(self.xyhome_event)

    # Get reference switch status (True -> reference is currently being saught,
    #                              False -> reference is not currently being saught)
    def limit_status(self, axis):
        switch = self.SendGram('RFS', 2, axis, 0)
        if switch != 0:
            return True
        else:
            return False

    # Get target position
    def target_pos(self, axis):
      target = self.SendGram('GAP', 0, axis, 10)
      if axis == 'Z':
          pos = -self.z_ustep2um(target)
      else:
          pos = -self.xy_ustep2um(target)
      return pos

    # Get current position
    def current_pos(self, axis):
      current = self.SendGram('GAP', 1, axis, 0)
      if axis == 'Z':
          pos = -self.z_ustep2um(current)
      else:
          pos = -self.xy_ustep2um(current)
      return pos

    # Move to absolute position
    def move_abs_pos(self, axis, pos):
      if axis == 'Z':
          steps = -self.z_um2ustep(pos)
      else:
          steps = -self.xy_um2ustep(pos)
      self.SendGram('MVP', 0, axis, steps)

    # Move by distance (UP is positive, DOWN in negative)
    def move_rel_pos(self, axis, um):
      if axis == 'Z':
          steps = -self.z_um2ustep(um)
      else:
          steps = -self.xy_um2ustep(um)
      self.SendGram('MVP', 1, axis, steps)
