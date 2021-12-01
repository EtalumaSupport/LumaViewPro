import serial

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
