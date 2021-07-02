# Create, Send, and Receive commands (datagrams) to a Trinamic Motion Board
# Using direct mode and Serial communication

# Converted from
# https://www.trinamic.com/fileadmin/assets/Support/Software/TMCLDatagram.c

# Trinamic 6110 board (now)
# Trinamic 3230 board (preferred)
address = 255

commands = {
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
    'CCO':32   # Capture Coordinate
}

motors = {
    'X':0,
    'Y':1,
    'Z':2
}

# Set up Serial Port connection
import serial
class motorport:
    def __init__(self, **kwargs):

        self.port = 'COM11'
        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=5 # seconds
        self.connect()

    def __del__(self):
        self.driver.close()

    def connect(self):
        try:
            self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)
            self.driver.close()
            self.driver.open()
        except:
            print("Did not connect to Trinamic Motor Control Board")

#----------------------------------------------------------
# Generate Datagram
#----------------------------------------------------------
def MakeGram(Address, Command, Type, Motor, Value):

    datagram = bytearray(9)
    datagram[0] = Address
    datagram[1] = Command
    datagram[2] = Type
    datagram[3] = Motor
    datagram[4] = Value >> 24  # shift by 24 bits i.e. divide by 2, 24 times
    datagram[5] = Value >> 16  # shift by 16 bits i.e. divide by 2, 16 times
    datagram[6] = Value >> 8   # shift by 8 bits i.e. divide by 2, 8 times
    datagram[7] = Value & 0xff # bitwise add with 0xff to get last 8 byte

    for i in range(8):         # generate checksum
        datagram[8] =  (datagram[8] + datagram[i]) & 0xff

    return datagram

#----------------------------------------------------------
# Send Datagram
#----------------------------------------------------------
def SendGram(datagram, ser):
    print(datagram)
    ser.write(datagram)
    return

#----------------------------------------------------------
# Receive Datagram
#----------------------------------------------------------
def GetGram(self):
    return

ser = motorport()

# Example: Stop 'Z' Motor
datagram = MakeGram(address, commands['MST'], 0, motors['Z'], 0)


SendGram(datagram, ser.driver)
print(datagram)
