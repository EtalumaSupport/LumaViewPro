#!/usr/bin/env python3

# Create, Send, and Receive commands (datagrams) to a Trinamic Motion Board
# Using direct mode via Serial communication

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
    'WAIT':27,  # Wait with further program execution
    'STOP':28,  # Stop program execution
    'SCO':30,   # Set Coordinate
    'GCO':31,   # Get Coordinate
    'CCO':32    # Capture Coordinate
}

commandtypes = {
    'MVP_ABS': 0,
    'MVP_REL': 1,
    'MVP_COORD': 2,
    'RFS_START': 0,
    'RFS_STOP': 1,
    'RFS_STATUS': 2
}

motors = {
    'X':0,
    'Y':1,
    'Z':2
}

#----------------------------------------------------------
# Set up Serial Port connection
#----------------------------------------------------------
import serial
import serial.tools.list_ports

class motorport:
    def __init__(self, **kwargs):
        ports = serial.tools.list_ports.comports(include_links = True)
        for port in ports:
            if (port.vid == 10812) and (port.pid == 256):
                print('Trinamic Motor Control Board identified at', port.device)
                self.port = port.device

        self.baudrate=9600
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
    datagram[4] = (Value >> 24)  & 0xff # shift by 24 bits i.e. divide by 2, 24 times
    datagram[5] = (Value >> 16)  & 0xff # shift by 16 bits i.e. divide by 2, 16 times
    datagram[6] = (Value >> 8)  & 0xff  # shift by 8 bits i.e. divide by 2, 8 times
    datagram[7] = Value & 0xff # bitwise add with 0xff to get last 8 byte

    for i in range(8):         # generate checksum
        datagram[8] =  (datagram[8] + datagram[i]) & 0xff

    return datagram

#----------------------------------------------------------
# Send Datagram
#----------------------------------------------------------
def SendGram(datagram, ser):
    print('Writing datagram:', datagram)
    ser.write(datagram)
    GetGram(ser)
    return

#----------------------------------------------------------
# Receive Datagram
#----------------------------------------------------------
def GetGram(ser):

    # receive the datagram
    datagram = ser.read(9)
    #print('Return datagram: ', datagram)

    checksum = 0
    for i in range(8):         # compare checksum
        checksum =  (checksum + datagram[i]) & 0xff

    if checksum != datagram[8]:
        return False

    Address=datagram[0]
    Status=datagram[2]
    print('Return Status:', Status)
    Value = int.from_bytes(datagram[4:8], byteorder='big', signed=True)
    return True

#----------------------------------------------------------
#----------------------------------------------------------
# TMC6110_X0_Y1_Z2_Axis_home_v1.tmc
#----------------------------------------------------------
#----------------------------------------------------------
ser = motorport()

'''
# Example: Stop 'Z' Motor
datagram = MakeGram(Address = address,
                    Command = commands['MST'],
                    Type = 0,
                    Motor = motors['Y'],
                    Value = 0)

SendGram(datagram, ser.driver)
'''
'''
#----------------------------------------------------------
# Z-Axis Initialization
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 6, 2, 16), ser.driver)      # SAP 6, 2, 16 // Maximum current
SendGram(MakeGram(255, commands['SAP'], 7, 2, 8), ser.driver)       # SAP 7, 2, 8 // Standby current
SendGram(MakeGram(255, commands['SAP'], 2, 2, 1000), ser.driver)    # SAP 2, 2, 1000  // Target Velocity
SendGram(MakeGram(255, commands['SAP'], 5, 2, 500), ser.driver)     # SAP 5, 2, 500 // Acceleration
SendGram(MakeGram(255, commands['SAP'], 140, 2, 5), ser.driver)     # SAP 140, 2, 5  //32X Microstepping
SendGram(MakeGram(255, commands['SAP'], 153, 2, 9), ser.driver)     # SAP 153, 2, 9  // Ramp Divisor 9
SendGram(MakeGram(255, commands['SAP'], 154, 2, 3), ser.driver)     # SAP 154, 2, 3  // Pulse Divisor 3
SendGram(MakeGram(255, commands['SAP'], 163, 2, 0), ser.driver)     # SAP 163, 2, 0  // Constant TOff Mode ( spreadcycle )


# Parameters for Limit Switches
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 12, 2, 0), ser.driver)      # SAP 12, 2, 0 // enable (un-disable) Right Limit switch
SendGram(MakeGram(255, commands['SAP'], 13, 2, 0), ser.driver)      # SAP 13, 2, 0 // enable (un-disable) Left Limit switch

# Parameters for Homing
#----------------------------------------------------------


# BUT DOESN'T THIS LOOK FOR THE 'RIGHT' LIMIT SWITCH? PG 116 MANUAL, Checked that this is 'down'
# "Add 64 to a mode for searching the right instead of the left reference switch"
SendGram(MakeGram(255, commands['SAP'], 193, 2, 65), ser.driver)   # SAP 193, 2, 65 // Search left Stop switch Only


SendGram(MakeGram(255, commands['SAP'], 194, 2, 1000), ser.driver) # SAP 194, 2, 1000 // Reference search speed
SendGram(MakeGram(255, commands['SAP'], 195, 2, 10), ser.driver)   # SAP 195, 2, 10 // Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
SendGram(MakeGram(255, commands['RFS'], 0, 2, 0), ser.driver)                     # RFS START, 2
SendGram(MakeGram(255, commands['WAIT'], commands['RFS'], 2, 0), ser.driver)      # WAIT RFS, 2 , 0

# Move out of home Position
#----------------------------------------------------------
SendGram(MakeGram(255, commands['MVP'], 0, 2, 100000), ser.driver) # MVP ABS, 2, 100000 // for the TMCM-6110 the parameter is positive.

'''







#----------------------------------------------------------
# X-Axis Initialization
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 6, 0, 16), ser.driver)     # SAP 6, 0, 16 // Maximum current
SendGram(MakeGram(255, commands['SAP'], 7, 0, 8), ser.driver)      # SAP 7, 0, 8 // Standby current
SendGram(MakeGram(255, commands['SAP'], 2, 0, 1000), ser.driver)   # SAP 2, 0, 1000  // Target Velocity
SendGram(MakeGram(255, commands['SAP'], 5, 0, 500), ser.driver)    # SAP 5, 0, 500 // Acceleration
SendGram(MakeGram(255, commands['SAP'], 140, 0, 5), ser.driver)    # SAP 140, 0, 5  //32X Microstepping
SendGram(MakeGram(255, commands['SAP'], 153, 0, 9), ser.driver)    # SAP 153, 0, 9  // Ramp Divisor 9
SendGram(MakeGram(255, commands['SAP'], 154, 0, 3), ser.driver)    # SAP 154, 0, 3  // Pulse Divisor 3
SendGram(MakeGram(255, commands['SAP'], 163, 0, 0), ser.driver)    # SAP 163, 0, 0  // Constant TOff Mode ( spreadcycle )

# Parameters for Limit Switches
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 12, 0, 0), ser.driver)     # SAP 12, 0, 0 // enable (un-disable) Right Limit switch
SendGram(MakeGram(255, commands['SAP'], 13, 0, 0), ser.driver)     # SAP 13, 0, 0 // enable (un-disable) Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 193, 0, 65), ser.driver)   # SAP 193, 0, 65 // Search left Stop switch Only
SendGram(MakeGram(255, commands['SAP'], 194, 0, 1000), ser.driver) # SAP 194, 0, 1000 // Reference search speed
SendGram(MakeGram(255, commands['SAP'], 195, 0, 10), ser.driver)   # SAP 195, 0, 10 // Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
SendGram(MakeGram(255, commands['RFS'], 0, 0, 0), ser.driver)      # RFS START, 0
SendGram(MakeGram(255, commands['WAIT'], commands['RFS'], 0, 0), ser.driver) # WAIT RFS, 0 , 0

# Move out of home Position
#----------------------------------------------------------
SendGram(MakeGram(255, commands['MVP'], 0, 0, 100000), ser.driver) # MVP ABS, 0, 100000 // for the TMCM-6110 the parameter is positive.
# # Added by anna
# SendGram(MakeGram(255, commands['WAIT'], 1, 0, 0), ser.driver) # WAIT, target position, x-motor , no-timeout


'''








#----------------------------------------------------------
# Y-Axis Initialization
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 6, 1, 16), ser.driver)    # SAP 6, 1, 16 // Maximum current
SendGram(MakeGram(255, commands['SAP'], 7, 1, 8), ser.driver)     # SAP 7, 1, 8 // Standby current
SendGram(MakeGram(255, commands['SAP'], 2, 1, 1000), ser.driver)  # SAP 2, 1, 1000  // Target Velocity
SendGram(MakeGram(255, commands['SAP'], 5, 1, 500), ser.driver)   # SAP 5, 1, 500 // Acceleration
SendGram(MakeGram(255, commands['SAP'], 140, 1, 5), ser.driver)   # SAP 140, 1, 5  //32X Microstepping
SendGram(MakeGram(255, commands['SAP'], 153, 1, 9), ser.driver)   # SAP 153, 1, 9  // Ramp Divisor 9
SendGram(MakeGram(255, commands['SAP'], 154, 1, 3), ser.driver)   # SAP 154, 1, 3  // Pulse Divisor 3
SendGram(MakeGram(255, commands['SAP'], 163, 1, 0), ser.driver)   # SAP 163, 1, 0  // Constant TOff Mode ( spreadcycle )

# Parameters for Limit Switches
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 12, 1, 0), ser.driver)    # SAP 12, 1, 0 // enable (un-disable) Right Limit switch
SendGram(MakeGram(255, commands['SAP'], 13, 1, 0), ser.driver)    # SAP 13, 1, 0 // enable (un-disable) Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
SendGram(MakeGram(255, commands['SAP'], 193, 1, 1), ser.driver)   # SAP 193, 1, 65 // Search left Stop switch Only
SendGram(MakeGram(255, commands['SAP'], 194, 1, 1000), ser.driver) # SAP 194, 1, 1000 // Reference search speed
SendGram(MakeGram(255, commands['SAP'], 195, 1, 10), ser.driver)   # SAP 195, 1, 10 // Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
SendGram(MakeGram(255, commands['RFS'], 0, 2, 0), ser.driver)      # RFS START, 1
SendGram(MakeGram(255, commands['WAIT'], commands['RFS'], 1, 0), ser.driver) # WAIT RFS, 1, 0

# Move out of home Position
#----------------------------------------------------------
SendGram(MakeGram(255, commands['MVP'], 0, 1, 100000), ser.driver)  # MVP ABS, 1, 100000 // for the TMCM-6110 the parameter is positive.

'''






'''
# Stop the Program
#----------------------------------------------------------
SendGram(MakeGram(255, commands['STOP'], 0, 0, 0), ser.driver)
'''
