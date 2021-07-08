#!/usr/bin/env python3

# Create, Send, and Receive commands (datagrams) to a Trinamic Motion Board
# Using direct mode via Serial communication

# Converted from
# https://www.trinamic.com/fileadmin/assets/Support/Software/TMCLDatagram.c


# Trinamic 3230 board (preferred)
# Trinamic 6110 board (now) Address
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

cmndtypes = {
    'MVP_ABS': 0,
    'MVP_REL': 1,
    'MVP_COORD': 2,
    'RFS_START': 0,
    'RFS_STOP': 1,
    'RFS_STATUS': 2
}

axis = {
    'X':0, # Left to Right is positive
    'Y':1, # Front to Back is positive
    'Z':2  # Down is positive
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
# Receive Datagram
#----------------------------------------------------------
def GetGram(ser, verbose = False):

    # receive the datagram
    datagram = ser.read(9)

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
def SendGram(datagram, ser):
    ser.write(datagram)
    return GetGram(ser)



# Wait for reference function to complete (homing)
import time
def RFS_Wait(Motor):
    value = 1
    while value != 0:
        value = SendGram(MakeGram(addr, cmnd['RFS'], 2, Motor, 0), ser.driver)
        time.sleep(0.1)


#----------------------------------------------------------
#----------------------------------------------------------
# TMC6110_X0_Y1_Z2_Axis_home_v1.tmc
#----------------------------------------------------------
#----------------------------------------------------------
ser = motorport()

'''
# Example: Stop 'Z' Motor
datagram = MakeGram(Address = addr,
                    Command = cmnd['MST'],
                    Type = 0,
                    Motor = axis['Y'],
                    Value = 0)

SendGram(datagram, ser.driver)
'''






#----------------------------------------------------------
# Z-Axis Initialization
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 6, axis['Z'], 16), ser.driver)      # Maximum current
SendGram(MakeGram(addr, cmnd['SAP'], 7, axis['Z'], 8), ser.driver)       # Standby current
SendGram(MakeGram(addr, cmnd['SAP'], 2, axis['Z'], 1000), ser.driver)    # Target Velocity
SendGram(MakeGram(addr, cmnd['SAP'], 5, axis['Z'], 500), ser.driver)     # Acceleration
SendGram(MakeGram(addr, cmnd['SAP'], 140, axis['Z'], 5), ser.driver)     # 32X Microstepping
SendGram(MakeGram(addr, cmnd['SAP'], 153, axis['Z'], 9), ser.driver)     # Ramp Divisor 9
SendGram(MakeGram(addr, cmnd['SAP'], 154, axis['Z'], 3), ser.driver)     # Pulse Divisor 3
SendGram(MakeGram(addr, cmnd['SAP'], 163, axis['Z'], 0), ser.driver)     # Constant TOff Mode (spreadcycle)


# Parameters for Limit Switches
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 12, axis['Z'], 0), ser.driver)      # enable Right Limit switch
SendGram(MakeGram(addr, cmnd['SAP'], 13, axis['Z'], 0), ser.driver)      # enable Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 193, axis['Z'], 65), ser.driver)    # Search Right Stop Switch (Down)
SendGram(MakeGram(addr, cmnd['SAP'], 194, axis['Z'], 1000), ser.driver)  # Reference search speed
SendGram(MakeGram(addr, cmnd['SAP'], 195, axis['Z'], 10), ser.driver)    # Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['RFS'], 0, axis['Z'], 0), ser.driver)       # Home to the Right Limit switch (Down)
RFS_Wait(axis['Z'])

# # Move out of home Position
# #----------------------------------------------------------
# SendGram(MakeGram(addr, cmnd['MVP'], 0, axis['Z'], -100), ser.driver)  # Move up by 100 (what is the unit?)









#----------------------------------------------------------
# X-Axis Initialization
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 6, axis['X'], 16), ser.driver)     # Maximum current
SendGram(MakeGram(addr, cmnd['SAP'], 7, axis['X'], 8), ser.driver)      # standby current
SendGram(MakeGram(addr, cmnd['SAP'], 2, axis['X'], 1000), ser.driver)   # Target Velocity
SendGram(MakeGram(addr, cmnd['SAP'], 5, axis['X'], 500), ser.driver)    # Acceleration
SendGram(MakeGram(addr, cmnd['SAP'], 140, axis['X'], 5), ser.driver)    # 32X Microstepping
SendGram(MakeGram(addr, cmnd['SAP'], 153, axis['X'], 9), ser.driver)    # Ramp Divisor 9
SendGram(MakeGram(addr, cmnd['SAP'], 154, axis['X'], 3), ser.driver)    # Pulse Divisor 3
SendGram(MakeGram(addr, cmnd['SAP'], 163, axis['X'], 0), ser.driver)    # Constant TOff Mode (spreadcycle)

# Parameters for Limit Switches
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 12, axis['X'], 0), ser.driver)     # enable Right Limit switch
SendGram(MakeGram(addr, cmnd['SAP'], 13, axis['X'], 0), ser.driver)     # enable Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
# 'Set Axis Parameter', 'Reference Search Mode', 'X-axis', '1+64 = Search Right Stop Switch Only'
SendGram(MakeGram(addr, cmnd['SAP'], 193, axis['X'], 65), ser.driver)   # Search Right Stop switch Only
SendGram(MakeGram(addr, cmnd['SAP'], 194, axis['X'], 1000), ser.driver) # Reference search speed
SendGram(MakeGram(addr, cmnd['SAP'], 195, axis['X'], 10), ser.driver)   # Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
# ----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['RFS'], 0, axis['X'], 0), ser.driver)      # Home to the Right Limit switch (Right)
RFS_Wait(axis['X'])

# Move out of home Position
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['MVP'], 0, axis['X'], -200000), ser.driver)  # Move left by 100000 (what is the unit?)








#----------------------------------------------------------
# Y-Axis Initialization
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 6, axis['Y'], 16), ser.driver)    # Maximum current
SendGram(MakeGram(addr, cmnd['SAP'], 7, axis['Y'], 8), ser.driver)     # Standby current
SendGram(MakeGram(addr, cmnd['SAP'], 2, axis['Y'], 1000), ser.driver)  # Target Velocity
SendGram(MakeGram(addr, cmnd['SAP'], 5, axis['Y'], 500), ser.driver)   # Acceleration
SendGram(MakeGram(addr, cmnd['SAP'], 140, axis['Y'], 5), ser.driver)   # 32X Microstepping
SendGram(MakeGram(addr, cmnd['SAP'], 153, axis['Y'], 9), ser.driver)   # Ramp Divisor 9
SendGram(MakeGram(addr, cmnd['SAP'], 154, axis['Y'], 3), ser.driver)   # Pulse Divisor 3
SendGram(MakeGram(addr, cmnd['SAP'], 163, axis['Y'], 0), ser.driver)   # Constant TOff Mode (spreadcycle)

# Parameters for Limit Switches
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 12, axis['Y'], 0), ser.driver)    # enable Right Limit switch
SendGram(MakeGram(addr, cmnd['SAP'], 13, axis['Y'], 0), ser.driver)    # enable Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['SAP'], 193, axis['Y'], 65), ser.driver)   # Search Right Stop switch Only (Back)
SendGram(MakeGram(addr, cmnd['SAP'], 194, axis['Y'], 1000), ser.driver) # Reference search speed
SendGram(MakeGram(addr, cmnd['SAP'], 195, axis['Y'], 10), ser.driver)   # Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['RFS'], 0, axis['Y'], 0), ser.driver)      # Home to the Right Limit switch (Back)
RFS_Wait(axis['Y'])

# Move out of home Position
#----------------------------------------------------------
SendGram(MakeGram(addr, cmnd['MVP'], 0, axis['Y'], -100000), ser.driver)  # Move forward by 100000 (what is the unit?)
