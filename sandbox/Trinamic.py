#!/usr/bin/env python3

# Create, Send, and Receive commands (datagrams) to a Trinamic Motion Board
# Using direct mode via Serial communication

# Converted from
# https://www.trinamic.com/fileadmin/assets/Support/Software/TMCLDatagram.c



import serial
import serial.tools.list_ports
import time

class TrinamicBoard:
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

    # cmndtypes = {
    #     'NA': 0,
    #     'MVP_ABS': 0,
    #     'MVP_REL': 1,
    #     'MVP_COORD': 2,
    #     'RFS_START': 0,
    #     'RFS_STOP': 1,
    #     'RFS_STATUS': 2
    # }

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
            print("Could not connect to Trinamic Motor Control Board")

    #----------------------------------------------------------
    # Receive Datagram
    #----------------------------------------------------------
    def GetGram(self, verbose = False):

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
        datagram[4] = (Value >> 24)  & 0xff # shift by 24 bits i.e. divide by 2, 24 times
        datagram[5] = (Value >> 16)  & 0xff # shift by 16 bits i.e. divide by 2, 16 times
        datagram[6] = (Value >> 8)  & 0xff  # shift by 8 bits i.e. divide by 2, 8 times
        datagram[7] = Value & 0xff # bitwise add with 0xff to get last 8 byte

        for i in range(8):         # generate checksum
            datagram[8] =  (datagram[8] + datagram[i]) & 0xff

        self.driver.write(datagram)
        return self.GetGram(verbose = True)


    # Wait for reference function to complete (homing)
    def RFS_Wait(self, Motor):
        value = 1
        while value != 0:
            value = self.SendGram('RFS', 2, Motor, 0)
            time.sleep(0.1)


#----------------------------------------------------------
#----------------------------------------------------------
# TMC6110_X0_Y1_Z2_Axis_home_v1.tmc
#----------------------------------------------------------
#----------------------------------------------------------
motion = TrinamicBoard()


# Example: Stop 'Z' Motor
motion.SendGram('MST',
                Type = 0,
                Motor = 'Y',
                Value = 0)



#----------------------------------------------------------
# Z-Axis Initialization
#----------------------------------------------------------
motion.SendGram('SAP', 6, 'Z', 16)      # Maximum current
motion.SendGram('SAP', 7, 'Z', 8)       # Standby current
motion.SendGram('SAP', 2, 'Z', 1000)    # Target Velocity
motion.SendGram('SAP', 5, 'Z', 500)     # Acceleration
motion.SendGram('SAP', 140, 'Z', 5)     # 32X Microstepping
motion.SendGram('SAP', 153, 'Z', 9)     # Ramp Divisor 9
motion.SendGram('SAP', 154, 'Z', 3)     # Pulse Divisor 3
motion.SendGram('SAP', 163, 'Z', 0)     # Constant TOff Mode (spreadcycle)


# Parameters for Limit Switches
#----------------------------------------------------------
motion.SendGram('SAP', 12, 'Z', 0)      # enable Right Limit switch
motion.SendGram('SAP', 13, 'Z', 0)      # enable Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
motion.SendGram('SAP', 193, 'Z', 65)    # Search Right Stop Switch (Down)
motion.SendGram('SAP', 194, 'Z', 1000)  # Reference search speed
motion.SendGram('SAP', 195, 'Z', 10)    # Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
motion.SendGram('RFS', 0, 'Z', 0)       # Home to the Right Limit switch (Down)
motion.RFS_Wait('Z')

# # Move out of home Position
# #----------------------------------------------------------
# motion.SendGram('MVP', 0, 'Z', -100)  # Move up by 100 (what is the unit?)









#----------------------------------------------------------
# X-Axis Initialization
#----------------------------------------------------------
motion.SendGram('SAP', 6, 'X', 16)     # Maximum current
motion.SendGram('SAP', 7, 'X', 8)      # standby current
motion.SendGram('SAP', 2, 'X', 1000)   # Target Velocity
motion.SendGram('SAP', 5, 'X', 500)    # Acceleration
motion.SendGram('SAP', 140, 'X', 5)    # 32X Microstepping
motion.SendGram('SAP', 153, 'X', 9)    # Ramp Divisor 9
motion.SendGram('SAP', 154, 'X', 3)    # Pulse Divisor 3
motion.SendGram('SAP', 163, 'X', 0)    # Constant TOff Mode (spreadcycle)

# Parameters for Limit Switches
#----------------------------------------------------------
motion.SendGram('SAP', 12, 'X', 0)     # enable Right Limit switch
motion.SendGram('SAP', 13, 'X', 0)     # enable Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
# 'Set Axis Parameter', 'Reference Search Mode', 'X-axis', '1+64 = Search Right Stop Switch Only'
motion.SendGram('SAP', 193, 'X', 65)   # Search Right Stop switch Only
motion.SendGram('SAP', 194, 'X', 1000) # Reference search speed
motion.SendGram('SAP', 195, 'X', 10)   # Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
# ----------------------------------------------------------
motion.SendGram('RFS', 0, 'X', 0)      # Home to the Right Limit switch (Right)
motion.RFS_Wait('X')

# Move out of home Position
#----------------------------------------------------------
motion.SendGram('MVP', 0, 'X', -200000)  # Move left by 100000 (what is the unit?)








#----------------------------------------------------------
# Y-Axis Initialization
#----------------------------------------------------------
motion.SendGram('SAP', 6, 'Y', 16)    # Maximum current
motion.SendGram('SAP', 7, 'Y', 8)     # Standby current
motion.SendGram('SAP', 2, 'Y', 1000)  # Target Velocity
motion.SendGram('SAP', 5, 'Y', 500)   # Acceleration
motion.SendGram('SAP', 140, 'Y', 5)   # 32X Microstepping
motion.SendGram('SAP', 153, 'Y', 9)   # Ramp Divisor 9
motion.SendGram('SAP', 154, 'Y', 3)   # Pulse Divisor 3
motion.SendGram('SAP', 163, 'Y', 0)   # Constant TOff Mode (spreadcycle)

# Parameters for Limit Switches
#----------------------------------------------------------
motion.SendGram('SAP', 12, 'Y', 0)    # enable Right Limit switch
motion.SendGram('SAP', 13, 'Y', 0)    # enable Left Limit switch

# Parameters for Homing
#----------------------------------------------------------
motion.SendGram('SAP', 193, 'Y', 65)   # Search Right Stop switch Only (Back)
motion.SendGram('SAP', 194, 'Y', 1000) # Reference search speed
motion.SendGram('SAP', 195, 'Y', 10)   # Reference switch speed (was 10X less than search speed in LumaView)

# Start the Trinamic Homing Procedure
#----------------------------------------------------------
motion.SendGram('RFS', 0, 'Y', 0)      # Home to the Right Limit switch (Back)
motion.RFS_Wait('Y')

# Move out of home Position
#----------------------------------------------------------
motion.SendGram('MVP', 0, 'Y', -100000)  # Move forward by 100000 (what is the unit?)
