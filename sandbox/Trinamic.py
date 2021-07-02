# Create, Send, and Receive commands (datagrams) to a Trinamic Motion Board
# Using direct mode and Serial communication

# Converted from
# https://www.trinamic.com/fileadmin/assets/Support/Software/TMCLDatagram.c

# Trinamic 6110 board (now)
# Trinamic 3230 board (preferred)

class Trinamic:
    def __init__(self, **kwargs):
        self.Address = 0
        self.connect()

    #---------------------------------------------------------
    # Address
    #----------------------------------------------------------

    # ---------------------------------------------------------
    # Commands
    #----------------------------------------------------------
    def Command(self, Code):
        command = {
            'ROR':1,
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
        return command[Code]

    # ---------------------------------------------------------
    # Type
    #----------------------------------------------------------


    # ---------------------------------------------------------
    # Motor
    #----------------------------------------------------------
    def Motor(self, Axis):
        motor = {'X':0, 'Y':1, 'Z':2}
        return motor[Axis]

    #----------------------------------------------------------
    # Generate Datagram
    #----------------------------------------------------------
    def MakeGram(self, Address, Command, Type, Motor, Value):

        a = byte(Address)
        c = byte(Command)
        t = byte(Type)
        m = byte(Motor)
        # 'Value' is an integer
        v1 = byte(Value >> 24)  # shift by 24 bits i.e. divide by 2, 24 times
        v2 = byte(Value >> 16)  # shift by 16 bits i.e. divide by 2, 24 times
        v3 = byte(Value >> 8)   # shift by 8 bits i.e. divide by 2, 24 times
        v4 = byte(Value) & 0xff # bitwise add with 0xff to get last 8 byte


    #
    #     TxBuffer[0]=Address
    #     TxBuffer[1]=Command
    #     TxBuffer[2]=Type
    #     TxBuffer[3]=Motor
    #     TxBuffer[4]=Value >> 24 # shift by 24 bits i.e. divide by 2, 24 times
    #     TxBuffer[5]=Value >> 16 # shift by 16 bits i.e. divide by 2, 16 times
    #     TxBuffer[6]=Value >> 8  # shift by 8 bits i.e. divide by 2, 8 times
    #     TxBuffer[7]=Value & 0xff
    #     TxBuffer[8]=0
    #     for(i=0; i<8; i++)
    #         TxBuffer[8]+=TxBuffer[i]
    #     return datagram
    #
    # #----------------------------------------------------------
    # # Send Datagram
    # #----------------------------------------------------------
    # def SendGram(self, datagram):
    #     return
    # 	# Now, send the 9 bytes stored in TxBuffer to the module
    #     # (this is MCU specific)
    #
    #
    #
    # # Get the result
    # # Return TRUE when checksum of the result if okay, else return FALSE
    # # The follwing values are returned:
    # #       *Address: Host address
    # #       *Status: Status of the module (100 means okay)
    # #       *Value: Value read back by the command
    # def GetGram(HANDLE Handle, UCHAR *Address, UCHAR *Status, INT *Value):
    # {
    # 	UCHAR RxBuffer[9], Checksum;
    # 	DWORD Errors, BytesRead;
    # 	COMSTAT ComStat;
    # 	int i;
    #
    #   //First, get 9 bytes from the module and store them in RxBuffer[0..8]
    #   //(this is MCU specific)
    #
    # 	//Check the checksum
    # 	Checksum=0;
    # 	for(i=0; i<8; i++)
    # 		Checksum+=RxBuffer[i];
    #
    # 	if(Checksum!=RxBuffer[8]) return FALSE;
    #
    # 	//Decode the datagram
    # 	*Address=RxBuffer[0];
    # 	*Status=RxBuffer[2];
    # 	*Value=(RxBuffer[4] << 24) | (RxBuffer[5] << 16) | (RxBuffer[6] << 8) | RxBuffer[7];
    #
    # 	return TRUE;
    # }
