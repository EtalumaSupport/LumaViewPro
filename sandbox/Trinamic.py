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

    def connect(self):
        return # needs to be filled
    #---------------------------------------------------------
    # Address
    #----------------------------------------------------------

    # ---------------------------------------------------------
    # Commands
    #----------------------------------------------------------
    def Command(self, Code):
        command = {
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
        return command[Code]
'''
        //Options for MVP commands
        #define MVP_ABS 0   # absolute
        #define MVP_REL 1   # relative
        #define MVP_COORD 2 # coordinate

        //Options for RFS command
        #define RFS_START 0
        #define RFS_STOP 1
        #define RFS_STATUS 2
'''
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

        datagram = bytearray(9)
        datagram[0] = Address
        datagram[1] = Command
        datagram[2] = Type
        datagram[3] = Motor
        # https://stackoverflow.com/questions/41274864/declaring-a-single-byte-variable-in-python
        # 'Value' is an integer
        datagram[4] = Value >> 24  # shift by 24 bits i.e. divide by 2, 24 times
        datagram[5] = Value >> 16  # shift by 16 bits i.e. divide by 2, 16 times
        datagram[6] = Value >> 8   # shift by 8 bits i.e. divide by 2, 8 times
        datagram[7] = Value & 0xff # bitwise add with 0xff to get last 8 byte

        for i in range(8):         # generate checksum
            datagram[8] += datagram[i]

        return datagram

'''
        TxBuffer[0]=Address
        TxBuffer[1]=Command
        TxBuffer[2]=Type
        TxBuffer[3]=Motor
        TxBuffer[4]=Value >> 24
        TxBuffer[5]=Value >> 16
        TxBuffer[6]=Value >> 8
        TxBuffer[7]=Value & 0xff
        TxBuffer[8]=0
        for(i=0; i<8; i++)
            TxBuffer[8]+=TxBuffer[i]
'''

    #----------------------------------------------------------
    # Send Datagram
    #----------------------------------------------------------
    def SendGram(self, datagram):
        return

    #----------------------------------------------------------
    # Receive Datagram
    #----------------------------------------------------------

    def GetGram(self):
        return

'''
    //Get the result
    //Return TRUE when checksum of the result if okay, else return FALSE
    // The follwing values are returned:
    //      *Address: Host address
    //      *Status: Status of the module (100 means okay)
    //      *Value: Value read back by the command
    UCHAR GetResult(HANDLE Handle, UCHAR *Address, UCHAR *Status, INT *Value)
    {
    	UCHAR RxBuffer[9], Checksum;
    	DWORD Errors, BytesRead;
    	COMSTAT ComStat;
    	int i;

      //First, get 9 bytes from the module and store them in RxBuffer[0..8]
      //(this is MCU specific)

    	//Check the checksum
    	Checksum=0;
    	for(i=0; i<8; i++)
    		Checksum+=RxBuffer[i];

    	if(Checksum!=RxBuffer[8]) return FALSE;

    	//Decode the datagram
    	*Address=RxBuffer[0];
    	*Status=RxBuffer[2];
    	*Value=(RxBuffer[4] << 24) | (RxBuffer[5] << 16) | (RxBuffer[6] << 8) | RxBuffer[7];

    	return TRUE;
    }
'''
