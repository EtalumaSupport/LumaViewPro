# Test Trinamic Motion Boards and commands
# Trinamic 6110 board (now)
# Trinamic 3230 board (preferred)
# Want to use in direct mode - i.e. send and receive datagrams

class Trinamic():

    # https://www.trinamic.com/fileadmin/assets/Support/Software/TMCLDatagram.c
    # Opcodes of all TMCL commands that can be used in direct mode
    TMCL_ROR = 1    #
    TMCL_ROL = 2    #
    TMCL_MST = 3    #
    TMCL_MVP = 4    #
    TMCL_SAP = 5    #
    TMCL_GAP = 6    #
    TMCL_STAP = 7   #
    TMCL_RSAP = 8   #
    TMCL_SGP = 9    #
    TMCL_GGP = 10   #
    TMCL_STGP = 11  #
    TMCL_RSGP = 12  #
    TMCL_RFS = 13   #
    TMCL_SIO = 14   #
    TMCL_GIO = 15   #
    TMCL_SCO = 30   #
    TMCL_GCO = 31   #
    TMCL_CCO = 32   #
