# Modified 5/21/2022
from numpy import False_
import serial
import serial.tools.list_ports as list_ports

class LEDBoard:    
    def __init__(self, **kwargs):
        ports = list_ports.comports(include_links = True)
        self.message = 'LEDBoard.__init__()'

        for port in ports:
            # print("vid ", port.vid, " pid ", port.pid)
            #if (port.vid == 11914) and (port.pid == 5):
            if (port.vid == 0x2E8A) and (port.pid == 0x0005):
                print('LED Control Board v3 identified at', port.device)
                self.port = port.device
                break

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=0.5 # seconds
        self.write_timeout=0.5 # seconds
        self.driver = False
        self.connect()

    def __del__(self):
        if self.driver != False:
            self.driver.close()

    def connect(self):
        try:
            self.driver = serial.Serial(port=self.port,
                                        baudrate=self.baudrate,
                                        bytesize=self.bytesize,
                                        parity=self.parity,
                                        stopbits=self.stopbits,
                                        timeout=self.timeout,
                                        write_timeout=self.write_timeout)
            self.driver.close()
            self.driver.open()
            self.message = 'LEDBoard.connect() succeeded'
            print('LEDBoard.connect() succeeded')
        except:
            self.driver = False
            self.message = 'LEDBoard.connect() failed'
            print('LEDBoard.connect() succeeded')
            
    def send_mssg(self, mssg):
        stream = mssg.encode('utf-8')+b"\r\n"
        if self.driver != False:
            try:
                self.driver.close()
                self.driver.open()
                self.driver.write(stream)
                self.message = 'LEDBoard.send_mssg('+mssg+') succeeded'
                # self.message = mssg
                return True
            except serial.SerialTimeoutException:
                self.message = 'LEDBoard.send_mssg('+mssg+') Serial Timeout Occurred'
                return False
        else:
            self.connect()
            return False

    def receive_mssg(self):
        if self.driver != False:
            try:
                self.driver.close()
                self.driver.open()
                stream = self.driver.readline()
                mssg = stream.decode("utf-8","ignore")
                return mssg[:-2]
            except serial.SerialTimeoutException:
                self.message = 'LEDBoard.receive_mssg('+mssg+') Serial Timeout Occurred'

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        else: # BF
            return 3

    def ch2color(self, channel):
        if channel == 0:
            return 'Blue'
        elif channel == 1:
            return 'Green'
        elif channel == 2:
            return 'Red'
        else:
            return 'BF'

    # interperet commands
    # ------------------------------------------
    # board status: 'STATUS' case insensitive
    # LED enable:   'LED' channel '_ENT' where channel is numbers 0 through 5, or S (plural/all)
    # LED disable:  'LED' channel '_ENF' where channel is numbers 0 through 5, or S (plural/all)
    # LED on:       'LED' channel '_MA' where channel is numbers 0 through 5, or S (plural/all)
    #                and MA is numerical representation of mA
    # LED off:      'LED' channel '_OFF' where channel is numbers 0 through 5, or S (plural/all)

    def leds_enable(self):
        command = 'LEDS_ENT'
        self.send_mssg(command)

    def leds_disable(self):
        command = 'LEDS_ENF'
        self.send_mssg(command)

    def led_on(self, channel, mA):
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        self.send_mssg(command)

    def led_off(self, channel):
        command = 'LED' + str(int(channel)) + '_OFF'
        self.send_mssg(command)

    def leds_off(self):
        command = 'LEDS_OFF'
        self.send_mssg(command)
