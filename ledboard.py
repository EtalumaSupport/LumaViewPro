# Modified 5/21/2022

import serial
import serial.tools.list_ports as list_ports

class LEDBoard:
    def __init__(self, **kwargs):
        ports = serial.tools.list_ports.comports(include_links = True)

        for port in ports:
            if (port.vid == 11914) and (port.pid == 5):
                print('LED Control Board v3 identified at', port.device)
                self.port = port.device

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=.5 # seconds
        self.driver = True
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
                                        timeout=self.timeout)
            self.driver.close()
            self.driver.open()
        except:
            if self.driver != False:
                print("It looks like a Lumaview compatible LED driver board is not plugged in.")
                print("Error: LEDBoard.connect() exception")
            self.driver = False

    def send_mssg(self, mssg):
        stream = mssg.encode('utf-8')+b"\r\n"
        if self.driver != False:
            self.driver.write(stream)

    def receive_mssg(self):
        if self.driver != False:
            stream = self.driver.readline()
            mssg = stream.decode("utf-8","ignore")
            return mssg[:-2]

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        else: # BF
            return 3

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
        command = 'LED' + str(channel) + '_' + str(int(mA))
        self.send_mssg(command)

    def led_off(self, channel):
        command = 'LED' + str(channel) + '_OFF'
        self.send_mssg(command)

    def leds_off(self):
        command = 'LEDS_OFF'
        self.send_mssg(command)
