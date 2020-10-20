import serial
# import time
#
# port = 'COM4'
# baudrate=9600
# bytesize=serial.EIGHTBITS
# parity=serial.PARITY_NONE
# stopbits=serial.STOPBITS_ONE
# timeout=10.0 # seconds
#
# driver = serial.Serial(port=port, baudrate=baudrate, bytesize=bytesize, parity=parity, stopbits=stopbits, timeout=timeout)
# driver.close()
# driver.open()
#
# for channel in range(3):
#     mA = 50
#     calibrate = '{CAL,'+ str(channel) + '}00'
#     turn_on   = '{TON,'+ str(channel) + ',H,' + str(mA) + '}00'
#     turn_off  = '{TOF}00'
#     driver.write(turn_on.encode('utf-8')+b"\r\n")
#     time.sleep(1)
# driver.close()

# ALTERNATIVE USING OBJECT ORIENTED PROGRAMMING #

class LEDBoard:
    def __init__(self, **kwargs):

        self.port = 'COM4'
        self.baudrate=9600
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=10.0 # seconds
        self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)
        self.driver.close()
        self.driver.open()

    def __del__(self):
        self.driver.close()

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        else:
            return 2

    def led_cal(self, channel):
        command = '{CAL,'+ str(channel) + '}00'
        self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_on(self, channel, mA):
        command = '{TON,'+ str(channel) + ',H,' + str(mA) + '}00'
        self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_off(self):
        command = '{TOF}00'
        self.driver.write(command.encode('utf-8')+b"\r\n")

EtaLumaDriver = LEDBoard()
EtaLumaDriver.led_on(EtaLumaDriver.color2ch('Red'), 50)
