import serial
import serial.tools.list_ports as list_ports
import time

# /dev/ttyACM0
#     desc: STM32 Virtual ComPort
#     hwid: USB VID:PID=0483:5740 SER=205835435736 LOCATION=1-5:1.0
# /dev/ttyS0
#     desc: ttyS0
#     hwid: PNP0501

class LEDBoard:
    def __init__(self, **kwargs):

        ports = list(list_ports.comports())
        if (len(ports)!=0):
            print(ports[0])
            self.port = ports[0].device
        #self.port="COM5"
<<<<<<< HEAD
        self.port="/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_205835435736-if00"
=======
        self.port = "/dev/ttyS0"
>>>>>>> 66d1251ce4200ecfdbcbf10dc9dccd9f9f1f0969
        self.baudrate=9600
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=10.0 # seconds
        self.driver = True
        self.connect()

    def __del__(self):
        if self.driver != False:
            self.driver.close()

    def connect(self):
        try:
            self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)
            self.driver.close()
            self.driver.open()
        except:
            if self.driver != False:
                print("It looks like a Lumaview compatible LED driver board is not plugged in")
            self.driver = False

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        else:
            return 3

    def led_cal(self, channel):
        command = '{CAL,'+ str(channel) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_on(self, channel, mA):
        command = '{TON,'+ str(channel) + ',H,' + str(mA) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_off(self):
        command = '{TOF}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

led_board = LEDBoard()
print('led_board connection attempted')

while True:
    mA = input('current:')
    if mA == 0:
        led_board.led_off()
    elif mA == 'q':
        break
    else:
        led_board.led_on(0, mA)
