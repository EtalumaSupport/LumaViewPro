import serial
import serial.tools.list_ports as list_ports
import time

'''
DEBUGGING NOTES
---------------
/dev/ttyACM0
    desc: STM32 Virtual ComPort
    hwid: USB VID:PID=0483:5740 SER=205835435736 LOCATION=1-5:1.0
/dev/ttyS0
    desc: ttyS0
    hwid: PNP0501
'''

class LEDBoard:
    def __init__(self, **kwargs):

        # find all available serial ports
        ports = list(list_ports.comports(include_links = True))
        for port in ports:
            if 'PJRC.COM' in port.manufacturer:
                self.port = port.device

        # self.port="/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_205835435736-if00"
        # self.port = "/dev/ttyS0"
        self.baudrate=11520
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=.5 # seconds
        self.driver = True
        self.connect()

    def __del__(self):
        if self.driver != False:
            self.led_off()
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

    def led_status(self):
        command = '{SST}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

            n_input = self.driver.in_waiting
            if n_input > 0:
                buffer = self.driver.read(n_input)
                status = buffer.decode('utf-8')
                print(status)
                if 'Etaluma' in status:
                    return True
                else:
                    return False

    def led_cal(self, channel):
        command = '{CAL,' + str(channel) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_on(self, channel, mA):
        command = '{TON,' + str(channel) + ',H,' + str(mA) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_off(self):
        command = '{TOF}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        else:
            return 3



# Create instance
led_board = LEDBoard()

#'''
# Command Prompt Control of LEDBoard
while True:
    CH = input('channel:')
    mA = input('current:')
    if mA == 0:
        led_board.led_off()
    elif mA == 'q':
        break
    elif CH == 'q':
        break
    else:
        led_board.led_on(CH, mA)
    print('')
#'''

'''
# Test Speed of LEDBoard
for i in range(10000):
    led_board.led_on(i%3, 50)
    led_board.led_off()
    time.sleep(1/10000)
'''
