#!/usr/bin/python3

'''
MIT License

Copyright (c) 2020 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.

AUTHORS:
Kevin Peter Hickerson, The Earthineering Company
Anna Iwaniec Hickerson, Keck Graduate Institute

MODIFIED:
January 22, 2023
'''

from numpy import False_
import serial
import serial.tools.list_ports as list_ports

class LEDBoard:    
    def __init__(self, **kwargs):
        ports = list_ports.comports(include_links = True)
        self.message = 'LEDBoard.__init__()'

        for port in ports:
            if (port.vid == 0x0424) and (port.pid == 0x704C):
                print('LED Controller at', port.device)
                self.port = port.device
                break

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=0.01 # seconds
        self.write_timeout=0.01 # seconds
        self.driver = False
        try:
            print('Found LED controller and about to establish connection.')
            self.connect()
        except:
            print('Found LED controller but unable to establish connection.')
            raise


    # def __del__(self):
    #     if self.driver != False:
    #         self.driver.close()

    def connect(self):
        """ Try to connect to the motor controller based on the known VID/PID"""
        try:
            print('Found LED controller and about to create driver.')
            self.driver = serial.Serial(port=self.port,
                                        baudrate=self.baudrate,
                                        bytesize=self.bytesize,
                                        parity=self.parity,
                                        stopbits=self.stopbits,
                                        timeout=self.timeout,
                                        write_timeout=self.write_timeout)
            print('Found LED controller and created driver.')
            self.driver.close()
            self.driver.open()
            print('Found LED controller and closed and opened again.')
            self.send_command ('import main.py')         
            print ('import main.py')         
            self.send_command ('import main.py') 
            print ('import main.py')         
            self.message = 'LEDBoard.connect() succeeded'
        except:
            self.driver = False
            self.message = 'LEDBoard.connect() failed'
            print('LEDBoard.connect() failed')
            raise
            
    def send_command(self, command):
        """ Send command through serial to LED controller
        This should NOT be used in a script. It is intended for other functions to access"""

        stream = command.encode('utf-8')+b"\r\n"
        if self.driver != False:
            try:
                self.driver.close()
                self.driver.open()
                self.driver.write(stream)
                self.message = 'LEDBoard.send_command('+command+') succeeded'
                # self.message = command
                return True
            except serial.SerialTimeoutException:
                self.message = 'LEDBoard.send_command('+command+') Serial Timeout Occurred'
                raise
            except:
                raise
        else:
            raise Exception('Driver for LED controller not set.')

    def receive_command(self):

        if self.driver != False:
            try:
                self.driver.close()
                self.driver.open()
                stream = self.driver.readline()
                command = stream.decode("utf-8","ignore")
                return command[:-2]
            except serial.SerialTimeoutException:
                self.message = 'LEDBoard.receive_command('+command+') Serial Timeout Occurred'
                raise

    def color2ch(self, color):
        """ Convert color name to numerical channel """
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        elif color == 'BF':
            return 3
        elif color == 'PC':
            return 4
        elif color == 'EP':
            return 5
        else: # BF
            return 3

    def ch2color(self, channel):
        """ Convert numerical channel to color name """
        if channel == 0:
            return 'Blue'
        elif channel == 1:
            return 'Green'
        elif channel == 2:
            return 'Red'
        elif channel == 3:
            return 'BF'
        elif channel == 4:
            return 'PC'
        elif channel == 5:
            return 'EP'
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
        self.send_command(command)

    def leds_disable(self):
        command = 'LEDS_ENF'
        self.send_command(command)

    def led_on(self, channel, mA):
        """ Turn on LED at channel number at mA power """
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        self.send_command(command)

    def led_off(self, channel):
        """ Turn off LED at channel number """
        command = 'LED' + str(int(channel)) + '_OFF'
        self.send_command(command)

    def leds_off(self):
        """ Turn off all LEDs """
        command = 'LEDS_OFF'
        self.send_command(command)
