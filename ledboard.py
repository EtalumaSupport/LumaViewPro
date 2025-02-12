#!/usr/bin/python3

'''
MIT License

Copyright (c) 2024 Etaluma, Inc.

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
Gerard Decker, The Earthineering Company
'''

from numpy import False_
import serial
import serial.tools.list_ports as list_ports
from lvp_logger import logger
import time

class LEDBoard:    

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        logger.info('[LED Class ] LEDBoard.__init__()')
        ports = list_ports.comports(include_links = True)
        self.found = False

        for port in ports:
            if (port.vid == 0x0424) and (port.pid == 0x704C):
                logger.info(f'[LED Class ] LED Controller at {port.device}')
                self.port = port.device
                self.found = True
                break

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=0.1 # seconds
        self.write_timeout=0.1 # seconds
        self.driver = False
        self.led_ma = {
            'BF': -1,
            'PC': -1,
            'DF': -1,
            'Red': -1,
            'Blue': -1,
            'Green': -1,
        }

        try:
            logger.info('[LED Class ] Found LED controller and about to establish connection.')
            self.connect()
        except:
            logger.exception('[LED Class ] Found LED controller but unable to establish connection.')
            raise


    def connect(self):
        """ Try to connect to the LED controller based on the known VID/PID"""
        try:
            logger.info('[LED Class ] Found LED controller and about to create driver.')
            self.driver = serial.Serial(port=self.port,
                                        baudrate=self.baudrate,
                                        bytesize=self.bytesize,
                                        parity=self.parity,
                                        stopbits=self.stopbits,
                                        timeout=self.timeout,
                                        write_timeout=self.write_timeout)

            #self.driver.close()
            #self.driver.open()

            # self.exchange_command('import main.py')
            # self.exchange_command('import main.py')
            logger.info('[LED Class ] LEDBoard.connect() succeeded')
            #Sometimes the firmware fails to start (or the port has a \x00 left in the buffer), this forces MicroPython to reset, and the normal firmware just complains 
            self.driver.write(b'\x04\n')
            logger.debug('[LED Class ] LEDBOARD.connect() port initial state: %r'%self.driver.readline())
        except:
            self.driver = False
            logger.exception('[LED Class ] LEDBoard.connect() failed')
            
    def exchange_command(self, command):
        """ Exchange command through serial to LED board
        This should NOT be used in a script. It is intended for other functions to access"""

        stream = command.encode('utf-8')+b"\n"

        if self.driver != False:
            try:
                self.driver.write(stream)
                response = self.driver.readline()
                self.driver.flushInput()
                self.driver.flush()
                response = response.decode("utf-8","ignore")

                logger.info('[LED Class ] LEDBoard.exchange_command('+command+') succeeded: %r'%response)
                return response[:-2]
            
            except serial.SerialTimeoutException:
                self.driver = False
                logger.exception('[LED Class ] LEDBoard.exchange_command('+command+') Serial Timeout Occurred')

            except:
                self.driver = False

        else:
            try:
                self.connect()
            except:
                return
      
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
        elif color == 'DF':
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
            return 'DF'
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
        self.exchange_command(command)

    def leds_disable(self):
        for color, mA in self.led_ma.items():
            self.led_ma[color] = -1

        command = 'LEDS_ENF'
        self.exchange_command(command)

    def get_led_ma(self, color):
        return self.led_ma.get(color, -1)
    

    def is_led_on(self, color) -> bool:
        mA = self.led_ma[color]
        if mA > 0:
            return True
        else:
            return False
        
    
    def get_led_state(self, color) -> dict:
        enabled = self.is_led_on(color=color)
        mA = self.get_led_ma(color=color)

        return {
            'enabled': enabled,
            'illumination': mA,
        }
    

    def get_led_states(self) -> dict:
        states = {}
        for color in self.led_ma.keys():
            states[color] = self.get_led_state(color=color)

        return states
        
    
    def led_on(self, channel, mA):
        """ Turn on LED at channel number at mA power """
        color = self.ch2color(channel=channel)
        self.led_ma[color] = mA

        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        self.exchange_command(command)

    def led_off(self, channel):
        """ Turn off LED at channel number """
        color = self.ch2color(channel=channel)
        self.led_ma[color] = -1

        command = 'LED' + str(int(channel)) + '_OFF'
        self.exchange_command(command)

    def leds_off(self):
        """ Turn off all LEDs """
        for color, mA in self.led_ma.items():
            self.led_ma[color] = -1

        command = 'LEDS_OFF'
        self.exchange_command(command)


