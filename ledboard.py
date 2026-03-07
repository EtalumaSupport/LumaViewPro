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

import serial
import serial.tools.list_ports as list_ports
from lvp_logger import logger
import time
import threading

class LEDBoard:

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        ports = list_ports.comports(include_links = True)
        self.found = False
        self._lock = threading.RLock()  # single lock for ALL serial access
        self.port = None
        self.firmware_version = None  # Detected on connect (e.g. '2.0.1' or None for legacy)

        for port in ports:
            if (port.vid == 0x0424) and (port.pid == 0x704C):
                self.port = port.device
                self.found = True
                logger.info(f'[LED Class ] Found LED controller at {port.device}')
                break

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=0.1 # seconds
        self.write_timeout=0.1 # seconds
        self.driver = None
        self.led_ma = {
            'BF': -1,
            'PC': -1,
            'DF': -1,
            'Red': -1,
            'Blue': -1,
            'Green': -1,
        }

        try:
            self.connect()
        except Exception:
            logger.error('[LED Class ] Failed to connect to LED controller')
            raise


    def connect(self):
        """ Try to connect to the LED controller based on the known VID/PID"""
        with self._lock:
            try:
                if self.port is None:
                    raise ValueError("No port found for LED controller")

                self.driver = serial.Serial(port=self.port,
                                            baudrate=self.baudrate,
                                            bytesize=self.bytesize,
                                            parity=self.parity,
                                            stopbits=self.stopbits,
                                            timeout=self.timeout,
                                            write_timeout=self.write_timeout)

                # Reset firmware in case of stale buffer state
                self.driver.write(b'\x04\n')
                logger.debug('[LED Class ] Port initial state: %r' % self.driver.readline())
                logger.info('[LED Class ] Connected to LED controller')

                # Detect firmware version
                self._detect_firmware_version()
            except Exception as e:
                self._close_driver()
                logger.error(f'[LED Class ] LEDBoard.connect() failed: {e}')

    def disconnect(self):
        logger.info('[LED Class ] Disconnecting from LED controller...')
        with self._lock:
            try:
                if self.driver is not None:
                    self._close_driver()
                    self.port = None
                    logger.info('[LED Class ] LEDBoard.disconnect() succeeded')
                else:
                    logger.info('[LED Class ] LEDBoard.disconnect(): not connected')
            except Exception as e:
                self._close_driver()
                logger.error(f'[LED Class ] LEDBoard.disconnect() failed: {e}')

    def is_connected(self) -> bool:
        with self._lock:
            return self.driver is not None

    def exchange_command(self, command):
        """ Exchange command through serial to LED board
        This should NOT be used in a script. It is intended for other functions to access"""
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return None

            if self.driver is None:
                return None

            stream = command.encode('utf-8')+b"\n"
            try:
                self.driver.write(stream)

                # Firmware sends an echo line ("RE: <cmd>\r\n") then a result
                # line. Read the echo, then read the actual result.
                echo = self.driver.readline().decode("utf-8", "ignore").strip()

                if echo.startswith('RE:'):
                    # New behavior: drain echo, read actual result
                    response = self.driver.readline().decode("utf-8", "ignore").strip()
                else:
                    # Future firmware without echo, or unexpected response
                    response = echo

                logger.debug(f'[LED Class ] LEDBoard.exchange_command({command}) -> {response!r}')
                return response

            except serial.SerialTimeoutException:
                logger.error(f'[LED Class ] LEDBoard.exchange_command({command}) Serial Timeout')
                self._close_driver()

            except Exception as e:
                logger.error(f'[LED Class ] LEDBoard.exchange_command({command}) failed: {e}')
                self._close_driver()

            return None
    
    def _detect_firmware_version(self):
        """Query INFO and parse firmware version string.

        v2.0+ firmware responds with lines like:
            Firmware:     2026-03-06 v2.0.1
        Legacy firmware has no 'v' version string.
        """
        try:
            resp = self.exchange_command('INFO')
            if resp and ' v' in resp:
                # Parse version from "... v2.0.1" or "Firmware: ... v2.0.1"
                import re
                match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', resp)
                if match:
                    self.firmware_version = match.group(1)
                    logger.info(f'[LED Class ] Firmware version: {self.firmware_version}')
                    return
            self.firmware_version = None
            logger.info(f'[LED Class ] Legacy firmware (no version string)')
        except Exception:
            self.firmware_version = None

    @property
    def is_v2(self) -> bool:
        """True if firmware is v2.0 or later."""
        if self.firmware_version is None:
            return False
        try:
            major = int(self.firmware_version.split('.')[0])
            return major >= 2
        except (ValueError, IndexError):
            return False

    def _close_driver(self):
        """Safely close and clear the serial driver."""
        try:
            if self.driver is not None:
                self.driver.close()
        except Exception:
            pass
        self.driver = None

    def _write_command_fast(self, command: str):
        """Write-only fast path: send command without sleeps or reading a response.
        Uses the same lock as exchange_command to prevent interleaved writes."""
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return

            if self.driver is None:
                return

            stream = command.encode('utf-8')+b"\n"
            try:
                self.driver.write(stream)
            except Exception as e:
                logger.error(f'[LED Class ] LEDBoard._write_command_fast({command}) failed: {e}')
                self._close_driver()
      
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

    def get_status(self):
        command = 'STATUS'
        return self.exchange_command(command)
    
    def wait_until_on(self):
        # Waits in loop until ledboard confirms that an LED is on (not turned off)

        status = self.get_status()
        while status is None or "STATUS" not in status:
            status = self.get_status()

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
        
    
    def led_on(self, channel, mA, block=False):
        """ 
        Turn on LED at channel number at mA power 
        If block=True, verify correct callback before returning
        """
        color = self.ch2color(channel=channel)
        self.led_ma[color] = mA

        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        response = self.exchange_command(command)

        def check_each_substr(list, result):
            for sub_str in list:
                if sub_str not in result:
                    return False
            return True

        if block:
            while response is None or (command not in response and not check_each_substr(['LED', str(int(channel)), str(int(mA))], response)):
                response = self.exchange_command(command)


    def led_off(self, channel):
        """ Turn off LED at channel number """
        color = self.ch2color(channel=channel)
        self.led_ma[color] = -1

        command = 'LED' + str(int(channel)) + '_OFF'
        self.exchange_command(command)

    def led_on_fast(self, channel, mA):
        """Fast write-only version of led_on for time-critical toggling."""
        color = self.ch2color(channel=channel)
        self.led_ma[color] = mA
        command = 'LED' + str(int(channel)) + '_' + str(int(mA))
        self._write_command_fast(command)

    def led_off_fast(self, channel):
        """Fast write-only version of led_off for time-critical toggling."""
        color = self.ch2color(channel=channel)
        self.led_ma[color] = -1
        command = 'LED' + str(int(channel)) + '_OFF'
        self._write_command_fast(command)

    def leds_off(self):
        """ Turn off all LEDs """
        for color, mA in self.led_ma.items():
            self.led_ma[color] = -1

        command = 'LEDS_OFF'
        self.exchange_command(command)

    def leds_off_fast(self):
        """Fast write-only version to turn off all LEDs."""
        for color, mA in self.led_ma.items():
            self.led_ma[color] = -1
        command = 'LEDS_OFF'
        self._write_command_fast(command)


