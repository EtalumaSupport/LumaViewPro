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

import re
from lvp_logger import logger
from serialboard import SerialBoard


class LEDBoard(SerialBoard):

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(vid=0x0424, pid=0x704C, label='[LED Class ]')

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

    def read_led_current(self, channel):
        """Read measured LED current (mA) from ADC feedback. Requires v2.0+ firmware in engineering mode.
        Returns measured current in mA, or None on error/unsupported."""
        if not self.is_v2:
            return None
        with self._lock:
            if self.driver is None:
                return None
            command = f'LEDREAD{int(channel)}'
            try:
                self.driver.write((command + '\n').encode('utf-8'))
                # Firmware sends: echo, I_SENS line, LED_K line
                lines = []
                for _ in range(4):  # echo + up to 2 data lines + margin
                    line = self.driver.readline().decode('utf-8', 'ignore').strip()
                    if not line:
                        break
                    lines.append(line)
                # Parse I_SENS line: "LED0 I_SENS  (AIN14): 1.2800V  ->   200.1 mA"
                for line in lines:
                    if 'I_SENS' in line and 'mA' in line:
                        m = re.search(r'([\d.]+)\s*mA', line)
                        if m:
                            return float(m.group(1))
            except Exception as e:
                logger.error(f'[LED Class ] read_led_current({channel}) failed: {e}')
            return None
