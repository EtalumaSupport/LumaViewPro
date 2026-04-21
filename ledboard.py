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

import threading

class LEDBoard:    

    #----------------------------------------------------------
    # Initialize connection through microcontroller
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        logger.info('[LED Class ] LEDBoard.__init__()')
        ports = list_ports.comports(include_links = True)
        self.found = False

        self.com_lock = threading.RLock()

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
            if self.found:
                logger.info('[LED Class ] Found LED controller and about to establish connection.')
                self.connect()
            else:
                logger.warning('[LED Class ] LED controller not found; running with inactive LED board.')
        except:
            logger.exception('[LED Class ] Found LED controller but unable to establish connection.')
            raise


    def connect(self):
        """ Try to connect to the LED controller based on the known VID/PID"""
        with self.com_lock:
            if (self.found is False) or (not hasattr(self, 'port')):
                self.driver = False
                logger.warning('[LED Class ] LEDBoard.connect() skipped; LED controller not found')
                return

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

        with self.com_lock:

            stream = command.encode('utf-8')+b"\n"

            if self.driver != False:
                try:
                    self.driver.flushInput()
                    self.driver.flush()
                    time.sleep(0.001)
                    self.driver.write(stream)
                    time.sleep(0.01)
                    response = self.driver.readline()
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
    
    def _write_command_fast(self, command: str):
        """Write-only fast path: send command without sleeps or reading a response."""
        stream = command.encode('utf-8')+b"\n"
        if self.driver != False:
            try:
                with self.com_lock:
                    self.driver.write(stream)
            except Exception as ex:
                logger.warning(f'[LED Class ] _write_command_fast({command}) write failed: {ex}')
        else:
            # If not connected, attempt to connect quickly; if it fails, just return
            try:
                self.connect()
                if self.driver:
                    with self.com_lock:
                        self.driver.write(stream)
            except Exception as ex:
                logger.warning(f'[LED Class ] _write_command_fast({command}) reconnect/write failed: {ex}')
      
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

    def supports_firmware_stim(self):
        """Probe firmware for STIM command support. Result cached after first call.

        Needed because host-side pulse scheduling is unreliable at short pulse
        widths — the USB-UART bridge batches back-to-back writes so the firmware
        sees ON/OFF ~3 ms apart regardless of host spacing (measured 2026-04-20).
        Firmware-side STIM (v3.0.8+) runs the pulse train in firmware with
        sub-microsecond timing accuracy.
        """
        if hasattr(self, '_supports_stim_cached'):
            return self._supports_stim_cached
        with self.com_lock:
            if self.driver is False:
                return False
            # STIM 0 0 1 2 1 is intentionally invalid (mA=0) — the STIM parser
            # rejects with "STIM: mA must be > 0". Pre-v3.0.8 firmware returns
            # "Command not recognized". Either tells us whether STIM exists.
            saved_timeout = self.driver.timeout
            self.driver.timeout = 1.0
            try:
                self.driver.flushInput()
                self.driver.flush()
                self.driver.write(b'STIM 0 0 1 2 1\n')
                got_stim = False
                import time as _t
                deadline = _t.time() + 2.5
                while _t.time() < deadline:
                    line = self.driver.readline()
                    if not line:
                        continue  # readline timeout — keep waiting until deadline
                    s = line.decode('utf-8', 'ignore').strip()
                    # Distinguish v3.0.8+ ("STIM: mA must be > 0" or "STIM_DIAG:"
                    # progress prints) from pre-v3.0.8 (which echoes the unknown
                    # command and emits " : Command not recognized" suffix —
                    # also starts with "STIM" because the firmware echoes
                    # arguments). Detect the not-recognized case explicitly.
                    if 'Command not recognized' in s:
                        got_stim = False
                        break
                    # v3.0.8 production firmware: "STIM: <error>" prefix.
                    if s.startswith('STIM:'):
                        got_stim = True
                        break
                    # v3.0.8 diag build: "STIM_DIAG: <progress>" prefix.
                    if s.startswith('STIM_DIAG:'):
                        got_stim = True
                        break
                    # Anything else (RE: echo, stale bytes, noise) — keep waiting.
                # Drain anything still in transit so subsequent commands see
                # a clean buffer (probe broke at first match; firmware may have
                # more diag/error lines en route).
                _t.sleep(0.2)
                if self.driver.in_waiting:
                    self.driver.read(self.driver.in_waiting)
                self._supports_stim_cached = got_stim
                logger.info('[LED Class ] firmware STIM support: %s', got_stim)
                return got_stim
            except Exception:
                logger.exception('[LED Class ] supports_firmware_stim probe failed')
                self._supports_stim_cached = False
                return False
            finally:
                if self.driver is not False:
                    self.driver.timeout = saved_timeout

    def stim_pulse_train(self, channel, mA, pulse_width_ms, period_ms, pulse_count):
        """Fire a pulse train via firmware STIM command (v3.0.8+).

        Returns True on firmware confirmation, False on timeout / error.
        Blocks for approximately (pulse_count * period_ms) plus round-trip.
        Caller should confirm supports_firmware_stim() first.
        """
        import time as _t
        with self.com_lock:
            if self.driver is False:
                logger.error('[LED Class ] stim_pulse_train: not connected')
                return False

            cmd = 'STIM {} {} {} {} {}'.format(
                int(channel), int(round(mA)), int(pulse_width_ms),
                int(period_ms), int(pulse_count))

            # Expected train duration + margin
            timeout_s = (pulse_count * period_ms) / 1000.0 + 3.0
            saved_timeout = self.driver.timeout
            self.driver.timeout = max(timeout_s, 3.0)

            try:
                # Drop the pyserial driver.flush() + sleep(0.001) pattern
                # used elsewhere on OG — under RLock on Windows VCP drivers
                # it intermittently returned late and the STIM command bytes
                # never reached the LED firmware. Symptom: LVP logged
                # stim_pulse_train timed out at the deadline AND no pulses
                # fired on the bench (2026-04-20 stim7.log, 2/15 commands).
                # 4.1 firmware-stim glue uses just reset_input_buffer + write
                # and has not reproduced the issue on the same firmware +
                # hardware.
                self.driver.flushInput()
                self.driver.write(cmd.encode('utf-8') + b'\n')

                # Use a short per-readline timeout inside an outer deadline loop.
                # Never break on empty-read alone — the firmware is busy-waiting
                # during the train and only prints the completion line at the end,
                # so we expect long stretches of no data before the real response.
                self.driver.timeout = 0.5
                deadline = _t.time() + timeout_s + 1.0
                while _t.time() < deadline:
                    line = self.driver.readline()
                    if not line:
                        continue
                    s = line.decode('utf-8', 'ignore').strip()
                    # Success: "STIM <ch> complete: ..."
                    if s.startswith('STIM ') and 'complete' in s:
                        logger.info('[LED Class ] stim_pulse_train(%s) OK: %s', cmd, s)
                        color = self.ch2color(channel=channel)
                        if color:
                            self.led_ma[color] = -1
                        return True
                    # Firmware-level error: "STIM: <reason>" (note the colon
                    # immediately after STIM with no channel; distinguishes
                    # error lines from STIM_DIAG: progress prints).
                    if s.startswith('STIM:'):
                        logger.warning('[LED Class ] stim_pulse_train firmware error: %s', s)
                        return False
                    # Anything else (RE: echo, STIM_DIAG: progress, unrelated
                    # noise) — keep reading until completion or timeout.
                logger.warning('[LED Class ] stim_pulse_train(%s) timed out', cmd)
                return False
            except Exception:
                logger.exception('[LED Class ] stim_pulse_train exception')
                return False
            finally:
                if self.driver is not False:
                    self.driver.timeout = saved_timeout


