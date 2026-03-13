# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
SerialBoard — base class for RP2040-based serial controllers.

Shared infrastructure for LEDBoard and MotorBoard: port discovery,
connect/disconnect, firmware version detection, serial exchange with
auto-reconnect and echo handling.
"""

import re
import time
import serial
import serial.tools.list_ports as list_ports
from lvp_logger import logger
import threading


class SerialBoard:

    def __init__(self, vid, pid, label, timeout=0.1, write_timeout=0.1):
        self._lock = threading.RLock()
        self._vid = vid
        self._pid = pid
        self._label = label
        self.found = False
        self.port = None
        self.firmware_version = None
        self.driver = None
        self._last_error_log_time = 0.0
        self._error_log_interval = 2.0  # seconds between repeated error logs
        self._min_command_interval = 0.0  # seconds; 0 = no rate limit (subclass can override)
        self._last_command_time = 0.0
        self.baudrate = 115200
        self.bytesize = serial.EIGHTBITS
        self.parity = serial.PARITY_NONE
        self.stopbits = serial.STOPBITS_ONE
        self.timeout = timeout
        self.write_timeout = write_timeout
        self._find_port()

    def _find_port(self):
        """Search for serial port matching VID/PID."""
        ports = list_ports.comports(include_links=True)
        for port in ports:
            if port.vid == self._vid and port.pid == self._pid:
                self.port = port.device
                self.found = True
                logger.info(f'{self._label} Found device at {port.device}')
                break

    # ------------------------------------------------------------------
    # Connection helpers (used by connect)
    # ------------------------------------------------------------------
    def _open_serial(self):
        """Open serial port and create driver."""
        if self.port is None:
            raise ValueError(f"No port found for {self._label}")
        self.driver = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout,
            write_timeout=self.write_timeout)

    def _reset_firmware(self):
        """Send Ctrl-D reset and detect firmware version."""
        self.driver.write(b'\x04\n')
        logger.debug(f'{self._label} Port initial state: %r' % self.driver.readline())
        self._detect_firmware_version()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        """Open serial connection, reset firmware, detect version."""
        with self._lock:
            try:
                self._open_serial()
                self._reset_firmware()
                if self.firmware_version is not None:
                    logger.info(f'{self._label} Connected (firmware v{self.firmware_version})')
                else:
                    logger.info(f'{self._label} Connected (legacy firmware)')
            except Exception as e:
                self._close_driver()
                logger.error(f'{self._label} connect() failed: {e}')

    def disconnect(self):
        """Close serial connection."""
        logger.info(f'{self._label} Disconnecting...')
        with self._lock:
            try:
                if self.driver is not None:
                    self._close_driver()
                    self.port = None
                    logger.info(f'{self._label} disconnect() succeeded')
                else:
                    logger.info(f'{self._label} disconnect(): not connected')
            except Exception as e:
                self._close_driver()
                logger.error(f'{self._label} disconnect() failed: {e}')

    def is_connected(self) -> bool:
        with self._lock:
            return self.driver is not None

    def _close_driver(self):
        """Safely close and clear the serial driver."""
        try:
            if self.driver is not None:
                self.driver.close()
        except Exception as e:
            logger.debug(f'{self._label} _close_driver() ignored: {e}')
        self.driver = None

    # ------------------------------------------------------------------
    # Firmware version
    # ------------------------------------------------------------------
    def _detect_firmware_version(self):
        """Query INFO and parse firmware version string."""
        try:
            resp = self.exchange_command('INFO')
            if resp and ' v' in resp:
                match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', resp)
                if match:
                    self.firmware_version = match.group(1)
                    logger.info(f'{self._label} Firmware version: {self.firmware_version}')
                    return
            self.firmware_version = None
            logger.info(f'{self._label} Legacy firmware (no version string)')
        except Exception as e:
            logger.debug(f'{self._label} version detection failed: {e}')
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

    # ------------------------------------------------------------------
    # Serial communication
    # ------------------------------------------------------------------
    def exchange_command(self, command, response_numlines=1):
        """Send command and read response(s).

        Handles auto-reconnect, LED echo detection (RE: prefix),
        multi-line responses, and firmware error logging.
        """
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception as e:
                    logger.error(f'{self._label} exchange_command({command}) reconnect failed: {e}')
                    return None

            if self.driver is None:
                return None

            # Rate limiting: enforce minimum interval between commands
            min_interval = getattr(self, '_min_command_interval', 0)
            if min_interval > 0:
                elapsed = time.monotonic() - getattr(self, '_last_command_time', 0.0)
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                self._last_command_time = time.monotonic()

            stream = command.encode('utf-8') + b"\n"
            try:
                # Flush any stale data in the input buffer before writing.
                # If a previous readline() timed out, the firmware's response
                # is still sitting in the buffer and would be misread as this
                # command's response, causing a permanent desync cascade.
                stale = self.driver.in_waiting
                if stale > 0:
                    discarded = self.driver.read(stale)
                    logger.debug(f'{self._label} Flushed {stale} stale bytes: {discarded!r}')

                self.driver.write(stream)
                resp_lines = []
                for _ in range(response_numlines):
                    line = self.driver.readline().decode("utf-8", "ignore").strip()
                    # Auto-detect and drain LED echo
                    if line.startswith('RE:'):
                        line = self.driver.readline().decode("utf-8", "ignore").strip()
                    resp_lines.append(line)

                response = resp_lines[0] if response_numlines == 1 else resp_lines
                # Truncate debug output to avoid logging large binary/config responses
                resp_repr = repr(response)
                if len(resp_repr) > 200:
                    resp_repr = resp_repr[:200] + '...'
                logger.debug(f'{self._label} exchange_command({command}) -> {resp_repr}')

                resp_str = str(response)
                if 'ERROR' in resp_str or 'FAIL' in resp_str or 'exceeds safe' in resp_str:
                    logger.warning(f'{self._label} Firmware error for {command}: {response}')

                return response

            except serial.SerialTimeoutException:
                now = time.monotonic()
                last = getattr(self, '_last_error_log_time', 0.0)
                interval = getattr(self, '_error_log_interval', 2.0)
                if now - last >= interval:
                    logger.warning(f'{self._label} exchange_command({command}) Serial Timeout')
                    self._last_error_log_time = now

            except Exception as e:
                now = time.monotonic()
                last = getattr(self, '_last_error_log_time', 0.0)
                interval = getattr(self, '_error_log_interval', 2.0)
                if now - last >= interval:
                    logger.error(f'{self._label} exchange_command({command}) failed: {e}')
                    self._last_error_log_time = now
                self._close_driver()

            return None

    def _write_command_fast(self, command: str):
        """Write-only fast path: send command without reading a response."""
        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return

            if self.driver is None:
                return

            stream = command.encode('utf-8') + b"\n"
            try:
                self.driver.write(stream)
            except Exception as e:
                now = time.monotonic()
                last = getattr(self, '_last_error_log_time', 0.0)
                interval = getattr(self, '_error_log_interval', 2.0)
                if now - last >= interval:
                    logger.error(f'{self._label} _write_command_fast({command}) failed: {e}')
                    self._last_error_log_time = now
                self._close_driver()
