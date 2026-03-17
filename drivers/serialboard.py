# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
SerialBoard — base class for RP2040-based serial controllers.

Shared infrastructure for LEDBoard and MotorBoard: port discovery,
connect/disconnect, firmware version detection, serial exchange with
auto-reconnect and echo handling, and raw REPL file operations
(config backup, firmware flash, INI updates).
"""

import re
import time
import serial
import serial.tools.list_ports as list_ports
from enum import Enum
from lvp_logger import logger
import threading

from drivers.raw_repl import (
    enter_raw_repl as _enter_raw_repl,
    exit_raw_repl as _exit_raw_repl,
    list_files as _list_files,
    read_file as _read_file,
    write_file as _write_file,
    verify_firmware_running as _verify_firmware_running,
)


class ProtocolVersion(Enum):
    LEGACY = "legacy"  # All pre-v3.0 firmware (including v2.0 dev builds)
    V3 = "v3"          # v3.0 JSON Lines protocol


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
        self._in_raw_repl = False
        self.protocol_version = ProtocolVersion.LEGACY
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
        """Query INFO and parse firmware version string.

        Also detects protocol version: if INFO response starts with '{'
        it's v3.0 JSON Lines; otherwise LEGACY.
        """
        try:
            resp = self.exchange_command('INFO')
            if resp:
                # v3.0 STUB: JSON Lines protocol detection
                # v3.0 firmware responds with JSON: {"cmd": "INFO", ...}
                if resp.lstrip().startswith('{'):
                    self.protocol_version = ProtocolVersion.V3
                    logger.info(f'{self._label} Detected v3.0 JSON Lines protocol')
                else:
                    self.protocol_version = ProtocolVersion.LEGACY
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
            self.protocol_version = ProtocolVersion.LEGACY

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

    def _build_command(self, cmd):
        """Build command string for current protocol version."""
        # v3.0 STUB: JSON command format
        # if self.protocol_version == ProtocolVersion.V3:
        #     import json
        #     return json.dumps({"cmd": cmd}) + "\n"
        return cmd + "\n"

    def _parse_response(self, response):
        """Parse response for current protocol version."""
        # v3.0 STUB: JSON Lines response parsing
        # if self.protocol_version == ProtocolVersion.V3:
        #     import json
        #     return json.loads(response)
        return response

    # ------------------------------------------------------------------
    # Serial communication
    # ------------------------------------------------------------------
    def exchange_command(self, command, response_numlines=1, timeout=None):
        """Send command and read response(s).

        Handles auto-reconnect, LED echo detection (RE: prefix),
        multi-line responses, and firmware error logging.

        Args:
            command: Serial command string to send.
            response_numlines: Number of response lines to read.
            timeout: Per-call read timeout in seconds. If provided,
                temporarily overrides the board's default timeout for
                this call only. Useful for long-running commands like
                HOME (5-15s) or CALIBRATE (30-60s).
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

            # Per-call timeout override
            saved_timeout = None
            if timeout is not None and self.driver is not None:
                saved_timeout = self.driver.timeout
                self.driver.timeout = timeout

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

            finally:
                if saved_timeout is not None and self.driver is not None:
                    self.driver.timeout = saved_timeout

            return None

    def exchange_multiline(self, command, timeout=60, end_markers=None):
        """Send command and read variable-length multi-line response.

        Reads lines until an end marker is found, no more data arrives,
        or the overall timeout expires.  LED echo lines (RE: prefix) are
        automatically stripped.

        Args:
            command: Serial command string to send.
            timeout: Overall timeout in seconds for the entire response.
            end_markers: List of strings to check for in each line
                (case-insensitive). When found, reads a few more drain
                lines then stops.  Defaults to common completion markers.

        Returns:
            Joined multi-line string, or None on error.
        """
        if end_markers is None:
            end_markers = ['PASS', 'FAIL', 'COMPLETE', 'DONE', 'ERROR']

        with self._lock:
            if self.driver is None:
                try:
                    self.connect()
                except Exception as e:
                    logger.error(f'{self._label} exchange_multiline({command}) reconnect failed: {e}')
                    return None
            if self.driver is None:
                return None

            saved_timeout = self.driver.timeout
            self.driver.timeout = min(timeout, 5.0)  # per-readline timeout

            try:
                # Flush stale data
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)

                self.driver.write(command.encode('utf-8') + b'\n')
                lines = []
                start = time.monotonic()
                while time.monotonic() - start < timeout:
                    raw = self.driver.readline()
                    if not raw:
                        if lines:
                            break
                        continue
                    line = raw.decode('utf-8', 'ignore').strip()
                    if line.startswith('RE:'):
                        continue
                    if line:
                        lines.append(line)
                    if any(m in line.upper() for m in [em.upper() for em in end_markers]):
                        # Drain a few trailing lines
                        for _ in range(5):
                            extra = self.driver.readline()
                            if extra:
                                decoded = extra.decode('utf-8', 'ignore').strip()
                                if decoded and not decoded.startswith('RE:'):
                                    lines.append(decoded)
                        break

                result = '\n'.join(lines) or None
                logger.debug(f'{self._label} exchange_multiline({command}) -> {len(lines)} lines')
                return result

            except serial.SerialTimeoutException:
                logger.warning(f'{self._label} exchange_multiline({command}) timeout')
                return '\n'.join(lines) if lines else None

            except Exception as e:
                logger.error(f'{self._label} exchange_multiline({command}) failed: {e}')
                self._close_driver()
                return None

            finally:
                if self.driver is not None:
                    self.driver.timeout = saved_timeout

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

    # ------------------------------------------------------------------
    # Raw REPL — file operations on board filesystem
    # ------------------------------------------------------------------
    def enter_raw_repl(self):
        """Interrupt firmware and enter MicroPython raw REPL.

        While in raw REPL, normal commands (exchange_command) cannot be
        used. Call exit_raw_repl() when done to reboot the firmware.

        Returns True on success, False on failure.
        """
        with self._lock:
            if self.driver is None:
                self._open_serial()
            if _enter_raw_repl(self.driver):
                self._in_raw_repl = True
                logger.info(f'{self._label} Entered raw REPL')
                return True
            logger.error(f'{self._label} Failed to enter raw REPL')
            return False

    def exit_raw_repl(self):
        """Exit raw REPL and reboot firmware.

        After exit, the board reboots and firmware resumes. The serial
        connection remains open — call exchange_command() normally after.
        """
        with self._lock:
            if self.driver is None:
                return
            _exit_raw_repl(self.driver)
            self._in_raw_repl = False
            logger.info(f'{self._label} Exited raw REPL, firmware rebooting')

    def repl_list_files(self):
        """List files on board filesystem (must be in raw REPL).

        Returns list of filenames, or empty list on failure.
        """
        with self._lock:
            if not self._in_raw_repl or self.driver is None:
                logger.error(f'{self._label} repl_list_files: not in raw REPL')
                return []
            return _list_files(self.driver)

    def repl_read_file(self, filename, verify=True):
        """Read a file from the board (must be in raw REPL).

        Returns file contents as bytes, or None on failure.
        """
        with self._lock:
            if not self._in_raw_repl or self.driver is None:
                logger.error(f'{self._label} repl_read_file: not in raw REPL')
                return None
            return _read_file(self.driver, filename, verify=verify)

    def repl_write_file(self, filename, data):
        """Write a file to the board with SHA256 verification (must be in raw REPL).

        Atomic write with backup: writes to .tmp, verifies SHA256,
        backs up existing file to .bak, then renames.

        Returns True on success, False on failure.
        """
        with self._lock:
            if not self._in_raw_repl or self.driver is None:
                logger.error(f'{self._label} repl_write_file: not in raw REPL')
                return False
            return _write_file(self.driver, filename, data)

    def verify_firmware_running(self, timeout=10):
        """Verify firmware is responding after raw REPL exit.

        Returns firmware response string, or None if not responding.
        """
        with self._lock:
            if self.driver is None:
                return None
            return _verify_firmware_running(self.driver, timeout=timeout)
