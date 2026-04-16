# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
SerialBoard — base class for RP2040-based serial controllers.

Shared infrastructure for LEDBoard and MotorBoard: port discovery,
connect/disconnect, firmware version detection, serial exchange with
auto-reconnect and echo handling, and raw REPL file operations
(config backup, firmware flash, INI updates).
"""

import logging
import re
import time
import serial
import serial.tools.list_ports as list_ports
from enum import Enum
from lvp_logger import logger
import threading

_serial_log = logging.getLogger('LVP.serial')

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

    def __init__(self, vid, pid, label, timeout=0.1, write_timeout=0.1, port=None):
        self._lock = threading.RLock()
        self._vid = vid
        self._pid = pid
        self._label = label
        self.found = False
        self.port = None
        self.firmware_version = None
        self.firmware_date = None
        self.firmware_responding = False
        # True iff the board sent ZERO bytes across the entire connect
        # sequence (drain steps + every detection attempt). Distinct
        # from firmware_responding=False, which also covers pre-v3.0
        # legacy boards that answer INFO with unparseable text. A
        # silent board is hung (or the port/hub is stuck) and needs
        # a hardware power cycle — see #619. Callers should check
        # this before issuing commands; exchange_command fails fast.
        self.firmware_silent = False
        # Running total of non-empty bytes captured by
        # _detect_firmware_version(). _reset_firmware() reads the
        # delta to know whether each detection attempt saw any bytes,
        # which is how it distinguishes "silent board" from "board
        # that responded with garbage."
        self._detect_response_bytes = 0
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
        if port is not None:
            self.port = port
            self.found = True
        else:
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
        try:
            self.driver = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout,
                write_timeout=self.write_timeout)
        except serial.SerialException:
            # M29: Port may have changed (different USB port) — re-scan.
            logger.info(f'{self._label} Port {self.port} failed, re-scanning...')
            old_port = self.port
            self.port = None
            self.found = False
            self._find_port()
            if self.port and self.port != old_port:
                logger.info(f'{self._label} Found at new port {self.port}')
                self.driver = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=self.bytesize,
                    parity=self.parity,
                    stopbits=self.stopbits,
                    timeout=self.timeout,
                    write_timeout=self.write_timeout)
            else:
                raise
        # Log opened-port state so connect diagnostics are visible in serial.log
        # even when nothing else gets logged (e.g. board goes silent before
        # first response). Added for #619 — "LED board found but totally
        # silent" needs a full diagnostic trail.
        try:
            _serial_log.info(
                f'{self._label} OPEN port={self.port} baud={self.baudrate} '
                f'timeout={self.timeout:.2f}s write_timeout={self.write_timeout:.2f}s '
                f'in_waiting={self.driver.in_waiting}B'
            )
        except Exception as e:
            _serial_log.info(f'{self._label} OPEN port={self.port} (state read failed: {e})')

    def _drain_serial(self):
        """Drain all pending data from the serial buffer.

        Returns the total number of bytes drained. Also logs what was
        drained to serial.log when non-empty — when a board goes silent
        (#619), knowing WHAT was in the buffer before we threw it away
        is the difference between "stale response to the previous
        command" (board was responding, we just missed it) and "USB
        garbage / boot noise" (port is in a bad state).

        The drained content is logged as repr, truncated to 200 chars
        so boot-output floods don't balloon the log file.
        """
        total = 0
        drained = bytearray()
        for _ in range(50):
            n = self.driver.in_waiting
            if n > 0:
                chunk = self.driver.read(n)
                total += len(chunk)
                drained.extend(chunk)
                time.sleep(0.05)
            else:
                saved = self.driver.timeout
                self.driver.timeout = 0.2
                leftover = self.driver.read(4096)
                self.driver.timeout = saved
                if not leftover:
                    break
                total += len(leftover)
                drained.extend(leftover)
        if total > 0:
            content = bytes(drained)
            snippet = repr(content[:200])
            if len(content) > 200:
                snippet = snippet[:-1] + f'...+{len(content) - 200}B)'
            _serial_log.info(f'{self._label} DRAIN {total}B: {snippet}')
        return total

    def _reset_firmware(self):
        """Ensure firmware is running and detect version.

        Handles all common states the board might be in on connect:
          - Normal operation (main.py running) — drain stale data, detect
          - Friendly REPL (>>> prompt) — Ctrl-D soft reset to restart
          - Raw REPL (Thonny left it here) — Ctrl-B to exit, then Ctrl-D
          - Boot output still arriving — drain before commands
          - Old firmware with WDT — Ctrl-D kills WDT timer, so skip it
          - **Silent board (hung firmware / stuck USB hub)** — sends zero
            bytes across everything. Skip destructive Ctrl-D recovery
            and surface as a hard failure (#619).

        Strategy:
          1. Drain stale buffer, try version detection.
          2. If that fails but we saw SOME bytes (drain content or
             partial INFO response), the board is doing something —
             try full recovery with Ctrl-C/B/D soft reset.
          3. If we've seen ZERO bytes, skip the soft reset path
             entirely (it isn't going to help a board that can't even
             echo garbage) and go straight to a gentle Ctrl-C retry.
          4. If the board STILL hasn't sent any byte after all
             retries, mark firmware_silent = True. connect() surfaces
             this as a user-visible error.

        Diagnostic logging (#619 Phase A): every step is logged to
        serial.log with per-step timing, bytes drained/written, and
        driver state.
        """
        t_total_start = time.monotonic()
        _serial_log.info(f'{self._label} RESET begin')
        # Reset per-attempt state so reconnect after a power cycle
        # starts fresh instead of carrying over a stale "silent"
        # verdict from the previous attempt.
        self.firmware_silent = False
        self._detect_response_bytes = 0
        # Tracks whether the board has produced ANY bytes during the
        # entire connect sequence. If this stays 0 through all
        # detection attempts, we're dealing with a hung board and the
        # Ctrl-D soft-reset path is not going to help.
        bytes_ever_seen = 0

        # Step 1: Drain stale data from boot or previous session
        t0 = time.monotonic()
        drained = self._drain_serial()
        bytes_ever_seen += drained
        _serial_log.info(
            f'{self._label} RESET step1 drain: {drained}B '
            f'in {(time.monotonic() - t0) * 1000:.0f}ms'
        )

        # Step 2: Flush board's input buffer — USB CDC enumeration can
        # leave stale bytes (e.g. \x00) that arrive after our drain and
        # get prepended to the first real command. A blank newline makes
        # the board process (and reject) any partial garbage, clearing
        # its input state.
        self._safe_write(b'\n', context='RESET step2 wake newline')
        time.sleep(0.1)
        t0 = time.monotonic()
        drained = self._drain_serial()
        bytes_ever_seen += drained
        _serial_log.info(
            f'{self._label} RESET step2 drain: {drained}B '
            f'in {(time.monotonic() - t0) * 1000:.0f}ms'
        )

        # Step 3: Try version detection — works if firmware is running
        _serial_log.info(f'{self._label} RESET step3 detect (in_waiting={self._safe_in_waiting()}B)')
        t0 = time.monotonic()
        pre_bytes = self._detect_response_bytes
        self._detect_firmware_version()
        bytes_ever_seen += max(0, self._detect_response_bytes - pre_bytes)
        _serial_log.info(
            f'{self._label} RESET step3 detect result: '
            f'responding={self.firmware_responding} '
            f'version={self.firmware_version} '
            f'in {(time.monotonic() - t0) * 1000:.0f}ms'
        )
        if self.firmware_responding:
            _serial_log.info(
                f'{self._label} RESET done (step3 ok) total='
                f'{(time.monotonic() - t_total_start) * 1000:.0f}ms'
            )
            return  # Firmware running (version may or may not be parseable)

        # Step 4: Soft-reset recovery — always attempted.
        #
        # We send Ctrl-C / Ctrl-C / Ctrl-B / Ctrl-D regardless of
        # whether any bytes have been seen so far. The reason
        # matters and is not obvious — an earlier version of this
        # code had a `skip_soft_reset = (bytes_ever_seen == 0)`
        # optimization that bypassed step 4 when no bytes had been
        # seen yet, on the theory "if the board sent nothing, Ctrl-D
        # won't help either." That theory was wrong for an important
        # case, and the skip optimization was reverted.
        #
        # Why we always send Ctrl-D (to recover from a MicroPython
        # REPL state left behind by Thonny or similar tools):
        #
        # A board that appears silent to drain + first INFO detect
        # is NOT necessarily a hung board. The most common benign
        # case is **a board that was just used by Thonny and then
        # disconnected**. Thonny drives MicroPython via raw REPL
        # mode (entered with Ctrl-A), and depending on how the
        # disconnect happened the board can be left in either
        # friendly REPL (`>>>` prompt, idle) or raw REPL (silent,
        # buffered). Either way our first INFO write doesn't reach
        # main.py — the REPL just echoes it as input.
        #
        # Observed example (2026-04-14, LS850T bench after Thonny
        # connect/disconnect cycle): step 4 drain after Ctrl-D
        # captured 252 bytes containing `>>>` followed by
        # `MPY: soft reboot` and the normal v3.0.9 INFO banner —
        # proving the board was sitting at the friendly-REPL prompt
        # and Ctrl-D triggered the soft reset that restarted
        # main.py. The raw-REPL case produces no echo at all but
        # the same Ctrl-D recovery applies.
        #
        # In both cases, `bytes_ever_seen` stays at 0 during steps
        # 1-3 even though the board is perfectly alive and listening.
        # Sending Ctrl-D is exactly what wakes either REPL state up:
        # it tells MicroPython "soft-reset" → main.py restarts →
        # normal operation resumes.
        #
        # The recovery sequence Ctrl-C / Ctrl-C / Ctrl-B / Ctrl-D
        # handles multiple possible pre-startup states:
        #
        #   - Ctrl-C interrupts any in-flight REPL input
        #   - Ctrl-B exits raw REPL back to friendly REPL
        #   - Ctrl-D soft-resets MicroPython → restarts main.py
        #
        # On a board that's truly silent / hung (the in-house bench
        # brick case from #619), this step still fires and costs
        # ~5 extra seconds before we fall through to step 6 and
        # the final silent verdict in step 7. That cost is worth
        # it. Skipping Ctrl-D to save 5 seconds breaks Thonny-user
        # workflows, which is a more common dev scenario than the
        # truly-silent bench-brick case.
        #
        # **Guiding principle for the whole recovery path: be as
        # robust as possible on startup. Try the cheap read-only
        # path first (steps 1-3). If that fails, run the recovery
        # fallback (this step) before declaring the board dead.**
        logger.info(f'{self._label} Firmware not responding — attempting recovery (Ctrl-C/Ctrl-B/Ctrl-D)')
        _serial_log.info(f'{self._label} RESET step4 soft-reset recovery begin')
        t0 = time.monotonic()
        self._safe_write(b'\x03', context='RESET step4 Ctrl-C #1')
        time.sleep(0.2)
        self._safe_write(b'\x03', context='RESET step4 Ctrl-C #2')
        time.sleep(0.2)
        self._safe_write(b'\x02', context='RESET step4 Ctrl-B (raw REPL exit)')
        time.sleep(0.2)
        self._safe_write(b'\x04', context='RESET step4 Ctrl-D (soft reset)')
        time.sleep(5.0)             # Wait for firmware to fully boot

        # Drain all boot output (motor firmware prints SPI init, etc.)
        drained = self._drain_serial()
        bytes_ever_seen += drained
        _serial_log.info(
            f'{self._label} RESET step4 drain after Ctrl-D+5s: {drained}B '
            f'(elapsed {(time.monotonic() - t0) * 1000:.0f}ms)'
        )

        # Step 5: Retry version detection after recovery
        _serial_log.info(f'{self._label} RESET step5 detect (in_waiting={self._safe_in_waiting()}B)')
        t0 = time.monotonic()
        pre_bytes = self._detect_response_bytes
        self._detect_firmware_version()
        bytes_ever_seen += max(0, self._detect_response_bytes - pre_bytes)
        _serial_log.info(
            f'{self._label} RESET step5 detect result: '
            f'responding={self.firmware_responding} '
            f'version={self.firmware_version} '
            f'in {(time.monotonic() - t0) * 1000:.0f}ms'
        )
        if self.firmware_responding:
            _serial_log.info(
                f'{self._label} RESET done (step5 ok after soft reset) total='
                f'{(time.monotonic() - t_total_start) * 1000:.0f}ms'
            )
            return  # Recovery with soft reset worked

        # Step 6: Gentle Ctrl-C-only retry.
        # Soft reset in step 4 can kill WDT on pre-v3.0.4 LED firmware
        # (Ctrl-D kills the Timer that feeds the 8388ms WDT, causing
        # board reset mid-recovery). Ctrl-C keeps the WDT alive and
        # gives the board one more chance to recover before we give
        # up and declare it silent.
        logger.info(f'{self._label} Trying WDT-safe Ctrl-C recovery')
        _serial_log.info(f'{self._label} RESET step6 WDT-safe retry begin')
        t0 = time.monotonic()
        self._safe_write(b'\x03', context='RESET step6 Ctrl-C #1')
        time.sleep(0.2)
        self._safe_write(b'\x03', context='RESET step6 Ctrl-C #2')
        time.sleep(0.5)
        drained = self._drain_serial()
        bytes_ever_seen += drained
        _serial_log.info(
            f'{self._label} RESET step6 drain after Ctrl-C: {drained}B'
        )

        # Send a blank line to exit any partial REPL state, then try INFO
        self._safe_write(b'\n', context='RESET step6 blank newline')
        time.sleep(0.2)
        drained = self._drain_serial()
        bytes_ever_seen += drained
        _serial_log.info(
            f'{self._label} RESET step6 drain after newline: {drained}B '
            f'(elapsed {(time.monotonic() - t0) * 1000:.0f}ms)'
        )

        _serial_log.info(f'{self._label} RESET step7 detect (in_waiting={self._safe_in_waiting()}B)')
        t0 = time.monotonic()
        pre_bytes = self._detect_response_bytes
        self._detect_firmware_version()
        bytes_ever_seen += max(0, self._detect_response_bytes - pre_bytes)
        _serial_log.info(
            f'{self._label} RESET step7 detect result: '
            f'responding={self.firmware_responding} '
            f'version={self.firmware_version} '
            f'in {(time.monotonic() - t0) * 1000:.0f}ms'
        )

        # Final decision: if the board has not produced a single byte
        # across the entire connect sequence, mark it silent so
        # connect() can surface the error and exchange_command() can
        # fail fast. See #619.
        if not self.firmware_responding and bytes_ever_seen == 0:
            self.firmware_silent = True
        _serial_log.info(
            f'{self._label} RESET done (final) total='
            f'{(time.monotonic() - t_total_start) * 1000:.0f}ms '
            f'responding={self.firmware_responding} '
            f'silent={self.firmware_silent} '
            f'bytes_ever_seen={bytes_ever_seen}'
        )

    def _safe_write(self, data: bytes, context: str) -> int:
        """Write raw bytes and log the outcome.

        Used by _reset_firmware() so that every write during the
        connect/recovery sequence shows up in serial.log with its byte
        count, elapsed time, and any exception. When a board goes
        silent (#619), this tells us whether our commands are even
        reaching the OS-level serial driver — critical for telling
        "board not responding" apart from "our write failed."

        Returns the number of bytes written (0 on failure). Exceptions
        are logged but not re-raised so the caller can continue its
        recovery sequence.
        """
        t0 = time.monotonic()
        try:
            n = self.driver.write(data)
            elapsed_ms = (time.monotonic() - t0) * 1000
            _serial_log.info(
                f'{self._label} WRITE {context}: {len(data)}B '
                f'written={n} in {elapsed_ms:.0f}ms'
            )
            return n or len(data)
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _serial_log.error(
                f'{self._label} WRITE {context}: {len(data)}B FAILED '
                f'after {elapsed_ms:.0f}ms ({e})'
            )
            return 0

    def _safe_in_waiting(self) -> int:
        """Return driver.in_waiting with exception handling for logging."""
        try:
            return self.driver.in_waiting
        except Exception:
            return -1

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        """Open serial connection, reset firmware, detect version.

        On a genuinely silent board (zero bytes across entire connect
        sequence — see #619), surfaces a user-visible error notification
        rather than silently degrading to "legacy, no version info."
        The legacy-no-version fallback is preserved for pre-v3.0 boards
        that DO respond to INFO with unparseable bytes.
        """
        with self._lock:
            try:
                self._open_serial()
                self._reset_firmware()
                if self.firmware_version is not None:
                    logger.info(f'{self._label} Connected (firmware v{self.firmware_version})')
                elif self.firmware_date is not None:
                    logger.info(f'{self._label} Connected (legacy firmware, date={self.firmware_date})')
                elif self.firmware_silent:
                    # Board detected on USB but sent zero bytes across
                    # every drain + detection attempt. Not a legacy
                    # firmware case — the firmware is hung or the USB
                    # hub UART bridge is stuck. Needs a hardware power
                    # cycle. See #619.
                    logger.error(
                        f'{self._label} Connected but board is SILENT — '
                        f'zero bytes received during connect sequence'
                    )
                else:
                    logger.info(f'{self._label} Connected (legacy firmware, no version info)')
            except Exception as e:
                self._close_driver()
                logger.error(f'{self._label} connect() failed: {e}')

    def disconnect(self):
        """Close serial connection and clear cached state."""
        logger.info(f'{self._label} Disconnecting...')
        with self._lock:
            try:
                if self.driver is not None:
                    self._close_driver()
                    self.port = None
                    self._on_disconnect()
                    logger.info(f'{self._label} disconnect() succeeded')
                else:
                    logger.info(f'{self._label} disconnect(): not connected')
            except Exception as e:
                self._close_driver()
                self._on_disconnect()
                logger.error(f'{self._label} disconnect() failed: {e}')

    def _on_disconnect(self):
        """Hook for subclasses to clear cached state on disconnect.

        Called under self._lock. Override in LEDBoard/MotorBoard to reset
        state caches so reconnect doesn't use stale data.
        """
        pass

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

        Reads multiple response lines to handle both motor (single-line
        INFO) and LED (multi-line INFO where version is on "Firmware:" line).
        Uses a short per-read timeout (0.5s) to avoid wasting time when
        the board sends fewer lines than requested.

        Also detects protocol version: if INFO response starts with '{'
        it's v3.0 JSON Lines; otherwise LEGACY.

        Sets:
            firmware_version: Parsed version string (e.g. "3.0.3") or None
            firmware_date: Parsed date string (e.g. "2024-02-01") or None
            firmware_responding: True if board sent a meaningful INFO response
        """
        # Snapshot pre-INFO driver state so serial.log shows what the
        # port looked like before we asked for version info. #619 —
        # when the board falls through to "legacy, no version info"
        # we need to know whether the port was truly silent or had
        # partial/stale bytes waiting.
        pre_in_waiting = self._safe_in_waiting()
        _serial_log.info(
            f'{self._label} DETECT begin (in_waiting={pre_in_waiting}B)'
        )
        try:
            # Use a short timeout for version detection — we don't want
            # to block for the board's default timeout (could be 2-30s)
            # on each of the 6 readline() calls. 0.5s per line is enough
            # for USB CDC response delivery.
            #
            # stop_on_empty=True so we break out of the readline loop
            # as soon as an empty line arrives after non-empty content.
            # Motor INFO is single-line — without this, we waste 5 ×
            # 0.5s = 2.5s on every motor connect waiting for lines
            # that never come. LED INFO is multi-line with no empty
            # lines inside the content, so this is also safe for LED.
            resp_lines = self.exchange_command('INFO', response_numlines=6,
                                              timeout=0.5,
                                              stop_on_empty=True)
            if isinstance(resp_lines, list):
                resp = '\n'.join(resp_lines)
            else:
                resp = resp_lines or ''

            # Accumulate non-empty response bytes so _reset_firmware
            # can track whether any detection attempt ever saw output.
            # Only non-empty lines count — empty strings from readline
            # timeouts are what we're trying to detect as "silent."
            # getattr default for the __new__ test-construction path.
            prev = getattr(self, '_detect_response_bytes', 0)
            if isinstance(resp_lines, list):
                self._detect_response_bytes = prev + sum(
                    len(ln) for ln in resp_lines if ln
                )
            elif resp_lines:
                self._detect_response_bytes = prev + len(resp_lines)
            else:
                self._detect_response_bytes = prev

            # Check if we got any meaningful content (not just empty lines)
            resp_stripped = resp.strip()
            if not resp_stripped:
                self.firmware_version = None
                self.firmware_responding = False
                logger.info(f'{self._label} No response from INFO')
                # Log full diagnostic snapshot so the failure case is
                # debuggable from a user-uploaded log alone (#619).
                _serial_log.warning(
                    f'{self._label} DETECT empty-response: '
                    f'pre_in_waiting={pre_in_waiting}B '
                    f'post_in_waiting={self._safe_in_waiting()}B '
                    f'raw={resp_lines!r}'
                )
                return

            # Board is responding — mark it even if we can't parse a version
            self.firmware_responding = True

            # v3.0 STUB: JSON Lines protocol detection
            if resp_stripped.startswith('{'):
                self.protocol_version = ProtocolVersion.V3
                logger.info(f'{self._label} Detected v3.0 JSON Lines protocol')
            else:
                self.protocol_version = ProtocolVersion.LEGACY

            # Try to parse version number (v3.0+ firmware)
            match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', resp)
            if match:
                self.firmware_version = match.group(1)
                logger.info(f'{self._label} Firmware version: {self.firmware_version}')
            else:
                self.firmware_version = None

            # Try to parse firmware date (all firmware formats)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', resp)
            if date_match:
                self.firmware_date = date_match.group(1)
                logger.info(f'{self._label} Firmware date: {self.firmware_date}')

            if self.firmware_version:
                logger.info(f'{self._label} Firmware v{self.firmware_version} detected')
            else:
                logger.info(f'{self._label} Legacy firmware (no version string, date={self.firmware_date})')

        except Exception as e:
            logger.debug(f'{self._label} version detection failed: {e}')
            self.firmware_version = None
            self.firmware_responding = False
            self.protocol_version = ProtocolVersion.LEGACY

    def detect_firmware_version(self):
        """Re-detect firmware version from the connected board.

        Useful after firmware updates when the version may have changed
        without a full reconnect cycle. Updates firmware_version,
        firmware_date, and firmware_responding attributes.
        """
        self._detect_firmware_version()

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
    def exchange_command(self, command, response_numlines=1, timeout=None,
                         stop_on_empty=False):
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
            stop_on_empty: If True, break out of the read loop as
                soon as readline() returns an empty line AFTER at
                least one non-empty line has been received. Used by
                _detect_firmware_version to avoid waiting the full
                per-line timeout on every subsequent line when the
                motor board sends its INFO as a single line
                (previously wasted 2.5s on every motor connect).
                Safe because neither motor nor LED INFO responses
                contain intentional empty lines in the middle of
                their content.
        """
        with self._lock:
            # Fail fast on silent boards (#619). exchange_command()
            # is called for version detection from inside
            # _detect_firmware_version, so we explicitly allow INFO
            # through — otherwise the silent flag becomes sticky and
            # a future reconnect can never clear it. Every other
            # command on a silent board is rejected immediately so
            # the user sees failures at full speed instead of dozens
            # of 3-second timeouts (the #619 symptom where LED
            # commands kept "succeeding" with empty responses for
            # minutes after the silent connect).
            #
            # getattr default is False so tests that construct boards
            # via __new__ (bypassing __init__) don't trip on a missing
            # attribute.
            if getattr(self, 'firmware_silent', False) and command.strip().upper() != 'INFO':
                _serial_log.warning(
                    f'{self._label} {command} -> REJECTED (board silent, '
                    f'power cycle required)'
                )
                return None

            if self.driver is None:
                try:
                    logger.info(f'{self._label} Auto-reconnect triggered by {command}')
                    self.connect()
                except Exception as e:
                    _serial_log.error(f'{self._label} {command} -> RECONNECT FAILED: {e}')
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

            cmd_upper = command.strip().upper()
            stream = command.encode('utf-8') + b"\n"
            t_start = time.monotonic()
            try:
                # Flush any stale data in the input buffer before writing.
                # If a previous readline() timed out, the firmware's response
                # is still sitting in the buffer and would be misread as this
                # command's response, causing a permanent desync cascade.
                stale = self.driver.in_waiting
                if stale > 0:
                    discarded = self.driver.read(stale)
                    _serial_log.info(f'{self._label} FLUSH {stale}B: {discarded!r}')

                self.driver.write(stream)
                resp_lines = []
                saw_content = False
                for _ in range(response_numlines):
                    line = self.driver.readline().decode("utf-8", "ignore").strip()
                    # Auto-detect and drain echoes:
                    # - LED board: "RE: INFO" prefix
                    # - Motor board: raw echo of command via MicroPython input()
                    if line.startswith('RE:') or line.upper() == cmd_upper:
                        line = self.driver.readline().decode("utf-8", "ignore").strip()
                    resp_lines.append(line)
                    if line:
                        saw_content = True
                    elif stop_on_empty and saw_content:
                        # Motor INFO fix: stop reading once we've seen
                        # at least one non-empty line and the next line
                        # is empty (timeout). Saves ~2.5s per motor
                        # connect. Pad resp_lines so response_numlines
                        # callers that expect exactly N entries still
                        # get a uniform-length list.
                        while len(resp_lines) < response_numlines:
                            resp_lines.append('')
                        break

                response = resp_lines[0] if response_numlines == 1 else resp_lines

                # Drain any remaining data from multi-line response bursts.
                # Old firmware (pre-v3.0) sends multi-line INFO/STATUS even
                # when we only requested 1 line. Without this drain, leftover
                # lines pollute the next command's response.
                time.sleep(0.02)  # Brief pause for remaining lines to arrive
                remaining = self.driver.in_waiting
                if remaining > 0:
                    self.driver.read(remaining)

                elapsed_ms = (time.monotonic() - t_start) * 1000

                # Serial log: compact command → response with timing
                resp_repr = repr(response)
                if len(resp_repr) > 200:
                    resp_repr = resp_repr[:200] + '...'
                _serial_log.info(f'{self._label} {command} -> {resp_repr} ({elapsed_ms:.1f}ms)')

                resp_str = str(response)
                if 'ERROR' in resp_str or 'FAIL' in resp_str or 'exceeds safe' in resp_str:
                    _serial_log.warning(f'{self._label} FIRMWARE ERROR: {command} -> {response}')

                return response

            except serial.SerialTimeoutException:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                now = time.monotonic()
                last = getattr(self, '_last_error_log_time', 0.0)
                interval = getattr(self, '_error_log_interval', 2.0)
                if now - last >= interval:
                    _serial_log.warning(f'{self._label} {command} -> TIMEOUT ({elapsed_ms:.1f}ms)')
                    self._last_error_log_time = now

            except Exception as e:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                now = time.monotonic()
                last = getattr(self, '_last_error_log_time', 0.0)
                interval = getattr(self, '_error_log_interval', 2.0)
                if now - last >= interval:
                    _serial_log.error(f'{self._label} {command} -> EXCEPTION: {e} ({elapsed_ms:.1f}ms)')
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
                    logger.info(f'{self._label} Auto-reconnect triggered by {command}')
                    self.connect()
                except Exception as e:
                    _serial_log.error(f'{self._label} {command} -> RECONNECT FAILED: {e}')
                    return None
            if self.driver is None:
                return None

            saved_timeout = self.driver.timeout
            self.driver.timeout = min(timeout, 5.0)  # per-readline timeout

            cmd_upper = command.strip().upper()
            t_start = time.monotonic()
            try:
                # Flush stale data
                stale = self.driver.in_waiting
                if stale > 0:
                    self.driver.read(stale)
                    _serial_log.info(f'{self._label} FLUSH {stale}B')

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
                    # Skip echo lines (LED "RE:" prefix or raw motor echo)
                    if line.startswith('RE:') or line.upper() == cmd_upper:
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

                elapsed_ms = (time.monotonic() - t_start) * 1000
                result = '\n'.join(lines) or None
                _serial_log.info(f'{self._label} {command} -> {len(lines)} lines ({elapsed_ms:.1f}ms)')
                return result

            except serial.SerialTimeoutException:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                _serial_log.warning(f'{self._label} {command} -> TIMEOUT ({elapsed_ms:.1f}ms)')
                return '\n'.join(lines) if lines else None

            except Exception as e:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                _serial_log.error(f'{self._label} {command} -> EXCEPTION: {e} ({elapsed_ms:.1f}ms)')
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
    def enter_raw_repl(self, soft_reset=True):
        """Interrupt firmware and enter MicroPython raw REPL.

        While in raw REPL, normal commands (exchange_command) cannot be
        used. Call exit_raw_repl() when done to reboot the firmware.

        Args:
            soft_reset: If True (default), soft-reset after entering raw REPL
                for a clean MicroPython state. Set to False for old firmware
                with WDT (soft reset kills the Timer that feeds WDT).

        Returns True on success, False on failure.
        """
        with self._lock:
            if self.driver is None:
                self._open_serial()
            if _enter_raw_repl(self.driver, soft_reset=soft_reset):
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

    def repl_exec(self, code, timeout=10):
        """Execute arbitrary code in raw REPL (must be in raw REPL).

        Returns (stdout, stderr) as bytes tuple, or None on error.
        """
        with self._lock:
            if not self._in_raw_repl or self.driver is None:
                logger.error(f'{self._label} repl_exec: not in raw REPL')
                return None
            from drivers.raw_repl import raw_exec as _raw_exec
            return _raw_exec(self.driver, code, timeout=timeout)

    def verify_firmware_running(self, timeout=10):
        """Verify firmware is responding after raw REPL exit.

        Returns firmware response string, or None if not responding.
        """
        with self._lock:
            if self.driver is None:
                return None
            return _verify_firmware_running(self.driver, timeout=timeout)
