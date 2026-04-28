# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
SerialBoard — base class for RP2040-based serial controllers.

Shared infrastructure for LEDBoard and MotorBoard: port discovery,
connect/disconnect, firmware version detection, serial exchange with
auto-reconnect and echo handling, and raw REPL file operations
(config backup, firmware flash, INI updates).
"""

import json
import logging
import os
import re
import time
import serial
import serial.tools.list_ports as list_ports
from enum import Enum
from lvp_logger import logger
from modules.profile_trace import TimedLock
import threading

_serial_log = logging.getLogger('LVP.serial')

# raw_repl.py is retained as a reference/backup path (see
# docs/MPREMOTE_MIGRATION_PLAN.md approval gate #6). Only
# verify_firmware_running still delegates there — it operates on the
# pyserial driver AFTER exit_raw_repl has restored it, outside the
# Transport abstraction mpremote owns. Raw-REPL file I/O now goes
# through drivers.mpremote_transport per plan §2 Phase 2.
from drivers.raw_repl import (
    verify_firmware_running as _verify_firmware_running,
)
from drivers.mpremote_transport import create_session as _create_mpremote_session

try:
    from modules import profile_trace
except ImportError:
    profile_trace = None


class ProtocolVersion(Enum):
    LEGACY = "legacy"  # All pre-v3.0 firmware, plus every currently shipping
                       # v3.0.x field unit. Treated as an equal-class permanent
                       # driver lane per primary-session posture (2026-04-21) —
                       # not a migration scaffold. Sealed or customer-opt-out
                       # field units may stay on LEGACY indefinitely.
    V3 = "v3"          # Historical stub for v3.0/v3.1 JSON Lines design. Never
                       # shipped as firmware; retained to keep the detection
                       # branch explicit and to avoid a schema change when
                       # someone digs up old v3.1 test firmware.
    V4 = "v4"          # FW4.0 JSON Lines protocol — motor controller (U51)
                       # only. Per the (A) decision the LED controller (U50)
                       # never shipped V4 — the UART-bridge 16 B FIFO budget
                       # forces short-text framing on LED. Source of truth:
                       # docs/FIRMWARE_PROTOCOL.md §2 (motor JSON).
                       # Invariants R1-R4 (single emit_line, cmd XOR event,
                       # optional id echo, one JSON line per read).
    V35 = "v35"        # LED v3.5.0 short-text protocol — LED controller
                       # (U50) only. Single-line key=val INFO, multi-line
                       # streaming with END_<CMD> terminators, no JSON.
                       # Source of truth: docs/FIRMWARE_PROTOCOL.md §3.


class SerialBoard:

    def __init__(self, vid, pid, label, timeout=0.1, write_timeout=0.1, port=None):
        # Threading audit §10.2 — TimedLock records acquire-wait + hold time to
        # lock_trace.csv when LVP_PROFILE_TRACE=1 is set (zero overhead when off).
        # The label (`[LED Class ]` / `[XYZ Class ]`) makes per-board contention
        # distinguishable in traces. Validates the 32 ms hold-time comment at
        # drivers/motorboard.py:79 across more sessions and surfaces outliers.
        _lock_label = (label or "SerialBoard").strip(" []") or "SerialBoard"
        self._lock = TimedLock(threading.RLock(), name=f"SerialBoard._lock.{_lock_label}")
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
        # Inter-byte timeout — pyserial gap-tolerance setting.
        # Without this, readline() returns whatever bytes it has when
        # `timeout` elapses from start of read, even mid-line. With it
        # set, readline waits up to `inter_byte_timeout` AFTER each
        # received byte; rapid-bytes-then-silence triggers return
        # immediately, mid-line stalls don't break frame.
        # Bench finding 2026-04-27 (SN 7162-19): 1-2% of alternating
        # LED_SET / LED_OFF iterations on v3.5 produced 'R' single-char
        # readline failures (host saw first byte of 'RE: LED_OFF',
        # then timeout fired before remainder arrived). 50 ms gap
        # tolerance absorbs the bridge's worst observed inter-byte
        # delay; combined with terminator-based read in subclasses
        # the failure mode is eliminated. Applies to all paths
        # (LED v3.0.x / LED v3.5 / motor LEGACY / motor V4) — pure
        # gap-tolerance, no protocol assumption.
        self.inter_byte_timeout = 0.05
        self._in_raw_repl = False
        # mpremote-backed raw-REPL session. Non-None only between
        # enter_raw_repl() and exit_raw_repl(). SerialTransport takes
        # exclusive ownership of the device path, so self.driver is
        # closed for the duration of the session and reopened on exit.
        self._mpremote_session = None
        self.protocol_version = ProtocolVersion.LEGACY
        # FW4.0 (V4) state — populated by _detect_firmware_version when the
        # board answers INFO with JSON that advertises protocol >= 4.0.
        # features[] is the authoritative capability signal; host code probes
        # via has_feature(name) rather than comparing firmware_version strings.
        self.features = []
        # Unsolicited event sink. Callers install a callback to receive
        # {"ok":true,"event":"...",...} lines parsed during exchange_json.
        # LEGACY/V3 paths never call on_event — events only exist under V4.
        self.on_event = None
        # Monotonic command-id counter for V4 exchange_json. Reset on
        # connect. The id is scoped per-session; firmware echoes it in the
        # response so the host can correlate despite any push events that
        # arrive in between. See FW40_COMMAND_REFERENCE §1c.
        self._v4_id_counter = 0
        # Holds JSON responses with `id` values that arrived out-of-order
        # (e.g. a late reply to a previous command interleaved with the
        # current command's request). Drained in exchange_json when the
        # caller's id matches a stashed response.
        self._v4_pending_by_id = {}
        # Populated by _run_connect_latency_bench() at end of connect().
        # {command: summary_dict} — see drivers/serial_latency.py. None
        # when the bench is skipped (env var, no firmware version, or
        # subclass opts out by returning an empty command tuple).
        self.connect_latency_summary = None
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
                write_timeout=self.write_timeout,
                inter_byte_timeout=self.inter_byte_timeout)
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
                    write_timeout=self.write_timeout,
                    inter_byte_timeout=self.inter_byte_timeout)
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
                self._run_connect_latency_bench()
            except Exception as e:
                self._close_driver()
                logger.error(f'{self._label} connect() failed: {e}')

    # ------------------------------------------------------------------
    # Connect-time latency fingerprint
    # ------------------------------------------------------------------
    # One sample per connect as a health fingerprint — enough to capture
    # "board is responsive and how long did INFO take" for diagnostics, not
    # enough to overwhelm FW4.0 LED's RX buffer under rapid-fire (20-iter
    # sweep was shipped in session 37 and pulled 2026-04-24 after SN 7162-19
    # bench surfaced the RX overflow). Explicit release-gate §2.3
    # characterization lives in `tools/firmware_tools bench` (1000-iter).
    _CONNECT_BENCH_ITERATIONS = 1
    _CONNECT_BENCH_WARMUP = 0

    def _connect_bench_callables(self):
        """Driver-method callables benched at connect. Override per subclass.

        Returns a list of `(name, callable)` tuples. Each callable is
        invoked zero-arg per iteration — typically a bound driver
        method (`self.fullinfo`, `self.get_info`) that dispatches
        v3.0.x vs FW4.0 internally. Benching at the driver-method
        layer (not raw firmware commands) is what keeps the §2.3
        comparison apples-to-apples across firmware versions.

        Base class returns [] — null/intermediate subclasses opt out
        silently. Concrete boards (MotorBoard, LEDBoard) override.
        """
        return []

    def _run_connect_latency_bench(self):
        """Fire a lightweight driver-method latency fingerprint.

        Called once at the end of a successful connect(). Produces
        release-gate §2.3 data as a byproduct of every connect, so
        FW4.0-vs-v3.0.x comparison falls out of the log. Also
        doubles as a per-unit health fingerprint — the tech support
        report reads self.connect_latency_summary instead of
        re-measuring.

        Skipped when:
          - env LVP_SKIP_CONNECT_BENCH=1 (tests, timing-sensitive tools)
          - firmware_version is None (can't trust the board)
          - subclass returns an empty callable list

        Any exception is caught so bench can never break connect.
        """
        if os.environ.get('LVP_SKIP_CONNECT_BENCH') == '1':
            return
        if self.firmware_version is None:
            return
        named = self._connect_bench_callables()
        if not named:
            return
        try:
            from drivers import serial_latency
            summary = serial_latency.measure_callable_latencies(
                named,
                iterations=self._CONNECT_BENCH_ITERATIONS,
                warmup=self._CONNECT_BENCH_WARMUP,
            )
            self.connect_latency_summary = summary
            logger.info(serial_latency.format_one_line(
                self._label, self.firmware_version, summary
            ))
        except Exception as e:
            logger.warning(
                f'{self._label} connect-latency bench failed: {e}'
            )

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
                # Plain-text INFO silent. JSON-INFO fallback exists for
                # motor FW4.0 (which speaks JSON only); LED v3.5 always
                # responds to text-INFO so the fallback is gated to motor-
                # eligible boards via _accepts_json_info_fallback().
                if self._accepts_json_info_fallback():
                    _serial_log.info(
                        f'{self._label} DETECT text-INFO silent, trying JSON fallback'
                    )
                    try:
                        json_info = self.exchange_json({'cmd': 'INFO'}, timeout=1.0)
                    except Exception as e:
                        json_info = None
                        _serial_log.info(
                            f'{self._label} DETECT JSON fallback raised: {e}'
                        )
                    if (json_info and isinstance(json_info, dict)
                            and json_info.get('ok') is True):
                        self.firmware_responding = True
                        self._apply_json_info(json_info)
                        _serial_log.info(
                            f'{self._label} DETECT JSON fallback OK: '
                            f'fw={self.firmware_version} '
                            f'protocol={self.protocol_version.value}'
                        )
                        return
                # Both paths silent — truly non-responsive.
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

            # Detection precedence:
            #  - '{' first byte → JSON Lines (motor FW4.0 V4 / stub V3).
            #  - 'INFO ' prefix with 'proto=3.5' → LED v3.5 single-line.
            #  - otherwise → LEGACY (v3.0.x text on either board).
            if resp_stripped.startswith('{'):
                self._parse_json_info(resp_stripped)
            elif (resp_stripped.startswith('INFO ')
                    and 'proto=3.5' in resp_stripped):
                self._parse_v35_info_line(resp_stripped)
            else:
                self.protocol_version = ProtocolVersion.LEGACY
                self.features = []
                self._parse_legacy_info_text(resp)

        except Exception as e:
            logger.debug(f'{self._label} version detection failed: {e}')
            self.firmware_version = None
            self.firmware_responding = False
            self.protocol_version = ProtocolVersion.LEGACY
            self.features = []

    def _parse_json_info(self, resp_stripped):
        """Parse INFO response body for V3 / V4. Robust to extra fields
        (firmware may grow features over time without us needing a schema
        bump here)."""
        try:
            # INFO might be multi-line if the board mixed legacy + JSON;
            # take the first JSON object on the first `{...}` line.
            first_line = resp_stripped.split('\n', 1)[0]
            info = json.loads(first_line)
        except Exception as e:
            # JSON-ish but not parseable. Fall back to legacy text parsing
            # so we don't misclassify a board whose output is corrupted.
            logger.warning(f'{self._label} INFO started with {{ but JSON parse failed: {e}')
            self.protocol_version = ProtocolVersion.LEGACY
            self.features = []
            self._parse_legacy_info_text(resp_stripped)
            return
        self._apply_json_info(info)

    def _apply_json_info(self, info):
        """Apply a parsed INFO dict to driver state. Shared between
        `_parse_json_info` (called from the text-INFO path when the response
        body happens to be JSON) and the JSON-INFO fallback in
        `detect_firmware_version` (for boards that don't accept legacy-text
        INFO at all — LED FW4.0 as of 2026-04-24)."""
        protocol_str = str(info.get('protocol', '')).strip()
        fw_version = info.get('fw_version') or info.get('version')
        if fw_version:
            self.firmware_version = str(fw_version)
        self.firmware_date = info.get('fw_date') or info.get('date') or None

        # Version classification: protocol field is authoritative; fw_version
        # major is the fallback. Unknown JSON → treat as V3 (stub path) so we
        # at least capture the features array if present.
        if protocol_str.startswith('4') or (fw_version and str(fw_version).startswith('4')):
            self.protocol_version = ProtocolVersion.V4
        else:
            self.protocol_version = ProtocolVersion.V3

        features = info.get('features')
        if isinstance(features, list):
            self.features = [str(f) for f in features]
        else:
            self.features = []

        logger.info(
            f'{self._label} Detected {self.protocol_version.value} protocol'
            f' (fw={self.firmware_version}, features={self.features})'
        )

    def _accepts_json_info_fallback(self):
        """Subclass override hook. Default True (motor speaks JSON on FW4.0
        and benefits from the fallback when text-INFO is silent). LEDBoard
        overrides to False — LED v3.5 always responds to text-INFO, so the
        fallback would only ever fire on a wedged LED board, where it adds
        a 1-second timeout to no useful effect."""
        return True

    def _parse_v35_info_line(self, line):
        """Parse v3.5 LED single-line INFO. Format documented in
        docs/FIRMWARE_PROTOCOL.md §3.4:

            INFO ver=3.5.0 date=2026-04-27 sub=LED proto=3.5 \\
                 serial=AABBCCDDEEFF0011 cal=1 reset=POWER_ON heap=82000 \\
                 adc=ok:0x30D5 ref=5.012 \\
                 features=led,adc_read,selftest,calibrate,...

        Tokenizer-style parse: split on whitespace, then per-token split
        on `=`. Order is canonical but the parser is not order-sensitive.
        """
        self.protocol_version = ProtocolVersion.V35
        fields = {}
        for token in line.split()[1:]:  # skip leading 'INFO'
            if '=' in token:
                k, v = token.split('=', 1)
                fields[k] = v
        self.firmware_version = fields.get('ver')
        self.firmware_date = fields.get('date')
        feats = fields.get('features', '')
        if feats:
            self.features = [f for f in feats.split(',') if f]
        else:
            self.features = []
        logger.info(
            f'{self._label} Detected v3.5 protocol '
            f'(fw={self.firmware_version}, features={self.features})'
        )

    def _parse_legacy_info_text(self, resp):
        """Text-based INFO parsing for pre-FW4.0 firmware. Equal-class path
        per primary-session posture (permanent support, not migration
        scaffold)."""
        match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', resp)
        if match:
            self.firmware_version = match.group(1)
        else:
            self.firmware_version = None
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', resp)
        if date_match:
            self.firmware_date = date_match.group(1)
            logger.info(f'{self._label} Firmware date: {self.firmware_date}')
        if self.firmware_version:
            logger.info(f'{self._label} Firmware v{self.firmware_version} detected')
        else:
            logger.info(
                f'{self._label} Legacy firmware (no version string, '
                f'date={self.firmware_date})'
            )

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
        """Build command string for legacy text protocol (v3.0.x and
        earlier). FW4.0 JSON Lines commands go through exchange_json."""
        return cmd + "\n"

    def _parse_response(self, response):
        """Parse response for legacy text protocol. FW4.0 JSON Lines
        responses are parsed inside exchange_json."""
        return response

    # ------------------------------------------------------------------
    # Serial communication
    # ------------------------------------------------------------------
    def exchange_command(self, command, response_numlines=1, timeout=None,
                         stop_on_empty=False, return_timing=False):
        if profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE:
            with profile_trace.timer(
                "serial_trace.csv",
                "ts_ms,duration_ms,board,command,response_lines",
                lambda: [self._label.strip("[] "), command.strip().replace(",", ";")[:40], response_numlines],
            ):
                return self._exchange_command_impl(
                    command, response_numlines, timeout, stop_on_empty,
                    return_timing=return_timing)
        return self._exchange_command_impl(
            command, response_numlines, timeout, stop_on_empty,
            return_timing=return_timing)

    def _exchange_command_impl(self, command, response_numlines=1, timeout=None,
                                stop_on_empty=False, return_timing=False):
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
            return_timing: If True, return ``(response, wire_seconds)``
                where ``wire_seconds`` is the elapsed time from the
                first ``driver.write`` byte to the last useful
                ``readline`` return — excludes lock-acquire wait,
                FLUSH, post-response drain. Used by reliability-soak
                to separate wire RTT from API duration. None on error
                paths (no wire activity occurred).
        """
        # Sentinel for early-return / error paths so callers using
        # return_timing=True can still tuple-unpack safely.
        _none = (None, None) if return_timing else None

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
                return _none

            if self.driver is None:
                # #539/#632: dedupe identical reconnect failures within a 2s
                # window. Pre-fix this fired a fresh full-stack log per command
                # while disconnected — measured at ~73 reconnect attempts/sec
                # during a mid-AF USB yank, dwarfing the actual signal.
                try:
                    logger.info(f'{self._label} Auto-reconnect triggered by {command}')
                    self.connect()
                except Exception as e:
                    err_class = type(e).__name__
                    now = time.monotonic()
                    last_class = getattr(self, '_last_reconnect_err_class', None)
                    last_time = getattr(self, '_last_reconnect_err_time', 0.0)
                    if last_class == err_class and (now - last_time) < 2.0:
                        # Same error class repeated within 2s — drop to debug;
                        # the first occurrence already has the full stack.
                        _serial_log.debug(f'{self._label} {command} -> RECONNECT FAILED: same {err_class}')
                    else:
                        _serial_log.error(f'{self._label} {command} -> RECONNECT FAILED: {e}')
                        self._last_reconnect_err_class = err_class
                        self._last_reconnect_err_time = now
                    return _none

            if self.driver is None:
                return _none

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

                t_write_start = time.monotonic()
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
                t_read_end = time.monotonic()
                wire_seconds = t_read_end - t_write_start

                response = resp_lines[0] if response_numlines == 1 else resp_lines

                # Drain any remaining data from multi-line response bursts.
                # Old firmware (pre-v3.0) sends multi-line INFO/STATUS even
                # when we only requested 1 line. Without this drain, leftover
                # lines pollute the next command's response.
                #
                # 20 ms is load-bearing on the LED UART-bridge path. Brief
                # 5 ms experiment 2026-04-27 produced 5 errors in 300
                # alternating LED_SET / LED_OFF iterations (1.7% error
                # rate, one LED_OFF response read as a single 'R' char —
                # host reading mid-transmission). The bridge's firmware→
                # host transmission isn't atomic per response; 5 ms
                # cushion wasn't enough for the next command's drain
                # check to wait out a still-in-flight prior response. The
                # RP2350 single-chip path retires this drain entirely via
                # read-until-terminator framing; until then, 20 ms.
                # Reliability > speed (per Eric 2026-04-27).
                time.sleep(0.020)
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

                if return_timing:
                    return response, wire_seconds
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

            return _none

    # ------------------------------------------------------------------
    # Capability probe + FW4.0 JSON-object command exchange.
    #
    # Per the 2026-04-21 primary-session posture:
    #   - INFO.features is the authoritative capability signal on V4.
    #     Host-side code probes via has_feature('stim'), NEVER compares
    #     firmware_version >= "4.1".
    #   - LEGACY dispatcher lane stays. Not a transitional fallback.
    #     A FW4.0-capable host asks v3.0.x firmware via exchange_command()
    #     and FW4.0 firmware via exchange_json() — two equal-class lanes.
    # ------------------------------------------------------------------
    def has_feature(self, name):
        """True if the connected board advertises `name` in INFO.features.
        Only meaningful for V4; returns False on LEGACY/V3."""
        return name in self.features

    def _next_v4_id(self):
        """Monotonic command id for V4. Host-scoped; firmware echoes
        verbatim in the response. Single-session counter is fine — id
        collisions across reconnects don't matter because pending queues
        are cleared on disconnect."""
        # Called under self._lock from exchange_json, so no separate lock.
        self._v4_id_counter += 1
        return self._v4_id_counter

    def exchange_json(self, payload, timeout=None):
        """Send a V4 JSON-object command and return the matching response.

        Args:
            payload: dict — must contain at least {"cmd": "..."}. If `id`
                is not present, one is auto-assigned and the caller's copy
                of the dict is NOT modified (we serialize a shallow merge).
                Other fields (ch, mA, axis, etc.) are passed through.
            timeout: per-call read-timeout override in seconds.

        Returns:
            The response dict on success (always has "ok", usually "cmd"
            echoes the request). None on timeout, disconnect, or JSON
            parse error.

        Event handling: unsolicited `{"event":"..."}` lines encountered
        while waiting for the response are dispatched to self.on_event
        (if set) and the read loop continues. This is the FW40 §6a
        push-event consumer. Response and event lines are disambiguated
        by R2 — exactly one of `cmd` or `event` per line.

        Id demux: if the response id matches the caller's id we return it.
        If the response has a different id (a late reply to a prior
        command), it's stashed in _v4_pending_by_id; a later call whose id
        matches drains it before issuing a new write.

        LEGACY/V3 boards: this method returns None without writing.
        Callers must gate via has_feature() or check protocol_version.
        """
        if self.protocol_version != ProtocolVersion.V4:
            _serial_log.warning(
                f'{self._label} exchange_json called on '
                f'{self.protocol_version.value} board — refusing'
            )
            return None
        if not isinstance(payload, dict) or 'cmd' not in payload:
            raise ValueError('exchange_json payload must be a dict with "cmd"')

        with self._lock:
            if getattr(self, 'firmware_silent', False):
                _serial_log.warning(
                    f'{self._label} exchange_json {payload.get("cmd")} '
                    f'-> REJECTED (board silent, power cycle required)'
                )
                return None

            if self.driver is None:
                try:
                    logger.info(
                        f'{self._label} Auto-reconnect triggered by '
                        f'exchange_json {payload.get("cmd")}'
                    )
                    self.connect()
                except Exception as e:
                    _serial_log.error(
                        f'{self._label} exchange_json RECONNECT FAILED: {e}'
                    )
                    return None
            if self.driver is None:
                return None

            # Assign id if caller didn't.
            out = dict(payload)
            caller_id = out.get('id')
            if caller_id is None:
                caller_id = self._next_v4_id()
                out['id'] = caller_id

            # Did a previous call stash our response out-of-order? If so,
            # drain and return without a round-trip.
            stashed = self._v4_pending_by_id.pop(caller_id, None)
            if stashed is not None:
                return stashed

            # Per-call timeout override.
            saved_timeout = None
            if timeout is not None:
                saved_timeout = self.driver.timeout
                self.driver.timeout = timeout

            t_start = time.monotonic()
            cmd_name = out.get('cmd')
            try:
                # Flush stale input before write. Any leftover lines are
                # either stale responses (drop) or events (dispatch to
                # on_event if JSON-valid) — the flush path calls the same
                # event-aware parser as the read loop below.
                stale = self.driver.in_waiting
                if stale > 0:
                    discarded = self.driver.read(stale)
                    _serial_log.info(
                        f'{self._label} FLUSH {stale}B before json cmd: '
                        f'{discarded!r}'
                    )

                stream = (json.dumps(out) + '\n').encode('utf-8')
                self.driver.write(stream)

                # Read lines until we get a response with our id, dispatching
                # events in between. Overall deadline = 2x line timeout by
                # default so a chatty event stream doesn't starve the caller.
                # Callers that want a hard deadline pass `timeout=`.
                deadline = time.monotonic() + max(2.0, (timeout or self.driver.timeout) * 20)
                while time.monotonic() < deadline:
                    raw = self.driver.readline().decode('utf-8', 'ignore').strip()
                    if not raw:
                        continue
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        # Non-JSON on a V4 board is an FW bug or line
                        # corruption — log and keep reading.
                        _serial_log.warning(
                            f'{self._label} non-JSON on V4 wire: {raw!r}'
                        )
                        continue
                    if not isinstance(msg, dict):
                        continue

                    if 'event' in msg:
                        # R2: events never carry `cmd`; dispatch + continue.
                        if callable(self.on_event):
                            try:
                                self.on_event(msg)
                            except Exception as e:
                                _serial_log.error(
                                    f'{self._label} on_event handler raised: {e}'
                                )
                        continue

                    msg_id = msg.get('id')
                    if msg_id == caller_id:
                        elapsed_ms = (time.monotonic() - t_start) * 1000
                        _serial_log.info(
                            f'{self._label} {cmd_name} id={caller_id} '
                            f'-> ok={msg.get("ok")} ({elapsed_ms:.1f}ms)'
                        )
                        if msg.get('ok') is False:
                            _serial_log.warning(
                                f'{self._label} FIRMWARE ERR: {cmd_name} '
                                f'-> {msg.get("err")} / {msg.get("msg")}'
                            )
                        return msg
                    # Different id — stash for a future matching caller.
                    if msg_id is not None:
                        self._v4_pending_by_id[msg_id] = msg
                        continue
                    # No id on the response — treat as single-inflight match.
                    # Firmware should always echo id when request had id, so
                    # this branch is unexpected but safe.
                    _serial_log.warning(
                        f'{self._label} V4 response missing id: {raw!r}'
                    )
                    return msg

                # Timed out.
                elapsed_ms = (time.monotonic() - t_start) * 1000
                _serial_log.warning(
                    f'{self._label} exchange_json {cmd_name} id={caller_id} '
                    f'-> TIMEOUT ({elapsed_ms:.1f}ms)'
                )
                return None

            except serial.SerialTimeoutException:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                _serial_log.warning(
                    f'{self._label} exchange_json {cmd_name} '
                    f'-> WRITE TIMEOUT ({elapsed_ms:.1f}ms)'
                )
                return None
            except Exception as e:
                _serial_log.error(
                    f'{self._label} exchange_json {cmd_name} -> EXCEPTION: {e}'
                )
                self._close_driver()
                return None
            finally:
                if saved_timeout is not None and self.driver is not None:
                    self.driver.timeout = saved_timeout

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
        """Write-only fast path: send command without reading a response.

        Emits a FAST entry to serial.log with lock-wait, write, and total
        timing — the exchange_command path logs end-to-end duration, but the
        fast path is exactly what stim uses and without this line every stim
        write is invisible in serial.log. Also reports in_waiting at entry so
        response backlog from prior fast-path writes shows up in the trace.
        """
        t_enter = time.monotonic()
        with self._lock:
            t_lock = time.monotonic()
            if self.driver is None:
                try:
                    self.connect()
                except Exception:
                    return

            if self.driver is None:
                return

            try:
                in_waiting = self.driver.in_waiting
            except Exception:
                in_waiting = -1

            stream = command.encode('utf-8') + b"\n"
            t_before_write = time.monotonic()
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
                return

            t_after_write = time.monotonic()
            lock_wait_ms = (t_lock - t_enter) * 1000
            write_ms = (t_after_write - t_before_write) * 1000
            total_ms = (t_after_write - t_enter) * 1000
            _serial_log.info(
                f'{self._label} FAST {command}: {len(stream)}B '
                f'in_waiting={in_waiting}B lock_wait={lock_wait_ms:.2f}ms '
                f'write={write_ms:.2f}ms total={total_ms:.2f}ms'
            )

    # ------------------------------------------------------------------
    # Raw REPL — file operations on board filesystem
    # ------------------------------------------------------------------
    def enter_raw_repl(self, soft_reset=True):
        """Interrupt firmware and enter MicroPython raw REPL.

        While in raw REPL, normal commands (exchange_command) cannot be
        used. Call exit_raw_repl() when done to reboot the firmware.

        Under the hood: closes the pyserial driver (mpremote's
        SerialTransport takes exclusive ownership of the device path),
        constructs an mpremote-backed session, and enters raw REPL.
        exit_raw_repl reverses the sequence.

        Args:
            soft_reset: If True (default), soft-reset after entering raw REPL
                for a clean MicroPython state. Set to False for old firmware
                with WDT (soft reset kills the Timer that feeds WDT).

        Returns True on success, False on failure.
        """
        with self._lock:
            # Make sure we know the device path — _open_serial runs
            # port discovery if self.port is None.
            if self.driver is None:
                self._open_serial()
            device_path = self.port
            # Release the pyserial port so mpremote can take exclusive
            # ownership. self.driver is restored in exit_raw_repl.
            self._close_driver()

            try:
                session = _create_mpremote_session(
                    device_path, baudrate=self.baudrate
                )
                session.enter(soft_reset=soft_reset)
            except Exception as e:
                logger.error(
                    f'{self._label} enter_raw_repl failed: {e}'
                )
                # Restore application-mode driver so the board stays
                # usable for exchange_command() callers.
                try:
                    self._open_serial()
                except Exception as e2:
                    logger.error(
                        f'{self._label} enter_raw_repl recovery '
                        f'_open_serial failed: {e2}'
                    )
                return False

            self._mpremote_session = session
            self._in_raw_repl = True
            logger.info(f'{self._label} Entered raw REPL')
            return True

    def exit_raw_repl(self):
        """Exit raw REPL and reboot firmware.

        After exit, the board reboots and firmware resumes. The serial
        connection is reopened — call exchange_command() normally after.
        """
        with self._lock:
            session = self._mpremote_session
            self._mpremote_session = None
            self._in_raw_repl = False

            if session is None:
                return

            try:
                session.exit()
            except Exception as e:
                logger.warning(
                    f'{self._label} exit_raw_repl session.exit: {e}'
                )

            # Reopen the application-mode pyserial driver.
            try:
                self._open_serial()
            except Exception as e:
                logger.error(
                    f'{self._label} exit_raw_repl _open_serial: {e}'
                )

            logger.info(
                f'{self._label} Exited raw REPL, firmware rebooting'
            )

    def repl_list_files(self):
        """List files on board filesystem (must be in raw REPL).

        Returns list of filenames, or empty list on failure.
        """
        with self._lock:
            if not self._in_raw_repl or self._mpremote_session is None:
                logger.error(f'{self._label} repl_list_files: not in raw REPL')
                return []
            return self._mpremote_session.list_files()

    def repl_read_file(self, filename, verify=True):
        """Read a file from the board (must be in raw REPL).

        Returns file contents as bytes, or None on failure.
        """
        with self._lock:
            if not self._in_raw_repl or self._mpremote_session is None:
                logger.error(f'{self._label} repl_read_file: not in raw REPL')
                return None
            return self._mpremote_session.read_file(filename, verify=verify)

    def repl_write_file(self, filename, data):
        """Write a file to the board with SHA256 verification (must be in raw REPL).

        Atomic write with backup: writes to .tmp, verifies SHA256,
        backs up existing file to .bak, then renames.

        Returns True on success, False on failure.
        """
        with self._lock:
            if not self._in_raw_repl or self._mpremote_session is None:
                logger.error(f'{self._label} repl_write_file: not in raw REPL')
                return False
            return self._mpremote_session.write_file(filename, data)

    def repl_exec(self, code, timeout=10):
        """Execute arbitrary code in raw REPL (must be in raw REPL).

        Returns (stdout, stderr) as bytes tuple, or None on error.
        """
        with self._lock:
            if not self._in_raw_repl or self._mpremote_session is None:
                logger.error(f'{self._label} repl_exec: not in raw REPL')
                return None
            return self._mpremote_session.raw_exec(code, timeout=timeout)

    def verify_firmware_running(self, timeout=10):
        """Verify firmware is responding after raw REPL exit.

        Returns firmware response string, or None if not responding.
        """
        with self._lock:
            if self.driver is None:
                return None
            return _verify_firmware_running(self.driver, timeout=timeout)
