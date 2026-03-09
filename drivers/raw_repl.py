# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
MicroPython raw REPL file transfer (Thonny-style).

Provides functions to interrupt running firmware on an RP2040 board,
enter the MicroPython raw REPL, execute arbitrary Python code, list
and read files from the board's filesystem, then exit cleanly so the
firmware restarts.

This is the same mechanism Thonny uses for file management. It works
across all MicroPython firmware versions (v1.19+) without requiring
any custom firmware commands.

Usage::

    from drivers.raw_repl import enter_raw_repl, exit_raw_repl, list_files, read_file

    if enter_raw_repl(serial_port):
        try:
            files = list_files(serial_port)
            for name in files:
                data = read_file(serial_port, name)
        finally:
            exit_raw_repl(serial_port)

Note: Some old firmware builds (pre-2023) can enter raw REPL but
cannot access the filesystem due to SPI bus contention after
interrupting main.py. ``list_files`` returns an empty list in
that case.
"""

import base64
import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MicroPython raw REPL control characters
# ---------------------------------------------------------------------------

CTRL_A = b'\x01'  # Enter raw REPL
CTRL_B = b'\x02'  # Exit raw REPL (back to friendly REPL)
CTRL_C = b'\x03'  # Keyboard interrupt
CTRL_D = b'\x04'  # Soft reset / execute in raw REPL


def enter_raw_repl(serial_port, timeout=5):
    """Interrupt running firmware and enter MicroPython raw REPL.

    Sends Ctrl+C twice (interrupt), then Ctrl+A (raw REPL mode).
    Returns True on success, False on failure.
    """
    old_timeout = serial_port.timeout
    serial_port.timeout = 0.5

    try:
        # Interrupt any running program
        serial_port.write(CTRL_C)
        time.sleep(0.1)
        serial_port.write(CTRL_C)
        time.sleep(0.3)
        serial_port.read(4096)  # Drain buffer

        # Enter raw REPL
        serial_port.write(CTRL_A)
        time.sleep(0.2)
        data = serial_port.read(4096)
        if data and b'raw REPL' in data:
            return True

        # Sometimes needs another try
        serial_port.write(CTRL_A)
        time.sleep(0.3)
        data = serial_port.read(4096)
        return data is not None and b'raw REPL' in data
    except Exception as e:
        logger.warning(f"Failed to enter raw REPL: {e}")
        return False
    finally:
        serial_port.timeout = old_timeout


def raw_exec(serial_port, code, timeout=10):
    """Execute Python code in raw REPL and return stdout output.

    The raw REPL protocol:
      1. Send code followed by Ctrl+D
      2. Board responds: OK<stdout output>\\x04<stderr output>\\x04

    Returns stdout as bytes, or None on error.
    """
    old_timeout = serial_port.timeout
    serial_port.timeout = timeout

    try:
        # Send code + Ctrl+D to execute
        serial_port.write(code.encode('utf-8') + CTRL_D)

        # Read until we get the two \x04 markers
        response = b''
        start = time.time()
        while time.time() - start < timeout:
            chunk = serial_port.read(1024)
            if chunk:
                response += chunk
                # Raw REPL output ends with \x04 (twice: stdout\x04stderr\x04)
                if response.count(b'\x04') >= 2:
                    break
            else:
                break

        # Parse: OK<stdout>\x04<stderr>\x04
        ok_idx = response.find(b'OK')
        if ok_idx == -1:
            logger.warning(f"Raw REPL: no OK in response ({response[:100]!r})")
            return None

        after_ok = response[ok_idx + 2:]
        parts = after_ok.split(b'\x04')
        stdout = parts[0] if len(parts) >= 1 else b''
        stderr = parts[1] if len(parts) >= 2 else b''

        if stderr.strip():
            logger.warning(
                f"Raw REPL stderr: {stderr.decode('utf-8', 'ignore').strip()}")

        return stdout
    except Exception as e:
        logger.warning(f"Raw REPL exec error: {e}")
        return None
    finally:
        serial_port.timeout = old_timeout


def exit_raw_repl(serial_port):
    """Exit raw REPL and soft-reset the board (restarts main.py).

    Ctrl+C cancels any running command, Ctrl+B exits raw REPL,
    then Ctrl+D triggers soft reset.
    """
    try:
        serial_port.write(CTRL_C)  # Cancel any stuck command
        time.sleep(0.1)
        serial_port.write(CTRL_B)  # Back to friendly REPL
        time.sleep(0.1)
        serial_port.write(CTRL_D)  # Soft reset → restarts main.py
        time.sleep(2.0)  # Wait for firmware to boot (old boards need more time)
        serial_port.read(4096)  # Drain startup output
    except Exception as e:
        logger.warning(f"Error exiting raw REPL: {e}")


def list_files(serial_port):
    """List all files on the board's filesystem via raw REPL.

    Some old firmware builds can enter raw REPL but hang on filesystem
    imports (flash SPI contention after interrupt). Uses a short timeout
    to detect this quickly.

    Returns list of filenames, or empty list on failure.
    """
    code = "import os\nfor f in os.listdir('/'):\n print(f)"
    stdout = raw_exec(serial_port, code, timeout=5)
    if stdout is None:
        logger.info(
            "os.listdir failed — filesystem may be unavailable on this board")
        return []
    return [line.strip() for line in stdout.decode('utf-8', 'ignore').split('\n')
            if line.strip()]


def read_file(serial_port, filename):
    """Read a single file from the board via raw REPL.

    Uses base64 encoding for safe binary transfer.
    The filename is quoted via ``repr()`` to prevent code injection.

    Returns file contents as bytes, or None on failure.
    """
    code = (
        "import ubinascii\n"
        f"with open({filename!r}, 'rb') as f:\n"
        " d = f.read()\n"
        "print(ubinascii.b2a_base64(d).decode(), end='')"
    )
    stdout = raw_exec(serial_port, code, timeout=10)
    if stdout is None:
        return None
    try:
        return base64.b64decode(stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to decode {filename}: {e}")
        return None
