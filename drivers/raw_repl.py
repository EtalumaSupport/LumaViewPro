# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
MicroPython raw REPL file transfer (Thonny-style).

Provides functions to interrupt running firmware on an RP2040 board,
enter the MicroPython raw REPL, execute arbitrary Python code, list
and read files from the board's filesystem, then exit cleanly so the
firmware restarts.

This module is designed for FIELD-DEPLOYED instruments. Every function
prioritises reliability over speed:
  - Retries with backoff on entering raw REPL
  - Soft reset for clean state before file operations
  - SHA256 verification after every file write
  - Write-to-temp-then-rename for atomic file operations
  - Backup (.bak) before overwriting config files
  - Conservative delays throughout — these operations run rarely
  - Read verification (double-read) for config backup

Safety model follows mpremote (MicroPython's official tool) with
additional verification steps that mpremote does NOT do.

Usage::

    from drivers.raw_repl import (
        enter_raw_repl, exit_raw_repl, list_files, read_file, write_file,
    )

    if enter_raw_repl(serial_port):
        try:
            files = list_files(serial_port)
            for name in files:
                data = read_file(serial_port, name)
            write_file(serial_port, 'config.json', new_data)
        finally:
            exit_raw_repl(serial_port)

Note: Some old firmware builds (pre-2023) can enter raw REPL but
cannot access the filesystem due to SPI bus contention after
interrupting main.py. ``list_files`` returns an empty list in
that case.
"""

import base64
import hashlib
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

# ---------------------------------------------------------------------------
# Conservative timing constants — reliability over speed
# ---------------------------------------------------------------------------

# Delays between serial control character sends
DELAY_AFTER_CTRL_C = 0.2       # Wait after each Ctrl+C interrupt
DELAY_AFTER_DRAIN = 0.5        # Wait after draining input buffer
DELAY_AFTER_CTRL_A = 0.5       # Wait after entering raw REPL
DELAY_AFTER_SOFT_RESET = 3.0   # Wait for board to reboot after soft reset
DELAY_BETWEEN_CHUNKS = 0.01    # 10ms between 256-byte code chunks (mpremote default)
DELAY_AFTER_EXIT = 3.0         # Wait for firmware to boot after exit
DELAY_BETWEEN_RETRIES = 1.0    # Wait between retry attempts

# Retry counts
ENTER_RETRIES = 3              # Attempts to enter raw REPL
READ_VERIFY_RETRIES = 2        # Read file twice and compare (catch corruption)
WRITE_VERIFY_RETRIES = 3       # Attempts to write + verify a file

# Chunk size for sending code to raw REPL (mpremote default)
CODE_CHUNK_SIZE = 256


def _drain_input(serial_port):
    """Drain all pending input from serial port (mpremote pattern).

    Loops until nothing left, unlike a single read(4096) which may
    miss data still in transit.
    """
    while True:
        n = serial_port.in_waiting if hasattr(serial_port, 'in_waiting') else 0
        if n > 0:
            serial_port.read(n)
            time.sleep(0.05)
        else:
            # One final read with short timeout to catch stragglers
            old_timeout = serial_port.timeout
            serial_port.timeout = 0.1
            leftover = serial_port.read(4096)
            serial_port.timeout = old_timeout
            if not leftover:
                break


def _send_chunked(serial_port, data):
    """Send data in chunks with inter-chunk delays.

    Prevents buffer overflows on the RP2040 USB CDC endpoint.
    Uses the same 256-byte/10ms pattern as mpremote.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    for i in range(0, len(data), CODE_CHUNK_SIZE):
        chunk = data[i:i + CODE_CHUNK_SIZE]
        serial_port.write(chunk)
        if i + CODE_CHUNK_SIZE < len(data):
            time.sleep(DELAY_BETWEEN_CHUNKS)


def enter_raw_repl(serial_port, soft_reset=True):
    """Interrupt running firmware and enter MicroPython raw REPL.

    Retries up to ENTER_RETRIES times with backoff. After entering,
    optionally performs a soft reset (Ctrl+D) to ensure clean state
    with no leftover variables or open file handles.

    Args:
        serial_port: pyserial Serial instance
        soft_reset: If True (default), soft-reset after entering raw REPL
            for a clean MicroPython state. Recommended before file operations.

    Returns True on success, False on failure after all retries exhausted.
    """
    old_timeout = serial_port.timeout

    for attempt in range(1, ENTER_RETRIES + 1):
        serial_port.timeout = 1.0
        try:
            logger.info(f"Entering raw REPL (attempt {attempt}/{ENTER_RETRIES})")

            # Step 1: Interrupt any running program (Ctrl+C twice)
            serial_port.write(CTRL_C)
            time.sleep(DELAY_AFTER_CTRL_C)
            serial_port.write(CTRL_C)
            time.sleep(DELAY_AFTER_CTRL_C)

            # Step 2: Drain all pending input (mpremote pattern)
            _drain_input(serial_port)
            time.sleep(DELAY_AFTER_DRAIN)

            # Step 3: Send Ctrl+A to enter raw REPL
            serial_port.write(b'\r' + CTRL_A)  # \r first, like mpremote
            time.sleep(DELAY_AFTER_CTRL_A)
            data = serial_port.read(4096)

            if not data or b'raw REPL' not in data:
                # Retry Ctrl+A — sometimes the first one gets eaten
                serial_port.write(CTRL_A)
                time.sleep(DELAY_AFTER_CTRL_A)
                data = serial_port.read(4096)

            if not data or b'raw REPL' not in data:
                logger.warning(
                    f"Raw REPL entry attempt {attempt} failed: {data!r:.100}")
                if attempt < ENTER_RETRIES:
                    time.sleep(DELAY_BETWEEN_RETRIES * attempt)  # Increasing backoff
                continue

            logger.info("Raw REPL entered successfully")

            # Step 4: Soft reset for clean state (mpremote best practice)
            if soft_reset:
                serial_port.write(CTRL_D)  # Soft reset inside raw REPL
                time.sleep(DELAY_AFTER_SOFT_RESET)
                _drain_input(serial_port)

                # After soft reset, we need to re-enter raw REPL
                # (soft reset drops back to friendly REPL)
                serial_port.write(CTRL_A)
                time.sleep(DELAY_AFTER_CTRL_A)
                data = serial_port.read(4096)
                if not data or b'raw REPL' not in data:
                    logger.warning("Failed to re-enter raw REPL after soft reset")
                    if attempt < ENTER_RETRIES:
                        time.sleep(DELAY_BETWEEN_RETRIES * attempt)
                    continue
                logger.info("Clean raw REPL state after soft reset")

            return True

        except Exception as e:
            logger.warning(f"Raw REPL entry attempt {attempt} error: {e}")
            if attempt < ENTER_RETRIES:
                time.sleep(DELAY_BETWEEN_RETRIES * attempt)

    logger.error(f"Failed to enter raw REPL after {ENTER_RETRIES} attempts")
    serial_port.timeout = old_timeout
    return False


def raw_exec(serial_port, code, timeout=10):
    """Execute Python code in raw REPL and return stdout output.

    The raw REPL protocol:
      1. Send code in chunks (256 bytes, 10ms delay) followed by Ctrl+D
      2. Board responds: OK<stdout output>\\x04<stderr output>\\x04

    Returns stdout as bytes, or None on error.
    """
    old_timeout = serial_port.timeout
    serial_port.timeout = timeout

    try:
        # Send code in chunks + Ctrl+D to execute
        _send_chunked(serial_port, code.encode('utf-8') + CTRL_D)

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

    Sends Ctrl+C (cancel stuck command), Ctrl+B (exit raw REPL),
    then Ctrl+D (soft reset). Waits conservatively for firmware to boot.
    """
    try:
        serial_port.write(CTRL_C)  # Cancel any stuck command
        time.sleep(DELAY_AFTER_CTRL_C)
        serial_port.write(CTRL_B)  # Back to friendly REPL
        time.sleep(0.2)
        serial_port.write(CTRL_D)  # Soft reset → restarts main.py
        time.sleep(DELAY_AFTER_EXIT)  # Wait for firmware to boot
        _drain_input(serial_port)  # Drain startup output
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


def read_file(serial_port, filename, verify=True):
    """Read a single file from the board via raw REPL.

    Uses base64 encoding for safe binary transfer.
    The filename is quoted via ``repr()`` to prevent code injection.

    If verify=True (default), reads the file twice and compares to
    detect serial corruption. This doubles the transfer time but
    catches single-bit errors that base64 alone cannot detect.

    Returns file contents as bytes, or None on failure.
    """
    code = (
        "import ubinascii\n"
        f"with open({filename!r}, 'rb') as f:\n"
        " d = f.read()\n"
        "print(ubinascii.b2a_base64(d).decode(), end='')"
    )

    def _do_read():
        stdout = raw_exec(serial_port, code, timeout=10)
        if stdout is None:
            return None
        try:
            return base64.b64decode(stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to decode {filename}: {e}")
            return None

    data = _do_read()
    if data is None:
        return None

    if verify:
        time.sleep(0.2)  # Brief pause between reads
        data2 = _do_read()
        if data2 is None:
            logger.warning(f"Verification read of {filename} failed — "
                           f"using first read ({len(data)} bytes)")
            # Still return first read — better than nothing for diagnostics
        elif data != data2:
            logger.error(
                f"VERIFICATION FAILED for {filename}: "
                f"read 1 = {len(data)} bytes, read 2 = {len(data2)} bytes. "
                f"Possible serial corruption. Retrying...")
            # Third attempt as tiebreaker
            time.sleep(0.5)
            data3 = _do_read()
            if data3 == data:
                logger.info(f"Third read matches first — using first read")
            elif data3 == data2:
                logger.info(f"Third read matches second — using second read")
                data = data2
            else:
                logger.error(
                    f"All three reads differ for {filename}! "
                    f"Serial link may be unreliable. Using largest read.")
                data = max(data, data2, data3, key=len)
        else:
            logger.debug(f"Verified {filename}: {len(data)} bytes, both reads match")

    return data


def write_file(serial_port, filename, data):
    """Write a file to the board via raw REPL with full safety.

    Safety measures (beyond what mpremote and Thonny do):
      1. Backup: renames existing file to filename.bak
      2. Atomic write: writes to filename.tmp first
      3. SHA256 verification: computes hash on-device, compares to local
      4. Rename: filename.tmp → filename only after verification
      5. Retries: up to WRITE_VERIFY_RETRIES attempts

    Args:
        serial_port: pyserial Serial instance (must be in raw REPL)
        filename: destination path on board (e.g. 'motorconfig.json')
        data: bytes to write

    Returns True on success, False on failure.
    """
    # Sanitize — prevent path traversal from host side
    import pathlib
    safe_name = pathlib.Path(filename).name
    if not safe_name:
        logger.error(f"Invalid filename: {filename!r}")
        return False

    tmp_name = safe_name + '.tmp'
    bak_name = safe_name + '.bak'

    # Compute expected SHA256
    expected_hash = hashlib.sha256(data).hexdigest()
    logger.info(
        f"Writing {safe_name} ({len(data)} bytes, SHA256={expected_hash[:16]}...)")

    # Encode data as base64 for safe transfer
    b64_data = base64.b64encode(data).decode('ascii')

    for attempt in range(1, WRITE_VERIFY_RETRIES + 1):
        logger.info(f"Write attempt {attempt}/{WRITE_VERIFY_RETRIES}")

        # Step 1: Back up existing file (ignore errors if file doesn't exist)
        backup_code = (
            "import os\n"
            "try:\n"
            f" os.rename({safe_name!r}, {bak_name!r})\n"
            f" print('backup_ok')\n"
            "except OSError:\n"
            " print('no_existing_file')\n"
        )
        result = raw_exec(serial_port, backup_code, timeout=5)
        if result is not None:
            msg = result.decode('utf-8', 'ignore').strip()
            logger.info(f"Backup step: {msg}")
        time.sleep(0.3)

        # Step 2: Write data to temp file via base64
        # Split base64 into chunks to avoid memory issues on RP2040
        chunk_size = 512  # base64 chars per chunk (384 bytes decoded)
        chunks = [b64_data[i:i + chunk_size]
                  for i in range(0, len(b64_data), chunk_size)]

        # Open temp file
        open_code = (
            "import ubinascii\n"
            f"_f = open({tmp_name!r}, 'wb')\n"
            "print('opened')"
        )
        result = raw_exec(serial_port, open_code, timeout=5)
        if result is None or b'opened' not in result:
            logger.warning(f"Failed to open {tmp_name} for writing")
            time.sleep(DELAY_BETWEEN_RETRIES)
            continue
        time.sleep(0.1)

        # Write chunks
        write_ok = True
        for i, chunk in enumerate(chunks):
            write_code = (
                f"_f.write(ubinascii.a2b_base64({chunk!r}))\n"
                f"print('chunk_{i}')"
            )
            result = raw_exec(serial_port, write_code, timeout=10)
            if result is None or f'chunk_{i}'.encode() not in result:
                logger.warning(f"Failed to write chunk {i}/{len(chunks)}")
                write_ok = False
                break
            time.sleep(0.05)  # Brief pause between chunks

        # Close file
        close_code = "_f.close()\nprint('closed')"
        result = raw_exec(serial_port, close_code, timeout=5)
        if result is None or b'closed' not in result:
            logger.warning("Failed to close temp file")
            write_ok = False
        time.sleep(0.3)

        if not write_ok:
            # Clean up partial temp file
            raw_exec(serial_port,
                     f"import os\ntry:\n os.remove({tmp_name!r})\nexcept: pass",
                     timeout=5)
            time.sleep(DELAY_BETWEEN_RETRIES)
            continue

        # Step 3: SHA256 verification on-device
        # MicroPython 1.19 has uhashlib (not hashlib)
        verify_code = (
            "import uhashlib, ubinascii\n"
            f"h = uhashlib.sha256()\n"
            f"with open({tmp_name!r}, 'rb') as f:\n"
            " while True:\n"
            "  b = f.read(256)\n"
            "  if not b: break\n"
            "  h.update(b)\n"
            "print(ubinascii.hexlify(h.digest()).decode())"
        )
        result = raw_exec(serial_port, verify_code, timeout=10)
        if result is None:
            logger.warning("SHA256 verification failed — no response")
            time.sleep(DELAY_BETWEEN_RETRIES)
            continue

        device_hash = result.decode('utf-8', 'ignore').strip()
        if device_hash != expected_hash:
            logger.error(
                f"SHA256 MISMATCH on {tmp_name}: "
                f"expected {expected_hash[:16]}..., got {device_hash[:16]}...")
            # Clean up bad temp file
            raw_exec(serial_port,
                     f"import os\ntry:\n os.remove({tmp_name!r})\nexcept: pass",
                     timeout=5)
            time.sleep(DELAY_BETWEEN_RETRIES)
            continue

        logger.info(f"SHA256 verified: {device_hash[:16]}...")

        # Step 4: Atomic rename — temp → final
        rename_code = (
            "import os\n"
            f"os.rename({tmp_name!r}, {safe_name!r})\n"
            "print('renamed')"
        )
        result = raw_exec(serial_port, rename_code, timeout=5)
        if result is None or b'renamed' not in result:
            logger.error(f"Failed to rename {tmp_name} → {safe_name}")
            time.sleep(DELAY_BETWEEN_RETRIES)
            continue

        logger.info(f"Successfully wrote {safe_name} "
                    f"({len(data)} bytes, verified, backup in {bak_name})")
        return True

    logger.error(f"Failed to write {safe_name} after {WRITE_VERIFY_RETRIES} attempts")
    return False


def verify_firmware_running(serial_port, command='INFO', timeout=10):
    """Verify that firmware is running and responding after raw REPL exit.

    Sends the given command repeatedly until a valid response is received
    or timeout is exceeded. Use this after exit_raw_repl() to confirm
    the board recovered.

    Returns the firmware response string, or None if firmware is not responding.
    """
    deadline = time.time() + timeout
    old_timeout = serial_port.timeout
    serial_port.timeout = 2.0

    try:
        while time.time() < deadline:
            try:
                _drain_input(serial_port)
                serial_port.write(f'{command}\n'.encode('utf-8'))
                time.sleep(0.5)
                response = serial_port.read(4096)
                if response:
                    text = response.decode('utf-8', 'ignore')
                    # Look for signs of a real firmware response
                    if ('Etaluma' in text or 'EL-09' in text
                            or 'Firmware' in text or 'Version' in text):
                        logger.info(f"Firmware responding: {text.strip()[:80]}")
                        return text.strip()
            except Exception:
                pass
            time.sleep(1.0)

        logger.error(
            f"Firmware not responding after {timeout}s. "
            f"Board may need power cycle.")
        return None
    finally:
        serial_port.timeout = old_timeout
