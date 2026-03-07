# Serial Communication Analysis

## Current State: Two Different Patterns

### LED Board (Host: `ledboard.py`, Firmware: `LED Controller/main.py`)

**Host → Firmware:**
- Send: `command.encode('utf-8') + b"\n"` (LF terminated)
- Firmware reads: `sys.stdin.readline()` then `.strip().upper()`

**Firmware → Host:**
- Echo: `print('RE:', command)` — immediate echo before processing
- Result: command-specific `print()` calls (e.g., `'LED 3 set to 100 mA.'`)
- MicroPython `print()` sends `\r\n` (CR+LF) by default

**Host reads:**
- `response = self.driver.readline()` — reads until `\n`
- `response[:-2]` — strips trailing `\r\n`
- Only reads ONE line (gets the `RE:` echo, misses the result line)

**Serial config:**
- 115200 baud, 8N1
- Timeout: 100ms read, 100ms write
- VID=0x0424, PID=0x704C

**Unnecessary overhead in `exchange_command`:**
```
flushInput()          ~0ms    Masks bugs — discards unread data
flush()               ~0ms    Redundant — output buffer should be empty
sleep(0.001)          1ms     Nothing to wait for
sleep(0.01)          10ms     Firmware responds in <1ms; readline() already blocks
logger.info()         —       Should be logger.debug()
```
**Total waste: ~11ms per command**

### Motor Board (Host: `motorboard.py`, Firmware: `Motor Controller/Firmware/main.py`)

**Host → Firmware:**
- Send: `command.encode('utf-8') + b"\n"` (LF terminated) — same as LED

**Firmware → Host:**
- No echo — processes command first, then `print(rv)` at the end (line 1174)
- Single response line for most commands
- MicroPython `print()` sends `\r\n`

**Host reads:**
- `resp_lines = [self.driver.readline() for _ in range(response_numlines)]`
- `.strip()` on each line, plus `\r` cleanup: `response[0].rsplit('\r')[-1]`
- Supports multi-line responses (`response_numlines` param)

**Serial config:**
- 115200 baud, 8N1
- Timeout: 30s read (for homing), 5s write
- VID=0x2E8A, PID=0x0005

**No unnecessary overhead** — clean write → readline pattern.

## Key Differences Summary

| Aspect | LED Board | Motor Board |
|--------|-----------|-------------|
| Echo before processing | Yes (`RE: <cmd>`) | No |
| Response line count | 2 lines (echo + result) | 1 line (result only) |
| Host reads | 1 line (only gets echo!) | 1 line (gets result) |
| Response prefix | `RE: ` | None |
| Host strips trailing | `[:-2]` (assumes CR+LF) | `.strip()` + `\r` rsplit |
| Unnecessary sleeps | 11ms total | None |
| Read timeout | 100ms | 30s |
| Write timeout | 100ms | 5s |
| `flushInput/flush` | Yes (before every cmd) | No |
| Log level | `info` (noisy) | `debug` |

## Problems

1. **LED host only reads 1 line but firmware sends 2** — The `RE:` echo is read, but the actual result line (e.g., `LED 3 set to 100 mA.`) is left in the buffer. This is why `flushInput()` exists — to discard the previous result. This works but means the host never sees the actual firmware response.

2. **LED 11ms overhead** — Every LED `exchange_command` wastes ~11ms in unnecessary sleeps. During multi-channel protocol captures, this adds up significantly.

3. **Inconsistent response parsing** — LED strips `[:-2]`, motor does `.strip()` + `\r` rsplit. Both should handle CR+LF consistently.

4. **Motor 30s timeout** — Necessary for homing, but means any serial error hangs for 30s. Could use a shorter default and temporarily increase for homing commands.

## Unified Design (Proposed)

### Protocol (backwards-compatible)

Both controllers would use the same pattern:
1. Host sends: `<COMMAND>\n`
2. Firmware responds: `<RESPONSE>\n` (single line, no echo prefix)
3. For multi-line responses, use a count parameter or terminator

**Backwards compatibility:**
- New host code should tolerate `RE: ` prefix (strip it if present)
- New host code should tolerate extra `\r` in response (strip it)
- New firmware should NOT send `RE:` echo (breaking old host code is OK if old host is updated simultaneously)
- Alternatively: new firmware sends both echo and result, new host reads 2 lines if echo detected

### Host: Shared Serial Driver Base Class

```python
class SerialBoard:
    """Base class for serial board communication."""

    def __init__(self, vid, pid, label, timeout=0.5, write_timeout=0.5):
        self._lock = threading.RLock()
        self._vid = vid
        self._pid = pid
        self._label = label
        self._timeout = timeout
        self._write_timeout = write_timeout
        self.driver = None
        self.port = None
        self.found = False
        self._find_port()

    def _find_port(self):
        for port in list_ports.comports(include_links=True):
            if port.vid == self._vid and port.pid == self._pid:
                self.port = port.device
                self.found = True
                break

    def connect(self):
        with self._lock:
            self.driver = serial.Serial(
                port=self.port, baudrate=115200,
                timeout=self._timeout, write_timeout=self._write_timeout,
            )
            # Reset firmware
            self.driver.write(b'\x04\n')
            self.driver.readline()  # discard reset response

    def exchange_command(self, command, timeout=None):
        with self._lock:
            if self.driver is None:
                self.connect()
            if self.driver is None:
                return None

            # Temporarily adjust timeout if needed (e.g., homing)
            if timeout is not None:
                self.driver.timeout = timeout

            try:
                self.driver.write(command.encode('utf-8') + b'\n')
                response = self.driver.readline()
                response = response.decode('utf-8', 'ignore').strip()

                # Handle RE: echo from older LED firmware
                if response.startswith('RE: '):
                    # Read actual response on next line
                    response = self.driver.readline()
                    response = response.decode('utf-8', 'ignore').strip()

                return response
            except serial.SerialTimeoutException:
                self._close_driver()
                return None
            finally:
                if timeout is not None:
                    self.driver.timeout = self._timeout

    def write_command(self, command):
        """Fire-and-forget write (no response read)."""
        with self._lock:
            if self.driver is None:
                self.connect()
            if self.driver is None:
                return
            try:
                self.driver.write(command.encode('utf-8') + b'\n')
            except Exception:
                self._close_driver()

    def _close_driver(self):
        try:
            if self.driver: self.driver.close()
        except Exception: pass
        self.driver = None
```

### Firmware: Unified Response Pattern

Remove the `RE:` echo from LED firmware. Both controllers would:
```python
# Parse command
command = sys.stdin.readline().strip().upper()
# Process and respond
rv = process_command(command)
print(rv)
```

### Migration Path

1. **Phase 1 (host only, no firmware change):**
   - Optimize LED `exchange_command`: remove sleeps, flushInput, flush
   - Read 2 lines when `RE:` echo detected (drain both echo and result)
   - Change `logger.info` → `logger.debug`
   - Standardize response parsing (`.strip()` instead of `[:-2]`)

2. **Phase 2 (firmware + host):**
   - Remove `RE:` echo from LED firmware
   - Create `SerialBoard` base class
   - Both `LEDBoard` and `MotorBoard` inherit from it
   - Add configurable timeout for homing commands

3. **Phase 3 (protocol improvements):**
   - Add firmware version query to negotiate protocol features
   - Add checksum/validation for critical commands
   - Consider binary protocol for high-frequency commands
