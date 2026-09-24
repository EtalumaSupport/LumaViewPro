# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A serial port whose far end is the real board firmware, run in MicroPython.

`EmulatedPort` is a pyserial `SerialBase`, so the board drivers read and
write it exactly as they do a USB serial port: pyserial's own `readline`,
timeouts and exceptions. Behind it is one MicroPython process per open port,
running the firmware's compiled `.mpy` with its stdin and stdout as the wire.

What a USB CDC link does that a pipe does not, the port does itself:

- Ctrl-C interrupts the running firmware. Over a pipe the runtime would hand
  the byte to `readline()` as data, so the port sends the process SIGINT,
  which raises the firmware's own KeyboardInterrupt and, because the runtime
  runs with -i, leaves the real MicroPython REPL behind it.
- Ctrl-D at that REPL is a soft reset: the port restarts the process, which
  runs the firmware from the top as the board does. While the firmware is
  running, Ctrl-D is ordinary data, as it is on the board.
- A board that goes away raises SerialException, as a pulled cable does.

The port owns the process. Closing the port kills it, and a watcher kills it
if this interpreter dies without closing: at stdin EOF the firmware's idle
loop spins a whole core forever, and nothing inside the firmware notices.

The reader thread always drains the process into a bounded buffer. A board
whose output nobody reads blocks on a full pipe and stops answering, which
would look like a hardware fault; overflowing the buffer is a loud failure
instead.

Tests reach the simulated hardware through the port:

- `inject` / `clear` switch a hardware fault in the chip model on or off.
  They go down a pipe the process inherits, which the model reads at every
  SPI transfer, so a fault set before a command is in effect from that
  command's first transfer. Faults are hardware and outlive a soft reset:
  the port hands them to every process it starts.
- With the board's oracle on, every register write the firmware makes
  comes back framed on the process's own output, ordered against its
  replies, and `take_writes` returns them. The frames never reach the
  driver.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

from serial.serialutil import PortNotOpenError, SerialBase, SerialException, to_bytes

from drivers.sim_wire.mp import channel
from drivers.sim_wire.mp.tmc5072 import AXES, FAULTS

CTRL_C = 0x03
CTRL_D = 0x04
REPL_PROMPT = b'>>> '

# Unread bytes held for the driver before the port refuses to hold more. A
# board's longest reply is a few kilobytes; a megabyte unread means nothing
# is reading the port.
RX_LIMIT_BYTES = 1 << 20

# Register writes held for a test before the port refuses to hold more. A
# homing is a few dozen; this many unread means nothing is taking them.
WRITES_LIMIT = 100_000

# How long a Ctrl-C may take to reach the REPL before the rest of a write goes
# on anyway. The traceback and prompt arrive within milliseconds; the bound
# only stops a firmware that swallows the interrupt from hanging the write.
INTERRUPT_SETTLE_S = 1.0

# Runs the firmware with a watcher beside it. The watcher polls this
# interpreter's pid and kills the firmware (whose pid is the shell's, via
# exec) when this interpreter is gone, or exits when the firmware is. The
# fault pipe's read end, passed as $3, becomes the firmware's fd 3.
_LAUNCH = (
    '(while kill -0 "$1" 2>/dev/null && kill -0 $$ 2>/dev/null; do sleep 1; done; '
    'kill -9 $$ 2>/dev/null) </dev/null >/dev/null 2>&1 & '
    'exec "$2" -i -c "import main" 3<&"$3"'
)


@dataclass(frozen=True)
class BoardImage:
    """Everything one board process needs: the runtime, the firmware, the
    files the firmware reads at boot, and the module path that shadows the
    runtime's hardware modules."""

    runtime: str
    firmware_mpy: str
    files: dict[str, bytes]
    module_path: tuple[str, ...]
    label: str
    oracle: bool = False


@dataclass(frozen=True)
class RegisterWrite:
    """One SPI register write the firmware made: the motor's axis and the
    register's offset within that motor, or, for a register no single motor
    owns, axis None and the chip address."""

    chip: str
    axis: str | None
    reg: int
    value: int


class EmulatedPort(SerialBase):
    def __init__(self, image: BoardImage, **kwargs):
        self._image = image
        self._proc: subprocess.Popen | None = None
        self._workdir: str | None = None
        self._reader: threading.Thread | None = None
        self._rx = bytearray()
        # The last bytes the board sent, read or not: whether it sits at the
        # REPL is a fact about what it last printed, not about what is unread.
        self._tail = b''
        self._cond = threading.Condition()
        self._failure: str | None = None
        self._faults: set[tuple[str, str]] = set()
        self._faults_w: int | None = None
        # A frame the reader has started and not finished, across chunks.
        self._frame: bytearray | None = None
        self._writes: list[RegisterWrite] = []
        super().__init__(**kwargs)

    # -- process lifetime -------------------------------------------------

    def open(self) -> None:
        if self.is_open:
            raise SerialException('Port is already open.')
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        self._workdir = tempfile.mkdtemp(prefix='lvp_simwire_')
        self._spawn()
        self.is_open = True

    def _spawn(self) -> None:
        for name, data in self._image.files.items():
            with open(os.path.join(self._workdir, name), 'wb') as f:
                f.write(data)
        shutil.copyfile(self._image.firmware_mpy, os.path.join(self._workdir, 'main.mpy'))
        env = dict(os.environ, MICROPYPATH=':'.join((*self._image.module_path, '.frozen')))
        faults_r, self._faults_w = os.pipe()
        # Written before the process starts, so the firmware's first transfer
        # at boot already sees them.
        for axis, name in sorted(self._faults):
            self._send_fault(True, axis, name)
        try:
            self._proc = subprocess.Popen(
                ['sh', '-c', _LAUNCH, 'sh', str(os.getpid()), self._image.runtime, str(faults_r)],
                cwd=self._workdir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                pass_fds=(faults_r,),
            )
        except OSError as e:
            os.close(self._faults_w)
            self._faults_w = None
            raise SerialException(f'{self._image.label}: board process did not start: {e}') from e
        finally:
            os.close(faults_r)
        with self._cond:
            self._failure = None
            self._frame = None
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(self._proc,),
            name=f'EmulatedPort.reader[{self._image.label}]',
            daemon=True,
        )
        self._reader.start()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if self._faults_w is not None:
            os.close(self._faults_w)
            self._faults_w = None
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None

    def _restart(self) -> None:
        """Soft reset: the firmware runs again from the top."""
        self._kill()
        with self._cond:
            self._rx.clear()
            self._tail = b''
        self._spawn()

    def close(self) -> None:
        if self.is_open:
            self.is_open = False
            self._kill()
            if self._workdir is not None:
                shutil.rmtree(self._workdir, ignore_errors=True)
                self._workdir = None
            with self._cond:
                self._cond.notify_all()
        super().close()

    def _read_loop(self, proc: subprocess.Popen) -> None:
        fd = proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                chunk = b''
            with self._cond:
                if proc is not self._proc:
                    return
                if not chunk:
                    self._failure = (
                        f'{self._image.label}: board process exited (code {proc.poll()})'
                    )
                    self._cond.notify_all()
                    return
                chunk = self._demux(chunk)
                if self._failure is not None:
                    self._cond.notify_all()
                    proc.kill()
                    return
                if len(self._rx) + len(chunk) > RX_LIMIT_BYTES:
                    self._failure = (
                        f'{self._image.label}: {len(self._rx)} bytes unread on the port; '
                        'nothing is reading it'
                    )
                    self._cond.notify_all()
                    proc.kill()
                    return
                self._rx.extend(chunk)
                self._tail = (self._tail + chunk)[-len(REPL_PROMPT) :]
                self._cond.notify_all()

    def _demux(self, chunk: bytes) -> bytes:
        """Take the oracle's frames out of a chunk of the board's output and
        return what is left for the driver. Holds the lock; sets the failure
        on a frame the port cannot keep."""
        to_driver = bytearray()
        i = 0
        while i < len(chunk):
            if self._frame is None:
                start = chunk.find(channel.ORACLE_START, i)
                if start < 0:
                    to_driver += chunk[i:]
                    break
                to_driver += chunk[i:start]
                self._frame = bytearray()
                i = start + 1
            else:
                end = chunk.find(channel.ORACLE_END, i)
                if end < 0:
                    self._frame += chunk[i:]
                    break
                self._frame += chunk[i:end]
                i = end + 1
                self._keep_write(bytes(self._frame))
                self._frame = None
                if self._failure is not None:
                    break
        return bytes(to_driver)

    def _keep_write(self, frame: bytes) -> None:
        if not self._image.oracle:
            self._failure = (
                f'{self._image.label}: the board sent a register-write frame with the oracle off'
            )
            return
        if len(self._writes) >= WRITES_LIMIT:
            self._failure = (
                f'{self._image.label}: {len(self._writes)} register writes untaken; '
                'nothing is taking them'
            )
            return
        self._writes.append(RegisterWrite(*channel.parse_write(frame)))

    def _at_repl(self) -> bool:
        with self._cond:
            return self._tail == REPL_PROMPT

    # -- the simulated hardware -------------------------------------------

    def inject(self, axis: str, fault: str) -> None:
        """Switch a hardware fault on; in effect from the next command's first
        SPI transfer, and across soft resets until cleared."""
        self._check_fault(axis, fault)
        self._faults.add((axis, fault))
        self._send_fault(True, axis, fault)

    def clear(self, axis: str, fault: str) -> None:
        self._check_fault(axis, fault)
        self._faults.discard((axis, fault))
        self._send_fault(False, axis, fault)

    def take_writes(self) -> list[RegisterWrite]:
        """The register writes since the last call, oldest first. Every write
        the firmware made before a reply the driver has read is here."""
        if not self._image.oracle:
            raise SerialException(f'{self._image.label}: the oracle is off for this board')
        with self._cond:
            writes, self._writes = self._writes, []
        return writes

    def _check_fault(self, axis: str, fault: str) -> None:
        if axis not in AXES:
            raise ValueError(f'unknown axis {axis!r}; axes are {AXES}')
        if fault not in FAULTS:
            raise ValueError(f'unknown fault {fault!r}; faults are {FAULTS}')

    def _send_fault(self, on: bool, axis: str, fault: str) -> None:
        if self._faults_w is None:
            raise SerialException(f'{self._image.label}: board process is not running')
        os.write(self._faults_w, channel.fault_line(on, axis, fault))

    # -- pyserial surface -------------------------------------------------

    def _reconfigure_port(self) -> None:
        pass

    def _update_break_state(self) -> None:
        pass

    def _update_rts_state(self) -> None:
        pass

    def _update_dtr_state(self) -> None:
        pass

    @property
    def in_waiting(self) -> int:
        if not self.is_open:
            raise PortNotOpenError()
        with self._cond:
            if not self._rx and self._failure:
                raise SerialException(self._failure)
            return len(self._rx)

    @property
    def out_waiting(self) -> int:
        return 0

    def read(self, size: int = 1) -> bytes:
        if not self.is_open:
            raise PortNotOpenError()
        deadline = None if self._timeout is None else time.monotonic() + self._timeout
        with self._cond:
            while len(self._rx) < size and self._failure is None and self.is_open:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._cond.wait(remaining)
            if not self._rx and self._failure:
                raise SerialException(self._failure)
            data = bytes(self._rx[:size])
            del self._rx[:size]
            return data

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise PortNotOpenError()
        data = to_bytes(data)
        start = 0
        for i, byte in enumerate(data):
            if byte == CTRL_C:
                self._send(data[start:i])
                self._interrupt()
                start = i + 1
            elif byte == CTRL_D and self._at_repl():
                self._send(data[start:i])
                self._restart()
                start = i + 1
        self._send(data[start:])
        return len(data)

    def _interrupt(self) -> None:
        """Ctrl-C. On the board the interrupt lands before the next byte does,
        so the rest of the write waits for the firmware to reach the REPL: a
        Ctrl-D sent after it must meet the REPL (a soft reset), not the
        runtime's stdin (end of input, which ends the process)."""
        self._signal(signal.SIGINT)
        deadline = time.monotonic() + INTERRUPT_SETTLE_S
        with self._cond:
            while self._tail != REPL_PROMPT and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

    def _send(self, data: bytes) -> None:
        if not data:
            return
        with self._cond:
            failure = self._failure
        proc = self._proc
        if failure is not None or proc is None:
            raise SerialException(failure or f'{self._image.label}: board process is not running')
        try:
            proc.stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            raise SerialException(f'{self._image.label}: write failed: {e}') from e

    def _signal(self, sig: int) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.send_signal(sig)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        if not self.is_open:
            raise PortNotOpenError()
        with self._cond:
            self._rx.clear()

    def reset_output_buffer(self) -> None:
        if not self.is_open:
            raise PortNotOpenError()

    @property
    def cts(self) -> bool:
        return True

    @property
    def dsr(self) -> bool:
        return True

    @property
    def ri(self) -> bool:
        return False

    @property
    def cd(self) -> bool:
        return True
