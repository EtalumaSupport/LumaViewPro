# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated board running the real firmware in MicroPython, and the
serial port a driver opens to it.

`EmulatedBoard` is the board: one MicroPython process running the
firmware's compiled `.mpy` with its stdin and stdout as the wire, the
simulated hardware behind it, and the USB link in front of it. It outlives
any one connection, as the EL-0940's motor controller does: it is an
internal USB device behind the mainboard's own hub and powered by the
board, so closing the port or pulling the host cable leaves its firmware
running with everything it knows, homing included.

`EmulatedPort` is one connection to a board. It is a pyserial `SerialBase`,
so the drivers read and write it exactly as they do a USB serial port:
pyserial's own `readline`, timeouts and exceptions.

What a USB CDC link does that a pipe does not, the board does itself:

- Ctrl-C interrupts the running firmware. Over a pipe the runtime would hand
  the byte to `readline()` as data, so the board sends the process SIGINT,
  which raises the firmware's own KeyboardInterrupt and, because the runtime
  runs with -i, leaves the real MicroPython REPL behind it.
- Ctrl-D at that REPL is a soft reset: the board restarts the process, which
  runs the firmware from the top as the board does, and the connection
  stays up. While the firmware is running, Ctrl-D is ordinary data, as it
  is on the board.
- A board that goes away raises SerialException on the port, as a pulled
  cable does.

A board ends its process when nothing holds it any more, and a watcher ends
it if this interpreter dies: at stdin EOF the firmware's idle loop spins a
whole core forever, and nothing inside the firmware notices.

A reader thread always drains the process. A board whose output nobody
reads blocks on a full pipe and stops answering, which would look like a
hardware fault; so output with no port attached is dropped, as the host
side of a USB link drops it, and a port that holds too much unread fails
loudly.

Tests reach the simulated hardware through the board:

- `inject` / `clear` switch a hardware fault in the chip model on or off.
  They go down a pipe the process inherits, which the model reads at every
  SPI transfer, so a fault set before a command is in effect from that
  command's first transfer. Faults are hardware and outlive a soft reset
  and a reboot: the board hands them to every process it starts.
- With the oracle on, every register write the firmware makes comes back
  framed on the process's own output, ordered against its replies, and
  `take_writes` returns them. The frames never reach the driver.
- `unplug` / `replug` pull and restore the host cable; `reboot` restarts
  the firmware, which drops the connection as USB re-enumerates.
- `drop_next_reply`, `delay_next_reply` and `garble_next_reply` spoil the
  first line the board sends after the port's next write.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import weakref
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

# Register writes held for a test before the board refuses to hold more. A
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


def _garble(line: bytes) -> bytes:
    """A reply spoiled on the wire, still one line: every printable byte is
    rotated to another printable byte, so the line keeps its length and its
    line ending and no longer says what the firmware said."""
    return bytes(33 + (b - 33 + 47) % 94 if 33 <= b <= 126 else b for b in line)


class _Process:
    """The firmware process and its pipes, with no reference back to the
    board, so a board nobody holds can be collected and its process ended."""

    def __init__(self, image: BoardImage, workdir: str, faults: set[tuple[str, str]]):
        for name, data in image.files.items():
            with open(os.path.join(workdir, name), 'wb') as f:
                f.write(data)
        shutil.copyfile(image.firmware_mpy, os.path.join(workdir, 'main.mpy'))
        env = dict(os.environ, MICROPYPATH=':'.join((*image.module_path, '.frozen')))
        faults_r, self.faults_w = os.pipe()
        try:
            # Written before the process starts, so the firmware's first
            # transfer at boot already sees them.
            for axis, name in sorted(faults):
                os.write(self.faults_w, channel.fault_line(True, axis, name))
            self.proc = subprocess.Popen(
                ['sh', '-c', _LAUNCH, 'sh', str(os.getpid()), image.runtime, str(faults_r)],
                cwd=workdir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                pass_fds=(faults_r,),
            )
        except OSError as e:
            os.close(self.faults_w)
            raise SerialException(f'{image.label}: board process did not start: {e}') from e
        finally:
            os.close(faults_r)

    def end(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for close in (
            lambda: os.close(self.faults_w),
            self.proc.stdin.close,
            self.proc.stdout.close,
        ):
            try:
                close()
            except OSError:
                pass


class _Life:
    """What must end when the board does: its process and its working
    directory. Held by the board's finalizer, which cannot hold the board."""

    def __init__(self):
        self.process: _Process | None = None
        self.workdir: str | None = None

    def end(self) -> None:
        if self.process is not None:
            self.process.end()
            self.process = None
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None


def _read_loop(board_ref: 'weakref.ref[EmulatedBoard]', process: _Process) -> None:
    fd = process.proc.stdout.fileno()
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            chunk = b''
        board = board_ref()
        if board is None or not board._from_process(process, chunk):
            return
        del board


class EmulatedBoard:
    """One simulated board: its firmware process, the simulated hardware
    behind it, and its USB link. Powered on by the first connection."""

    def __init__(self, image: BoardImage):
        self._image = image
        self._cond = threading.Condition()
        self._life = _Life()
        weakref.finalize(self, self._life.end)
        # The last bytes the board sent: whether it sits at the REPL is a fact
        # about what it last printed.
        self._tail = b''
        self._port: EmulatedPort | None = None
        self._plugged = True
        self._faults: set[tuple[str, str]] = set()
        # A frame the reader has started and not finished, across chunks.
        self._frame: bytearray | None = None
        self._writes: list[RegisterWrite] = []
        self._failure: str | None = None
        # A reply fault: armed by a test, active from the port's next write
        # until the first line after it is complete.
        self._reply_fault: tuple[str, float] | None = None
        self._reply_active = False
        self._reply = bytearray()
        # Output held back by a late reply until the delay has passed.
        self._held = bytearray()
        self._holding = False

    @property
    def label(self) -> str:
        return self._image.label

    @property
    def plugged(self) -> bool:
        with self._cond:
            return self._plugged

    # -- process lifetime (every caller holds the board's lock) -------------

    def _start(self) -> None:
        if self._life.workdir is None:
            self._life.workdir = tempfile.mkdtemp(prefix='lvp_simwire_')
        process = _Process(self._image, self._life.workdir, self._faults)
        self._life.process = process
        self._tail = b''
        self._frame = None
        self._failure = None
        # Not joined when the process is replaced: a reader whose process is
        # no longer the board's stops at its next read, and the board's lock
        # is never let go mid-change for it (a reconnect in that gap would
        # start a second process).
        threading.Thread(
            target=_read_loop,
            args=(weakref.ref(self), process),
            name=f'EmulatedBoard.reader[{self._image.label}]',
            daemon=True,
        ).start()

    def _stop(self) -> None:
        process, self._life.process = self._life.process, None
        if process is not None:
            process.end()

    def _running(self) -> bool:
        process = self._life.process
        return process is not None and process.proc.poll() is None

    def _drop_port(self, reason: str) -> None:
        port, self._port = self._port, None
        if port is not None:
            port._fail(reason)

    # -- the connection ----------------------------------------------------

    def _attach(self, port: 'EmulatedPort') -> None:
        with self._cond:
            if not self._plugged:
                raise SerialException(f'{self.label}: no board: the cable is unplugged')
            if self._port is not None:
                raise SerialException(f'{self.label}: the port is already open elsewhere')
            if not self._running():
                self._stop()
                self._start()
            self._port = port

    def _detach(self, port: 'EmulatedPort') -> None:
        with self._cond:
            if self._port is port:
                self._port = None

    def _write(self, port: 'EmulatedPort', data: bytes) -> None:
        with self._cond:
            if self._port is not port:
                raise SerialException(f'{self.label}: the connection to the board is gone')
            if self._reply_fault is not None and data:
                self._reply_active = True
            start = 0
            for i, byte in enumerate(data):
                if byte == CTRL_C:
                    self._send(data[start:i])
                    self._interrupt()
                    start = i + 1
                elif byte == CTRL_D and self._tail == REPL_PROMPT:
                    self._send(data[start:i])
                    self._soft_reset()
                    start = i + 1
            self._send(data[start:])

    def _send(self, data: bytes) -> None:
        if not data:
            return
        process = self._life.process
        if self._failure is not None or process is None:
            raise SerialException(self._failure or f'{self.label}: board process is not running')
        try:
            process.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            raise SerialException(f'{self.label}: write failed: {e}') from e

    def _interrupt(self) -> None:
        """Ctrl-C. On the board the interrupt lands before the next byte does,
        so the rest of the write waits for the firmware to reach the REPL: a
        Ctrl-D sent after it must meet the REPL (a soft reset), not the
        runtime's stdin (end of input, which ends the process)."""
        process = self._life.process
        if process is not None and process.proc.poll() is None:
            process.proc.send_signal(signal.SIGINT)
        deadline = time.monotonic() + INTERRUPT_SETTLE_S
        while self._tail != REPL_PROMPT and self._failure is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cond.wait(remaining)

    def _soft_reset(self) -> None:
        """Ctrl-D at the REPL: the firmware runs again from the top. The
        connection stays up, and what the port already holds stays there, as
        it does in the host's serial buffer."""
        self._stop()
        self._start()

    # -- the board's output --------------------------------------------------

    def _from_process(self, process: _Process, chunk: bytes) -> bool:
        """One read from the process; False once the reader should stop."""
        with self._cond:
            if process is not self._life.process:
                return False
            if not chunk:
                self._failure = f'{self.label}: board process exited (code {process.proc.poll()})'
                self._drop_port(self._failure)
                self._cond.notify_all()
                return False
            chunk = self._demux(chunk)
            if self._failure is not None:
                self._drop_port(self._failure)
                self._cond.notify_all()
                process.proc.kill()
                return False
            if chunk:
                self._tail = (self._tail + chunk)[-len(REPL_PROMPT) :]
                self._to_port(chunk)
            self._cond.notify_all()
            return True

    def _demux(self, chunk: bytes) -> bytes:
        """Take the oracle's frames out of a chunk of the board's output and
        return what is left for the driver. Holds the lock; sets the failure
        on a frame the board cannot keep."""
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
                f'{self.label}: the board sent a register-write frame with the oracle off'
            )
            return
        if len(self._writes) >= WRITES_LIMIT:
            self._failure = (
                f'{self.label}: {len(self._writes)} register writes untaken; nothing is taking them'
            )
            return
        self._writes.append(RegisterWrite(*channel.parse_write(frame)))

    def _to_port(self, data: bytes) -> None:
        """Deliver output to the attached port through any armed reply fault.
        With no port attached, or the cable pulled, output is dropped."""
        if self._reply_active:
            end = data.find(b'\n')
            if end < 0:
                self._reply += data
                return
            reply, data = bytes(self._reply + data[: end + 1]), data[end + 1 :]
            self._reply.clear()
            self._reply_active = False
            kind, seconds = self._reply_fault
            self._reply_fault = None
            if kind == 'garble':
                data = _garble(reply) + data
            elif kind == 'delay':
                data = reply + data
                self._holding = True
                threading.Timer(seconds, self._release).start()
        if self._holding:
            self._held += data
            return
        if data and self._port is not None:
            self._port._deliver(data)

    def _release(self) -> None:
        with self._cond:
            self._holding = False
            held, self._held = bytes(self._held), bytearray()
            if held and self._port is not None:
                self._port._deliver(held)

    # -- the simulated hardware --------------------------------------------

    def inject(self, axis: str, fault: str) -> None:
        """Switch a hardware fault on; in effect from the next command's first
        SPI transfer, and across soft resets and reboots until cleared."""
        self._set_fault(axis, fault, True)

    def clear(self, axis: str, fault: str) -> None:
        self._set_fault(axis, fault, False)

    def _set_fault(self, axis: str, fault: str, on: bool) -> None:
        if axis not in AXES:
            raise ValueError(f'unknown axis {axis!r}; axes are {AXES}')
        if fault not in FAULTS:
            raise ValueError(f'unknown fault {fault!r}; faults are {FAULTS}')
        with self._cond:
            if on:
                self._faults.add((axis, fault))
            else:
                self._faults.discard((axis, fault))
            process = self._life.process
            if process is not None:
                os.write(process.faults_w, channel.fault_line(on, axis, fault))

    def take_writes(self) -> list[RegisterWrite]:
        """The register writes since the last call, oldest first. Every write
        the firmware made before a reply the driver has read is here."""
        if not self._image.oracle:
            raise SerialException(f'{self.label}: the oracle is off for this board')
        with self._cond:
            writes, self._writes = self._writes, []
        return writes

    # -- the USB link --------------------------------------------------------

    def unplug(self) -> None:
        """Pull the host cable: the open port raises from now on and
        discovery finds no board. The firmware runs on, powered by the board."""
        with self._cond:
            self._plugged = False
            self._drop_port(f'{self.label}: the board was unplugged')

    def replug(self) -> None:
        with self._cond:
            self._plugged = True

    def reboot(self) -> None:
        """The firmware restarts and loses everything it knew. The open port
        raises, as USB re-enumerates, and the board is found again at once."""
        with self._cond:
            self._drop_port(f'{self.label}: the board rebooted; USB re-enumerated')
            self._stop()
            self._start()

    def drop_next_reply(self) -> None:
        self._arm_reply('drop')

    def garble_next_reply(self) -> None:
        self._arm_reply('garble')

    def delay_next_reply(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError(f'a reply is delayed by a positive time, not {seconds!r}')
        self._arm_reply('delay', seconds)

    def _arm_reply(self, kind: str, seconds: float = 0.0) -> None:
        with self._cond:
            if self._reply_fault is not None:
                raise ValueError(
                    f'{self.label}: a {self._reply_fault[0]} reply fault is already armed'
                )
            self._reply_fault = (kind, seconds)


class EmulatedPort(SerialBase):
    """One connection to an `EmulatedBoard`."""

    def __init__(self, board: EmulatedBoard, **kwargs):
        self._board = board
        self._rx = bytearray()
        self._cond = threading.Condition()
        self._failure: str | None = None
        super().__init__(**kwargs)

    def open(self) -> None:
        if self.is_open:
            raise SerialException('Port is already open.')
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        self._board._attach(self)
        self.is_open = True

    def close(self) -> None:
        if self.is_open:
            self.is_open = False
            self._board._detach(self)
            with self._cond:
                self._cond.notify_all()
        super().close()

    # -- from the board, which holds its own lock ----------------------------

    def _deliver(self, data: bytes) -> None:
        with self._cond:
            if self._failure is not None:
                return
            if len(self._rx) + len(data) > RX_LIMIT_BYTES:
                self._failure = (
                    f'{self._board.label}: {len(self._rx)} bytes unread on the port; '
                    'nothing is reading it'
                )
            else:
                self._rx.extend(data)
            self._cond.notify_all()

    def _fail(self, reason: str) -> None:
        with self._cond:
            if self._failure is None:
                self._failure = reason
            self._cond.notify_all()

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
        with self._cond:
            failure = self._failure
        if failure is not None:
            raise SerialException(failure)
        self._board._write(self, data)
        return len(data)

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
