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

CTRL_C = 0x03
CTRL_D = 0x04
REPL_PROMPT = b'>>> '

# Unread bytes held for the driver before the port refuses to hold more. A
# board's longest reply is a few kilobytes; a megabyte unread means nothing
# is reading the port.
RX_LIMIT_BYTES = 1 << 20

# Runs the firmware with a watcher beside it. The watcher polls this
# interpreter's pid and kills the firmware (whose pid is the shell's, via
# exec) when this interpreter is gone, or exits when the firmware is.
_LAUNCH = (
    '(while kill -0 "$1" 2>/dev/null && kill -0 $$ 2>/dev/null; do sleep 1; done; '
    'kill -9 $$ 2>/dev/null) </dev/null >/dev/null 2>&1 & '
    'exec "$2" -i -c "import main"'
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
        try:
            self._proc = subprocess.Popen(
                ['sh', '-c', _LAUNCH, 'sh', str(os.getpid()), self._image.runtime],
                cwd=self._workdir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except OSError as e:
            raise SerialException(f'{self._image.label}: board process did not start: {e}') from e
        with self._cond:
            self._failure = None
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

    def _at_repl(self) -> bool:
        with self._cond:
            return self._tail == REPL_PROMPT

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
                self._signal(signal.SIGINT)
                start = i + 1
            elif byte == CTRL_D and self._at_repl():
                self._send(data[start:i])
                self._restart()
                start = i + 1
        self._send(data[start:])
        return len(data)

    def _send(self, data: bytes):
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
