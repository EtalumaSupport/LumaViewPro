# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scripted serial port behind the drivers' own ``backend=`` seam.

For a claim about the port itself -- a write that times out, a link that
fails, a board that sends nothing at all -- the firmware plays no part, so
the real driver runs every line over a port whose far end the test sets.
A claim about what a board answers runs on the firmware instead
(`tests/wire_boards.py`): a reply written here by hand is not the
firmware's.

A read on this port returns at once with what has arrived, or with nothing:
what a read that waited out its timeout returns, without the wait.
"""

import threading

import serial
from serial.tools.list_ports_common import ListPortInfo

LED_VID, LED_PID = 0x0424, 0x704C


class ScriptedLink:
    """The far end of every port the backend opens: what the board answers
    each command with, what the link does to a write, and what reached it.

    ``answers`` maps a command to the lines the board sends back; a command
    with no entry is answered with nothing. ``write_fault`` and
    ``read_fault``, when set, are raised by every write or every read, as
    pyserial raises a write timeout or a failed link.
    """

    def __init__(self, answers: dict[str, list[str]] | None = None):
        self.answers = dict(answers or {})
        self.write_fault: Exception | None = None
        self.read_fault: Exception | None = None
        self.written: list[bytes] = []
        self.opens = 0
        self._lock = threading.Lock()

    def commands(self) -> list[str]:
        """Every command line that reached the board, in order."""
        with self._lock:
            return [w.strip().decode() for w in self.written if w.endswith(b'\n') and w.strip()]


class ScriptedPort:
    """The pyserial surface the serial drivers use, over a `ScriptedLink`."""

    def __init__(self, link: ScriptedLink, timeout=None, write_timeout=None, **_kwargs):
        self._link = link
        self._buffer = bytearray()
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.is_open = True

    def write(self, data: bytes) -> int:
        if self._link.write_fault is not None:
            raise self._link.write_fault
        with self._link._lock:
            self._link.written.append(bytes(data))
            answer = self._link.answers.get(data.strip().decode('utf-8', 'ignore'), [])
        for line in answer:
            self._buffer += line.encode() + b'\r\n'
        return len(data)

    @property
    def in_waiting(self) -> int:
        return len(self._buffer)

    def read(self, size: int = 1) -> bytes:
        if self._link.read_fault is not None:
            raise self._link.read_fault
        chunk = bytes(self._buffer[:size])
        del self._buffer[:size]
        return chunk

    def readline(self) -> bytes:
        end = self._buffer.find(b'\n')
        return self.read(len(self._buffer) if end < 0 else end + 1)

    def reset_input_buffer(self) -> None:
        self._buffer.clear()

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False


class ScriptedBackend:
    """One port, with the LED board's USB ids unless told otherwise, opened
    onto the given link."""

    def __init__(self, link: ScriptedLink, vid: int = LED_VID, pid: int = LED_PID):
        self.link = link
        self._vid, self._pid = vid, pid

    def comports(self) -> list[ListPortInfo]:
        info = ListPortInfo('scripted')
        info.vid, info.pid = self._vid, self._pid
        return [info]

    def open(self, **kwargs) -> ScriptedPort:
        if kwargs.get('port') != 'scripted':
            raise serial.SerialException(f'no port {kwargs.get("port")!r}')
        self.link.opens += 1
        return ScriptedPort(self.link, **kwargs)
