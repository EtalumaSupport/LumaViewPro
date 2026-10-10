# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where a SerialBoard's port comes from: discovery and open, and nothing else.

A board driver finds its port by USB VID/PID and opens it; everything after
the open -- connect, reset, detection, exchange, timeouts -- is the driver's
and talks to whatever object the open returned through pyserial's own
surface. That makes the open the one place a simulated port can stand in for
a real one while every line of the driver still runs: the simulator answers
discovery with its own ports and opens them as pyserial ``SerialBase``
objects, so the driver cannot tell the difference and has no simulate branch.

The pyserial backend resolves ``serial.Serial`` and ``list_ports.comports``
at call time, not at import, so anything that replaces those module
attributes (the test suite's real-port refusal does) still sees every open
and every enumeration made through it.
"""

from typing import Protocol

import serial
import serial.tools.list_ports as list_ports
from serial.tools.list_ports_common import ListPortInfo


class SerialBackend(Protocol):
    def comports(self) -> list[ListPortInfo]:
        """Every port this backend can open, with its USB VID/PID."""
        ...

    def open(self, **kwargs) -> serial.SerialBase:
        """Open a port found by ``comports()``; ``kwargs`` are pyserial's
        ``Serial`` arguments. Raises ``serial.SerialException`` on failure."""
        ...


class PyserialBackend:
    """The machine's real serial ports."""

    def comports(self) -> list[ListPortInfo]:
        return list_ports.comports(include_links=True)

    def open(self, **kwargs) -> serial.SerialBase:
        return serial.Serial(**kwargs)


PYSERIAL = PyserialBackend()
