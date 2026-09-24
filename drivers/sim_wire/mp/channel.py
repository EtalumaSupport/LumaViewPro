# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The control channel between the port (LumaViewPro's interpreter) and the
# simulated hardware (the MicroPython child): both ends of each message are
# here, so the two sides cannot disagree. Imports under both interpreters;
# pure Python, nothing host-only.
#
# - A fault change goes to the child as one line on its fault pipe.
# - A register write comes back on the child's stdout as a frame between
#   ORACLE_START and ORACLE_END, bytes the firmware never prints, so a frame
#   is found anywhere in the stream, not only at a line start.

ORACLE_START = 0x1E
ORACLE_END = 0x1F


def fault_line(on: bool, axis: str, name: str) -> bytes:
    return f'{"FAULT" if on else "CLEAR"} {axis} {name}\n'.encode()


def parse_fault_line(line: bytes) -> tuple:
    """(on, axis, name)"""
    verb, axis, name = line.decode().split()
    if verb not in ('FAULT', 'CLEAR'):
        raise ValueError(f'not a fault line: {repr(line)}')
    return verb == 'FAULT', axis, name


def write_frame(chip: str, axis: str | None, reg: int, value: int) -> str:
    return f'{chr(ORACLE_START)}W {chip} {axis or "-"} 0x{reg:02x} 0x{value:08x}{chr(ORACLE_END)}'


def parse_write(body: bytes) -> tuple:
    """(chip, axis or None, reg, value) from a frame's body, without its
    start and end bytes."""
    tag, chip, axis, reg, value = body.decode().split()
    if tag != 'W':
        raise ValueError(f'not a register-write frame: {repr(body)}')
    return chip, None if axis == '-' else axis, int(reg, 16), int(value, 16)
