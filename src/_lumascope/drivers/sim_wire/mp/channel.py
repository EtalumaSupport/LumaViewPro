# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The control channel between the port (LumaViewPro's interpreter) and the
# simulated hardware (the MicroPython child): both ends of each message are
# here, so the two sides cannot disagree. Imports under both interpreters;
# pure Python, nothing host-only.
#
# - A fault change goes to the child as one line on its fault pipe.
# - A register write, and a chip's state (what an LED DAC's channels are
#   driving), comes back on the child's stdout as a frame between
#   ORACLE_START and ORACLE_END, bytes the firmware never prints, so a frame
#   is found anywhere in the stream, not only at a line start. A frame's
#   first word is its kind: W a write, S a state.

import json

ORACLE_START = 0x1E
ORACLE_END = 0x1F

WRITE = 'W'
STATE = 'S'

# The environment variable naming the fault pipe's fd in the child. The fd
# keeps its number across the launch, so no shell redirection moves it: dash,
# Linux's sh, refuses one naming an fd above 9. A handle between the two
# processes, like MICROPYPATH, not a setting.
FAULTS_FD_ENV = 'SIM_WIRE_FAULTS_FD'


def fault_line(on: bool, axis: str, name: str) -> bytes:
    return f'{"FAULT" if on else "CLEAR"} {axis} {name}\n'.encode()


def parse_fault_line(line: bytes) -> tuple:
    """(on, axis, name)"""
    verb, axis, name = line.decode().split()
    if verb not in ('FAULT', 'CLEAR'):
        raise ValueError(f'not a fault line: {repr(line)}')
    return verb == 'FAULT', axis, name


def write_frame(chip: str, axis: str | None, reg: int, value: int) -> str:
    return f'{chr(ORACLE_START)}{WRITE} {chip} {axis or "-"} 0x{reg:02x} 0x{value:08x}{chr(ORACLE_END)}'


def state_frame(chip: str, state: list) -> str:
    return f'{chr(ORACLE_START)}{STATE} {chip} {json.dumps(state)}{chr(ORACLE_END)}'


def frame_kind(body: bytes) -> str:
    """WRITE or STATE, from a frame's body without its start and end bytes."""
    return body[:1].decode()


def parse_state(body: bytes) -> tuple:
    """(chip, state) from a state frame's body."""
    tag, chip, state = body.decode().split(' ', 2)
    if tag != STATE:
        raise ValueError(f'not a state frame: {repr(body)}')
    return chip, json.loads(state)


def parse_write(body: bytes) -> tuple:
    """(chip, axis or None, reg, value) from a frame's body, without its
    start and end bytes."""
    tag, chip, axis, reg, value = body.decode().split()
    if tag != WRITE:
        raise ValueError(f'not a register-write frame: {repr(body)}')
    return chip, None if axis == '-' else axis, int(reg, 16), int(value, 16)
