# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated TMCM-6110 at the wire, for the LS720 in the simulator.

`SimulatedTmcm6110Backend` is a `SerialBackend`: it answers discovery with
one port carrying the 6110's USB identity and opens it as a pyserial
`SerialBase`, so the production `Tmcm6110Board` runs over it unchanged, every
datagram encoded and decoded as on the bench.

`SimulatedTmcm6110` is the board behind the port. It answers the TMCL
commands the driver and LumaView Classic's homing send -- SAP, GAP, MVP,
ROR, MST, RFS, SGP, SIO, GIO and the firmware query -- and moves three axes
in time: trapezoidal ramps from each axis's speed and acceleration
registers, a deceleration after MST, limit switches that stop an axis
driving into them, an index pulse inside each X/Y right switch, the deck
lid and the stage's power input. Time is the board's clock, which a test
may run fast.

What no source records and the bench has not measured is chosen here and
said so: the board's power-up register values, where each axis sits at
power-up, where its switches and index pulse are, and the extent of travel.

Tests reach the hardware through the board: open or close the lid, pull
the stage's power, silence the board, unplug its USB, make an axis ignore
MST, break a right switch, and read every command the board was sent.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field

from serial.serialutil import PortNotOpenError, SerialBase, SerialException, to_bytes
from serial.tools.list_ports_common import ListPortInfo

from drivers.tmcm6110 import (
    AP_ACTUAL_POSITION,
    AP_ACTUAL_VELOCITY,
    AP_LEFT_SWITCH,
    AP_MAX_ACCELERATION,
    AP_MAX_POSITIONING_SPEED,
    AP_RIGHT_SWITCH,
    AP_TARGET_POSITION,
    DATAGRAM_BYTES,
    FIRMWARE_VERSION,
    GAP,
    GIO,
    LID_INPUT,
    MODULE_ADDRESS,
    MOTORS,
    MST,
    MVP,
    MVP_ABS,
    MVP_REL,
    POWER_INPUT,
    RFS,
    RFS_START,
    RFS_STATUS,
    RFS_STOP,
    ROR,
    SAP,
    SGP,
    SIO,
    STATUS_OK,
    USB_IDS,
    TmclCommand,
    decode_command,
    encode_reply,
    encode_version_reply,
)
from drivers.tmcm6110_config import usteps_per_s, usteps_per_s2

DEVICE = 'sim:tmcm6110'

# The firmware the bench LS720's board is expected to run (its image is
# the only one Classic's installer carries).
FIRMWARE = '6110V135'

# Status codes the simulated board answers with when it cannot do a command.
_WRONG_CHECKSUM = 1
_INVALID_COMMAND = 2
_WRONG_TYPE = 3
_INVALID_VALUE = 4

# Axis parameters beyond the ones the driver names.
_AP_TARGET_REACHED = 8
_AP_MICROSTEP_RESOLUTION = 140
_AP_RAMP_DIVISOR = 153
_AP_PULSE_DIVISOR = 154
_AP_RIGHT_SWITCH_DISABLE = 12
_AP_LEFT_SWITCH_DISABLE = 13
_AP_REFERENCE_SEARCH_MODE = 193
_AP_REFERENCE_SEARCH_SPEED = 194

# The register values the board powers up with. No source records the
# board's own; these are the ones Classic writes, so a simulated board
# moves at Classic's speeds before anything is written to it.
_POWER_UP_PARAMETERS = {
    'X': {4: 1000, 5: 500, 6: 16, 7: 2, 140: 5, 153: 9, 154: 3},
    'Y': {4: 1000, 5: 500, 6: 16, 7: 4, 140: 5, 153: 9, 154: 3},
    'Z': {4: 250, 5: 2000, 6: 48, 7: 8, 140: 5, 153: 9, 154: 3},
}


@dataclass(frozen=True)
class AxisLayout:
    """Where an axis's switches and reference are, in microsteps on the
    board's own scale with 0 at the reference.

    The board's positive direction drives toward the right switch, which
    sits at the reference end (LumaView Classic homes every axis into its
    right switch). Travel runs negative from the reference.
    """

    # The right switch is engaged at or beyond this position.
    right_switch_at: int
    # The left switch is engaged at or beyond this position (negative).
    left_switch_at: int
    # Where the axis sits when the board powers up.
    start: int
    # Whether the axis has an index pulse at 0; Z's reference is its switch.
    has_index: bool


# Chosen, not measured: each switch just past the travel the shipped
# defaults give (X 120, Y 80, Z 11 mm), each axis mid-travel at power-up.
LAYOUTS = {
    'X': AxisLayout(right_switch_at=3_200, left_switch_at=-774_400, start=-384_000, has_index=True),
    'Y': AxisLayout(right_switch_at=3_200, left_switch_at=-518_400, start=-256_000, has_index=True),
    'Z': AxisLayout(right_switch_at=0, left_switch_at=-218_584, start=-100_000, has_index=False),
}

# The power input's reading with the stage's supply in and out (the values
# Classic's author recorded).
POWER_PRESENT = 239
POWER_ABSENT = 13

# The integration step, in seconds of board time.
_STEP_S = 0.001

# Commands kept for a test to read; enough for any one test, bounded so a
# long simulator session does not grow without end.
COMMAND_LOG_LIMIT = 20_000


@dataclass
class _Axis:
    layout: AxisLayout
    # The physical position, in microsteps from the reference.
    p: float
    params: dict[int, int] = field(default_factory=dict)
    v: float = 0.0
    # The actual-position register reads p - offset.
    offset: float = 0.0
    target: int = 0
    # 'idle', 'position', 'velocity', 'stopping', 'search'.
    mode: str = 'idle'
    velocity_setpoint: float = 0.0
    search_mode: int = 0
    ignores_stop: bool = False
    right_switch_dead: bool = False

    def actual(self) -> int:
        return round(self.p - self.offset)

    def vmax(self, speed: int) -> float:
        return usteps_per_s(speed, self.params.get(_AP_PULSE_DIVISOR, 0))

    def amax(self) -> float:
        return usteps_per_s2(
            self.params.get(AP_MAX_ACCELERATION, 0),
            self.params.get(_AP_PULSE_DIVISOR, 0),
            self.params.get(_AP_RAMP_DIVISOR, 0),
        )

    def right_engaged(self) -> bool:
        return not self.right_switch_dead and self.p >= self.layout.right_switch_at

    def left_engaged(self) -> bool:
        return self.p <= self.layout.left_switch_at


class SimulatedTmcm6110:
    """The board: three axes, two inputs, one reply per datagram."""

    def __init__(self, *, clock=time.monotonic, usb_id: tuple[int, int] = USB_IDS[0]):
        self._clock = clock
        self._now = clock()
        self._lock = threading.Lock()
        self.usb_id = usb_id
        self.version = FIRMWARE
        self.axes = {
            name: _Axis(
                layout=layout, p=float(layout.start), params=dict(_POWER_UP_PARAMETERS[name])
            )
            for name, layout in LAYOUTS.items()
        }
        for axis in self.axes.values():
            # The actual-position register reads 0 at power-up, wherever
            # the axis sits.
            axis.offset = axis.p
        self._by_motor = {MOTORS[name]: axis for name, axis in self.axes.items()}
        self.lid_open = False
        self.powered = True
        self.silent = False
        self.plugged = True
        self._globals: dict[tuple[int, int], int] = {}
        self._outputs: dict[tuple[int, int], int] = {}
        self.commands: collections.deque[TmclCommand] = collections.deque(maxlen=COMMAND_LOG_LIMIT)

    # -- what a test changes --------------------------------------------

    def ignore_stop(self, axis: str, ignores: bool = True) -> None:
        """Make ``axis`` keep moving through MST, as a board that has lost it."""
        with self._lock:
            self.axes[axis].ignores_stop = ignores

    def break_right_switch(self, axis: str) -> None:
        """Make ``axis``'s right switch never close, as a broken or
        unplugged switch would: the axis drives on past it."""
        with self._lock:
            self.axes[axis].right_switch_dead = True

    def position(self, axis: str) -> int:
        """The actual-position register of ``axis``, after the time passed."""
        with self._lock:
            self._advance()
            return self.axes[axis].actual()

    # -- time ----------------------------------------------------------

    def _advance(self) -> None:
        now = self._clock()
        while self._now + _STEP_S <= now:
            self._now += _STEP_S
            for axis in self.axes.values():
                self._step(axis)
        # Partial steps are kept for the next call.

    def _step(self, axis: _Axis) -> None:
        if axis.mode == 'idle':
            return
        a = axis.amax()
        # With the stage's supply out, nothing moves: the board's own
        # behaviour then is not recorded, and this is the simplest guess.
        if not self.powered or a <= 0:
            axis.v = 0.0
            return
        dv = a * _STEP_S
        before = axis.p
        if axis.mode == 'position':
            target_p = axis.target + axis.offset
            d = target_p - axis.p
            if abs(d) < 0.5:
                # The position register is whole microsteps: this is the target.
                axis.p, axis.v = target_p, 0.0
                return
            direction = 1 if d > 0 else -1
            vmax = axis.vmax(axis.params.get(AP_MAX_POSITIONING_SPEED, 0))
            if axis.v * direction < 0:
                axis.v += direction * min(dv, abs(axis.v))
            elif axis.v * axis.v / (2 * a) >= abs(d):
                # Decelerating onto the target, creeping at one step's
                # worth of speed so the ramp's last fraction still arrives.
                axis.v = direction * max(abs(axis.v) - dv, dv)
            else:
                axis.v = direction * min(abs(axis.v) + dv, vmax)
            axis.p += axis.v * _STEP_S
            if (before - target_p) * (axis.p - target_p) <= 0:
                axis.p, axis.v = target_p, 0.0
        elif axis.mode in ('velocity', 'search'):
            setpoint = axis.velocity_setpoint
            if axis.v < setpoint:
                axis.v = min(axis.v + dv, setpoint)
            else:
                axis.v = max(axis.v - dv, setpoint)
            axis.p += axis.v * _STEP_S
        elif axis.mode == 'stopping':
            axis.v -= (1 if axis.v > 0 else -1) * min(dv, abs(axis.v))
            axis.p += axis.v * _STEP_S
            if axis.v == 0:
                axis.mode = 'idle'

        # Hard stops at the switches (the soft-stop flag 0 that Classic sets).
        if axis.v > 0 and axis.right_engaged() and not axis.params.get(_AP_RIGHT_SWITCH_DISABLE):
            axis.p, axis.v = float(axis.layout.right_switch_at), 0.0
            if axis.mode in ('velocity', 'stopping'):
                axis.mode = 'idle'
        if axis.v < 0 and axis.left_engaged() and not axis.params.get(_AP_LEFT_SWITCH_DISABLE):
            axis.p, axis.v = float(axis.layout.left_switch_at), 0.0
            if axis.mode in ('velocity', 'stopping'):
                axis.mode = 'idle'

        if axis.mode == 'search':
            self._search_step(axis, before)

    def _search_step(self, axis: _Axis, before: float) -> None:
        """A reference search ends at its reference, which reads 0 there.

        Mode 65 drives to the right switch; mode 5 drives negative to the
        index pulse, reversing at the left switch.
        """
        if axis.search_mode == 65:
            if axis.right_engaged():
                self._reference_found(axis, float(axis.layout.right_switch_at))
        elif axis.search_mode == 5:
            if axis.layout.has_index and before != axis.p and before * axis.p <= 0:
                self._reference_found(axis, 0.0)
            elif axis.left_engaged():
                axis.velocity_setpoint = abs(axis.velocity_setpoint)

    @staticmethod
    def _reference_found(axis: _Axis, at: float) -> None:
        axis.p, axis.v = at, 0.0
        axis.offset = at
        axis.mode = 'idle'

    # -- the wire ------------------------------------------------------

    def answer(self, datagram: bytes) -> bytes | None:
        """The board's reply to one datagram, or None for no reply."""
        with self._lock:
            self._advance()
            if self.silent:
                return None
            try:
                cmd = decode_command(datagram)
            except ValueError:
                return encode_reply(_WRONG_CHECKSUM, datagram[1], 0)
            if cmd.address != MODULE_ADDRESS:
                return None
            self.commands.append(cmd)
            if cmd.command == FIRMWARE_VERSION:
                if cmd.type != 0:
                    return encode_reply(_WRONG_TYPE, cmd.command, 0)
                return encode_version_reply(self.version)
            status, value = self._execute(cmd)
            return encode_reply(status, cmd.command, value)

    def _execute(self, cmd: TmclCommand) -> tuple[int, int]:
        if cmd.command == GIO:
            if (cmd.type, cmd.motor) == LID_INPUT:
                return STATUS_OK, int(self.lid_open)
            if (cmd.type, cmd.motor) == POWER_INPUT:
                return STATUS_OK, POWER_PRESENT if self.powered else POWER_ABSENT
            return STATUS_OK, 0
        if cmd.command == SIO:
            self._outputs[(cmd.type, cmd.motor)] = cmd.value
            return STATUS_OK, cmd.value
        if cmd.command == SGP:
            self._globals[(cmd.type, cmd.motor)] = cmd.value
            return STATUS_OK, cmd.value

        axis = self._by_motor.get(cmd.motor)
        if axis is None:
            return _INVALID_VALUE, 0
        if cmd.command == GAP:
            return STATUS_OK, self._get(axis, cmd.type)
        if cmd.command == SAP:
            self._set(axis, cmd.type, cmd.value)
            return STATUS_OK, cmd.value
        if cmd.command == MVP:
            if cmd.type == MVP_ABS:
                axis.target = cmd.value
            elif cmd.type == MVP_REL:
                axis.target = axis.actual() + cmd.value
            else:
                return _WRONG_TYPE, 0
            axis.mode = 'position'
            return STATUS_OK, axis.target if cmd.type == MVP_REL else cmd.value
        if cmd.command == ROR:
            axis.mode = 'velocity'
            axis.velocity_setpoint = axis.vmax(cmd.value)
            return STATUS_OK, cmd.value
        if cmd.command == MST:
            if not axis.ignores_stop:
                axis.mode = 'stopping'
            return STATUS_OK, 0
        if cmd.command == RFS:
            return self._reference_search(axis, cmd.type)
        return _INVALID_COMMAND, 0

    def _reference_search(self, axis: _Axis, type_: int) -> tuple[int, int]:
        if type_ == RFS_START:
            mode = axis.params.get(_AP_REFERENCE_SEARCH_MODE, 0)
            if mode not in (5, 65):
                # Only the two searches Classic runs are modelled.
                return _INVALID_VALUE, 0
            speed = axis.vmax(axis.params.get(_AP_REFERENCE_SEARCH_SPEED, 0))
            axis.search_mode = mode
            axis.velocity_setpoint = speed if mode == 65 else -speed
            axis.mode = 'search'
            return STATUS_OK, 0
        if type_ == RFS_STOP:
            if axis.mode == 'search':
                axis.mode = 'stopping'
            return STATUS_OK, 0
        if type_ == RFS_STATUS:
            return STATUS_OK, int(axis.mode == 'search')
        return _WRONG_TYPE, 0

    def _get(self, axis: _Axis, parameter: int) -> int:
        if parameter == AP_TARGET_POSITION:
            return axis.target
        if parameter == AP_ACTUAL_POSITION:
            return axis.actual()
        if parameter == AP_ACTUAL_VELOCITY:
            unit = axis.vmax(1)
            return round(axis.v / unit)
        if parameter == _AP_TARGET_REACHED:
            return int(axis.v == 0 and axis.actual() == axis.target)
        if parameter == AP_RIGHT_SWITCH:
            return int(axis.right_engaged())
        if parameter == AP_LEFT_SWITCH:
            return int(axis.left_engaged())
        return axis.params.get(parameter, 0)

    @staticmethod
    def _set(axis: _Axis, parameter: int, value: int) -> None:
        if parameter == AP_TARGET_POSITION:
            axis.target = value
            axis.mode = 'position'
        elif parameter == AP_ACTUAL_POSITION:
            # The position register is rewritten where the axis stands; it
            # then holds still until it is given a target.
            axis.offset = axis.p - value
            if axis.mode == 'position':
                axis.mode = 'idle'
                axis.v = 0.0
        else:
            axis.params[parameter] = value


class SimulatedTmcm6110Port(SerialBase):
    """One connection to a `SimulatedTmcm6110`."""

    def __init__(self, board: SimulatedTmcm6110, **kwargs):
        self._board = board
        self._rx = bytearray()
        self._tx = bytearray()
        super().__init__(**kwargs)

    def open(self) -> None:
        if self.is_open:
            raise SerialException('Port is already open.')
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        if not self._board.plugged:
            raise SerialException(f'{self._port}: no such device')
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def _reconfigure_port(self) -> None:
        pass

    def _check(self) -> None:
        if not self.is_open:
            raise PortNotOpenError()
        if not self._board.plugged:
            raise SerialException(f'{self._port}: the device was unplugged')

    @property
    def in_waiting(self) -> int:
        self._check()
        return len(self._rx)

    def write(self, data: bytes) -> int:
        self._check()
        data = to_bytes(data)
        self._tx.extend(data)
        while len(self._tx) >= DATAGRAM_BYTES:
            datagram = bytes(self._tx[:DATAGRAM_BYTES])
            del self._tx[:DATAGRAM_BYTES]
            reply = self._board.answer(datagram)
            if reply is not None:
                self._rx.extend(reply)
        return len(data)

    def read(self, size: int = 1) -> bytes:
        self._check()
        if len(self._rx) < size and self._timeout:
            # Nothing more is coming: the board answers as the command is
            # written, so a short reply is one it never sent.
            time.sleep(self._timeout)
        data = bytes(self._rx[:size])
        del self._rx[:size]
        return data

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._check()
        self._rx.clear()

    def reset_output_buffer(self) -> None:
        self._check()
        self._tx.clear()


class SimulatedTmcm6110Backend:
    """Discovery and open for one simulated TMCM-6110."""

    def __init__(self, board: SimulatedTmcm6110 | None = None):
        self.board = board if board is not None else SimulatedTmcm6110()

    def comports(self) -> list[ListPortInfo]:
        if not self.board.plugged:
            return []
        info = ListPortInfo(DEVICE)
        info.vid, info.pid = self.board.usb_id
        info.description = 'Simulated TMCM-6110'
        return [info]

    def open(self, **kwargs) -> SimulatedTmcm6110Port:
        if kwargs.get('port') != DEVICE:
            raise SerialException(f'no simulated TMCM-6110 at {kwargs.get("port")!r}')
        return SimulatedTmcm6110Port(self.board, **kwargs)
