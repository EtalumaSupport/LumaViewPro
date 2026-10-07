# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Tmcm6110Board -- the LS720's stage: a Trinamic TMCM-6110 speaking TMCL.

The board is a USB virtual COM port that answers 9-byte binary datagrams,
one reply per command, nothing unsolicited. It holds no per-unit config and
runs no Etaluma firmware; everything it is told comes from the "TMCM-6110"
section of the shipped motor defaults (``Tmcm6110Config``).

Positions are microsteps on the board, 0 at each axis's index (Z's: its
switch, backed off upward) and negative away from it. The API sees
micrometres in the frame the plate transform shares with every model: the
board's 0 sits at the axis's index position from the config, and its
direction says which way the API's position grows. X grows away from the plate's column-12 end, as
on the EL-0940, so X runs down from its index; Y and Z run up from theirs.

The board's own registers are the one store of where each axis is and where
it is going: the target is read back with ``GAP 0``, never remembered here.

The lid. The LS720's deck lid is an input on the board, not an interlock the
board enforces. As LumaView Classic did, every command that starts X or Y
motion reads it first, in the one path every datagram takes and under the
same lock as the command, a home's starts included; open, all three axes
are stopped and the command is refused with ``MotionInterlockError``. Z
moves with the lid open, and stopping is never gated.

The datagram codec is here once, both directions: the simulated board
(``drivers/simulated_tmcm6110.py``) decodes what this driver encodes.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple, NoReturn

import serial

from drivers.exceptions import HardwareError, MotionInterlockError
from drivers.motorconfig import read_only_axes_config
from drivers.registry import motor_registry
from drivers.serial_backend import PYSERIAL, SerialBackend
from drivers.tmcm6110_config import Tmcm6110Config

logger = logging.getLogger('LVP.drivers.tmcm6110')

# The two USB identities a TMCM-6110 enumerates as: the current Trinamic
# one, and the one boards of the V1.26 firmware era used.
USB_IDS = ((0x2A3C, 0x0100), (0x16D0, 0x0650))

# The board's factory module address, which LumaView Classic used; replies
# come from host address 2.
MODULE_ADDRESS = 1
HOST_ADDRESS = 2

DATAGRAM_BYTES = 9
STATUS_OK = 100

# TMCL command numbers.
ROR = 1
MST = 3
MVP = 4
SAP = 5
GAP = 6
SGP = 9
RFS = 13
SIO = 14
GIO = 15
FIRMWARE_VERSION = 136

COMMAND_NAMES = {
    ROR: 'ROR',
    MST: 'MST',
    MVP: 'MVP',
    SAP: 'SAP',
    GAP: 'GAP',
    SGP: 'SGP',
    RFS: 'RFS',
    SIO: 'SIO',
    GIO: 'GIO',
    FIRMWARE_VERSION: 'FIRMWARE_VERSION',
}

# Reply status codes, as the TMCL firmware manual names them.
STATUS_WORDS = {
    1: 'wrong checksum',
    2: 'invalid command',
    3: 'wrong type',
    4: 'invalid value',
    5: 'configuration EEPROM locked',
    6: 'command not available',
}

# MVP types.
MVP_ABS = 0
MVP_REL = 1

# RFS types.
RFS_START = 0
RFS_STOP = 1
RFS_STATUS = 2

# Axis parameters.
AP_TARGET_POSITION = 0
AP_ACTUAL_POSITION = 1
AP_ACTUAL_VELOCITY = 3
AP_MAX_POSITIONING_SPEED = 4
AP_MAX_ACCELERATION = 5
AP_TARGET_REACHED = 8
AP_RIGHT_SWITCH = 10
AP_LEFT_SWITCH = 11
AP_REFERENCE_SEARCH_MODE = 193
AP_REFERENCE_SEARCH_SPEED = 194
AP_REFERENCE_SWITCH_SPEED = 195

# The axis parameters an axis is initialised with, by the name the config
# section uses, in the order Classic writes them: speed and acceleration
# after the two divisors, as the board needs.
INIT_PARAMETERS = (
    ('Right Limit Switch Disable', 12),
    ('Left Limit Switch Disable', 13),
    ('Soft Stop Flag', 149),
    ('Max Current', 6),
    ('Standby Current', 7),
    ('Microstep Resolution', 140),
    ('Ramp Divisor', 153),
    ('Pulse Divisor', 154),
    ('Max Positioning Speed', AP_MAX_POSITIONING_SPEED),
    ('Max Acceleration', AP_MAX_ACCELERATION),
)

# Global parameters (bank 0) and outputs, as (number, bank).
GP_AUTO_START_MODE = (77, 0)
GP_END_SWITCH_POLARITY = (79, 0)
SWITCH_PULLUPS_OUTPUT = (0, 0)
FAN_OUTPUT = (1, 2)

# The inputs, as (GIO type, bank). The lid reads 1 open; the stage's supply
# reads about 240 present and about 13 absent, so above 20 is present
# (LumaView Classic's threshold, from Trinamic).
LID_INPUT = (5, 0)
POWER_INPUT = (8, 1)
POWER_PRESENT_ABOVE = 20

MOTORS = {'X': 0, 'Y': 1, 'Z': 2}
LID_GATED_AXES = ('X', 'Y')
LID_GATED_MOTORS = frozenset(MOTORS[axis] for axis in LID_GATED_AXES)

# How long one reply may take: LumaView Classic's per-command wait.
REPLY_TIMEOUT_S = 0.5

# How long a stop waits for every stopped axis to read velocity 0 before
# writing its target. Classic's shipped ramps decelerate from full speed in
# about a second (derived from the ramp constants, not measured).
STOP_SETTLE_S = 2.0
STOP_POLL_S = 0.01

# How often a home reads the lid, the abort and its phase's progress
# (Classic's homing thread polled every 100 ms), and how long one phase may
# take (Classic's 200 polls of 500 ms).
HOME_POLL_S = 0.1
HOME_PHASE_TIMEOUT_S = 100.0

_DATAGRAM = struct.Struct('>BBBBiB')


class TmclCommand(NamedTuple):
    address: int
    command: int
    type: int
    motor: int
    value: int


class TmclReply(NamedTuple):
    reply_address: int
    module_address: int
    status: int
    command: int
    value: int


def _checksum(head: bytes) -> int:
    return sum(head) & 0xFF


def _pack(a: int, b: int, c: int, d: int, value: int) -> bytes:
    head = _DATAGRAM.pack(a, b, c, d, value, 0)[:8]
    return head + bytes([_checksum(head)])


def _unpack(datagram: bytes) -> tuple[int, int, int, int, int]:
    if len(datagram) != DATAGRAM_BYTES:
        raise ValueError(f'a TMCL datagram is {DATAGRAM_BYTES} bytes, not {len(datagram)}')
    if _checksum(datagram[:8]) != datagram[8]:
        raise ValueError(f'TMCL datagram checksum mismatch: {datagram.hex(" ")}')
    return _DATAGRAM.unpack(datagram)[:5]


def encode_command(command: int, type_: int, motor: int, value: int = 0) -> bytes:
    """One command datagram to the board at ``MODULE_ADDRESS``."""
    return _pack(MODULE_ADDRESS, command, type_, motor, value)


def decode_command(datagram: bytes) -> TmclCommand:
    """Raises ValueError on a wrong length or checksum."""
    return TmclCommand(*_unpack(datagram))


def encode_reply(status: int, command: int, value: int) -> bytes:
    """One reply datagram from the board at ``MODULE_ADDRESS``."""
    return _pack(HOST_ADDRESS, MODULE_ADDRESS, status, command, value)


def decode_reply(datagram: bytes) -> TmclReply:
    """Raises ValueError on a wrong length or checksum."""
    return TmclReply(*_unpack(datagram))


def starts_lid_gated_motion(command: int, type_: int, motor: int) -> bool:
    """Whether a datagram starts X or Y moving: ``ROR``, ``MVP``, or an
    ``RFS`` start on one of them. A stop (``MST``, the ``SAP 0`` after it),
    an ``RFS`` stop, a poll and the lid read itself start nothing."""
    return motor in LID_GATED_MOTORS and (
        command in (ROR, MVP) or (command, type_) == (RFS, RFS_START)
    )


def encode_version_reply(version: str) -> bytes:
    """The reply to FIRMWARE_VERSION type 0: the host address, then the
    version's 8 ASCII characters where the rest of a reply would be, with
    no checksum."""
    text = version.encode('ascii')
    if len(text) != DATAGRAM_BYTES - 1:
        raise ValueError(f'a TMCL version string is 8 characters, not {version!r}')
    return bytes([HOST_ADDRESS]) + text


def decode_version_reply(datagram: bytes) -> str:
    return datagram[1:DATAGRAM_BYTES].decode('ascii', errors='replace')


# Tried after the EL-0940 motor board: an LS720 host has no EL-0940 board,
# and on every other host this finds nothing and opens nothing.
@motor_registry.register('tmcm6110', priority=80)
class Tmcm6110Board:
    """The TMCM-6110 stage controller behind the LS720's X, Y and Z.

    Constructed on every host the registry tries it on, so it opens only
    ports with a 6110's USB identity, closes any whose firmware query does
    not answer as a 6110, and never raises for finding none: it answers
    ``found = False``.
    """

    def __init__(self, *, motorconfig_defaults: Mapping, backend: SerialBackend = PYSERIAL):
        # One datagram in flight: the board must not get a second command
        # before it has replied to the first. Re-entrant so a lid refusal
        # can stop the motors inside the command it refuses.
        self._lock = threading.RLock()
        # Set by motor_stop before it takes the lock, so a home in another
        # thread sends nothing more once a stop has begun.
        self._home_abort = threading.Event()
        self._backend = backend
        self._serial: serial.SerialBase | None = None
        self.port: str | None = None
        self.firmware_version: str | None = None
        self.firmware_date = None
        self.motorconfig: Tmcm6110Config | None = None
        self.axes_config = read_only_axes_config({})

        self._serial = self._open_identified()
        self.found = self._serial is not None
        self.firmware_responding = self.found
        if not self.found:
            logger.info('[TMCM-6110 ] No TMCM-6110 found')
            return

        try:
            self.motorconfig = Tmcm6110Config(motorconfig_defaults)
        except ValueError:
            self.disconnect()
            raise
        self.axes_config = read_only_axes_config(
            {
                axis: {
                    'limits': self.motorconfig.limits_um(axis),
                    'move_func': partial(self._um2ustep, axis),
                }
                for axis in MOTORS
            }
        )
        logger.info(f'[TMCM-6110 ] Found {self.firmware_version} on {self.port}')

    # ------------------------------------------------------------------
    # The port
    # ------------------------------------------------------------------

    def _open_identified(self) -> serial.SerialBase | None:
        """Open the first port with a 6110's USB identity whose firmware
        query answers as a 6110, or None."""
        for info in self._backend.comports():
            if (info.vid, info.pid) not in USB_IDS:
                continue
            try:
                port = self._backend.open(
                    port=info.device,
                    baudrate=9600,
                    timeout=REPLY_TIMEOUT_S,
                    write_timeout=REPLY_TIMEOUT_S,
                )
            except serial.SerialException as e:
                logger.warning(f'[TMCM-6110 ] {info.device} could not be opened: {e}')
                continue
            version = self._query_version(port)
            if version is not None and version.startswith('6110'):
                self.firmware_version = version
                self.port = info.device
                return port
            port.close()
            logger.warning(
                f'[TMCM-6110 ] {info.device} has a TMCM-6110 USB identity but answered '
                f'{version!r} to the firmware query; closed'
            )
        return None

    @staticmethod
    def _query_version(port: serial.SerialBase) -> str | None:
        try:
            port.write(encode_command(FIRMWARE_VERSION, 0, 0))
            reply = port.read(DATAGRAM_BYTES)
        except serial.SerialException:
            return None
        if len(reply) < DATAGRAM_BYTES:
            return None
        return decode_version_reply(reply)

    def _drop_port(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except serial.SerialException as e:
                logger.warning(f'[TMCM-6110 ] Closing {self.port} failed: {e}')
            self._serial = None

    def _exchange(self, command: int, type_: int, motor: int, value: int = 0) -> int:
        """Send one command and return its reply's value.

        A port that is gone is looked for again first, as a replugged
        board would be. A command that starts X or Y moving reads the lid
        first, under the same lock. Whatever is waiting on the port is
        discarded before the write, so a reply that came late is never
        read as the next command's. One reply that does not come, does not
        check, or answers another command is one failed command: the port
        is kept, as the EL-0940 keeps its port, and the next command starts
        clean. The port is dropped only when the port object itself raises,
        whatever the type -- a pulled cable raises pyserial's error, a
        flush on a vanished device the OS's -- which is a lost board.

        Raises:
            MotionInterlockError: the lid is open and the command would
                start X or Y; all three axes stopped, the command not sent.
            HardwareError: no board, no reply, a garbled reply, or a status
                other than success; the message names the command.
        """
        what = f'{COMMAND_NAMES.get(command, command)} {type_}, motor {motor}, value {value}'
        with self._lock:
            if self._serial is None:
                self._serial = self._open_identified()
                if self._serial is None:
                    raise HardwareError(f'{what}: the TMCM-6110 is not connected')
            if starts_lid_gated_motion(command, type_, motor) and self._lid_open():
                self._refuse_for_lid(moved=False)
            try:
                self._serial.reset_input_buffer()
                self._serial.write(encode_command(command, type_, motor, value))
                raw = self._serial.read(DATAGRAM_BYTES)
            except serial.SerialTimeoutException as e:
                raise HardwareError(
                    f'{what}: the TMCM-6110 did not take the write within {REPLY_TIMEOUT_S} s'
                ) from e
            except Exception as e:
                self._drop_port()
                raise HardwareError(f'{what}: the TMCM-6110 port failed: {e}') from e
            if len(raw) < DATAGRAM_BYTES:
                raise HardwareError(
                    f'{what}: no reply from the TMCM-6110 within {REPLY_TIMEOUT_S} s'
                )
            try:
                reply = decode_reply(raw)
            except ValueError as e:
                raise HardwareError(f'{what}: {e}') from e
            if reply.command != command:
                raise HardwareError(f'{what}: the reply answers command {reply.command}')
            if reply.status != STATUS_OK:
                raise HardwareError(
                    f'{what}: the TMCM-6110 refused it with status {reply.status} '
                    f'({STATUS_WORDS.get(reply.status, "unknown status")})'
                )
            return reply.value

    def connect(self) -> None:
        with self._lock:
            if self._serial is None:
                self._serial = self._open_identified()

    def disconnect(self) -> None:
        with self._lock:
            self._drop_port()

    def is_connected(self) -> bool:
        return self._serial is not None

    # ------------------------------------------------------------------
    # The inputs and the one stop
    # ------------------------------------------------------------------

    def _lid_open(self) -> bool:
        return self._exchange(GIO, *LID_INPUT) == 1

    def interlocks(self) -> frozenset[str]:
        """The lid and the stage's power, read from the board now."""
        with self._lock:
            open_now = set()
            if self._lid_open():
                open_now.add('lid_open')
            if self._exchange(GIO, *POWER_INPUT) <= POWER_PRESENT_ABOVE:
                open_now.add('stage_unpowered')
            return frozenset(open_now)

    def _stop(self, axes) -> None:
        """Stop ``axes`` and leave each target equal to where it stopped.

        ``MST`` decelerates; the target is written only once the axis reads
        velocity 0, since a target written while it is still decelerating
        would send it back. The target is written with ``SAP 0``, never
        ``MVP``, which is a motion command and gated on the lid. Never gated:
        a stop must work with the lid open.

        Raises:
            HardwareError: an axis still moving after ``STOP_SETTLE_S``,
                naming it; every axis that did stop has its target written.
        """
        with self._lock:
            for axis in axes:
                self._exchange(MST, 0, MOTORS[axis])
            deadline = time.monotonic() + STOP_SETTLE_S
            moving = list(axes)
            while True:
                moving = [
                    axis for axis in moving if self._exchange(GAP, AP_ACTUAL_VELOCITY, MOTORS[axis])
                ]
                if not moving or time.monotonic() >= deadline:
                    break
                time.sleep(STOP_POLL_S)
            for axis in axes:
                if axis in moving:
                    continue
                actual = self._exchange(GAP, AP_ACTUAL_POSITION, MOTORS[axis])
                self._exchange(SAP, AP_TARGET_POSITION, MOTORS[axis], actual)
            if moving:
                raise HardwareError(f'{", ".join(moving)} still moving {STOP_SETTLE_S} s after MST')

    def _refuse_for_lid(self, *, moved: bool) -> NoReturn:
        """The lid is open: stop all three axes and refuse.

        The refusal says a stop was tried whether or not it settled, so a
        move waiting on an axis the stop halted learns it was stopped. An
        axis still moving at the settle bound is logged and left moving for
        the monitor to fault, as after a Stop that did not settle.
        """
        try:
            self._stop(MOTORS)
        except HardwareError as e:
            logger.error(f'[TMCM-6110 ] Lid open: the stop did not complete: {e}')
        raise MotionInterlockError('lid_open', moved=moved, stopped=True)

    def motor_stop(self) -> bool:
        """Stop all three axes, each target left where it stopped, and end
        any home in progress.

        Returns:
            bool: True once every axis has stopped and its target is written.

        Raises:
            HardwareError: the board did not answer, or an axis did not stop.
        """
        self._home_abort.set()
        self._stop(MOTORS)
        return True

    # ------------------------------------------------------------------
    # Moves and arrival
    # ------------------------------------------------------------------

    def _motor(self, axis: str) -> int:
        if axis not in MOTORS:
            raise HardwareError(f'Unsupported axis ({axis})')
        return MOTORS[axis]

    def _index_usteps(self, axis: str) -> int:
        return self._um2ustep(axis, self.motorconfig.index_position_um(axis))

    def _to_board(self, axis: str, usteps: int) -> int:
        """An API position in microsteps as the board's position."""
        return self.motorconfig.direction(axis) * (usteps - self._index_usteps(axis))

    def _from_board(self, axis: str, raw: int) -> int:
        """The board's position as an API position in microsteps."""
        return self._index_usteps(axis) + self.motorconfig.direction(axis) * raw

    def move(self, axis: str, steps: int) -> None:
        """Move ``axis`` to the API position ``steps`` microsteps.

        Returns once the board holds the new target.

        Raises:
            MotionInterlockError: X or Y with the lid open; nothing moved.
            HardwareError: an unsupported axis, or the board did not take it.
        """
        self._exchange(MVP, MVP_ABS, self._motor(axis), self._to_board(axis, steps))

    def backlash_um(self) -> float:
        """Z antibacklash, um: none. Classic compensated no backlash on the
        LS720, and none is measured, so a Z move has no approach leg."""
        return 0.0

    def move_abs_pos(self, axis: str, pos: float) -> None:
        """Move to ``pos`` micrometres, in one leg.

        Travel is not checked here: the motion API refuses a target
        outside travel before it calls this.
        """
        self._motor(axis)
        self.move(axis, self.axes_config[axis]['move_func'](pos))

    def _read_usteps(self, axis: str, parameter: int) -> int:
        return self._from_board(axis, self._exchange(GAP, parameter, self._motor(axis)))

    def target_pos_steps(self, axis: str) -> int:
        """The board's target in API microsteps.

        Raises:
            HardwareError: the board did not report the position.
        """
        return self._read_usteps(axis, AP_TARGET_POSITION)

    def current_pos_steps(self, axis: str) -> int:
        """The actual position in API microsteps.

        Raises:
            HardwareError: the board did not report the position.
        """
        return self._read_usteps(axis, AP_ACTUAL_POSITION)

    def _usteps_to_um(self, axis: str, usteps: int) -> float:
        return self._ustep2um(axis, usteps)

    def target_pos(self, axis: str) -> float:
        """The board's target in micrometres.

        Raises:
            HardwareError: the board did not report the position.
        """
        return self._usteps_to_um(axis, self.target_pos_steps(axis))

    def current_pos(self, axis: str) -> float:
        """The actual position in micrometres.

        Raises:
            HardwareError: the board did not report the position.
        """
        return self._usteps_to_um(axis, self.current_pos_steps(axis))

    def target_status(self, axis: str) -> bool:
        """True when the axis stands still at the board's target.

        Straight after ``MVP`` the axis may not have started, so standing
        still alone is not arrival; the position must equal the target.

        Raises:
            HardwareError: the board did not answer.
        """
        motor = self._motor(axis)
        with self._lock:
            if self._exchange(GAP, AP_ACTUAL_VELOCITY, motor) != 0:
                return False
            actual = self._exchange(GAP, AP_ACTUAL_POSITION, motor)
            return actual == self._exchange(GAP, AP_TARGET_POSITION, motor)

    def reference_status(self, axis: str) -> int:
        """The reference search's state (``RFS 2``): 0 when none is running."""
        return self._exchange(RFS, RFS_STATUS, self._motor(axis))

    def limit_switch_status(self, axis: str) -> tuple[int, int]:
        """``(left, right)``: 1 engaged, 0 clear, -1 each when unread."""
        motor = self._motor(axis)
        try:
            with self._lock:
                left = self._exchange(GAP, AP_LEFT_SWITCH, motor)
                right = self._exchange(GAP, AP_RIGHT_SWITCH, motor)
        except HardwareError as e:
            logger.warning(f'[TMCM-6110 ] limit_switch_status({axis}) failed: {e}')
            return -1, -1
        return int(left != 0), int(right != 0)

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------

    def home(self) -> bool:
        """Home all three axes with LumaView Classic's sequence.

        Z first, to its switch and back up off it to its 0, so the
        objective is down before X or Y moves; then X to its switch; then
        Y and X each to their switch, off it, back onto it slowly and on to
        the index pulse, where the position is set to 0. Every axis's target is then set to 0, so
        target and actual agree, and each axis reads its index position
        in the API. Each phase polls the lid and the abort.

        Returns:
            bool: True once every axis is at its reference.

        Raises:
            MotionInterlockError: the stage has no power, or the lid is
                open (``moved=False``: nothing was started) or was opened
                during the home (``moved=True``; all three axes stopped).
            HardwareError: a phase did not complete in time, a stop ended
                the home, or the board did not answer; naming the phase.
        """
        return self._home(
            (self._home_z, self._home_x_to_switch, self._home_y, self._home_x), ('X', 'Y', 'Z')
        )

    def zhome(self) -> bool:
        """Home Z alone, as ``home`` homes it, with the same checks."""
        return self._home((self._home_z,), ('Z',))

    def _home(self, phases, axes: tuple[str, ...]) -> bool:
        """Run ``phases`` in order, then set the target of each of ``axes`` to 0."""
        self._home_abort.clear()
        with self._lock:
            if self._exchange(GIO, *POWER_INPUT) <= POWER_PRESENT_ABOVE:
                raise MotionInterlockError('stage_unpowered', moved=False, stopped=False)
            if self._lid_open():
                self._refuse_for_lid(moved=False)
        try:
            self._home_send(SIO, *FAN_OUTPUT, 1)
            self._set_switch_polarities()
            try:
                for phase in phases:
                    phase()
            except MotionInterlockError as e:
                if e.moved:
                    raise
                # The exchange path refused a start inside the phases: an
                # earlier phase had already moved an axis.
                raise MotionInterlockError(e.reason, moved=True, stopped=e.stopped) from e
            for axis in axes:
                self._home_send(SAP, AP_TARGET_POSITION, MOTORS[axis], 0)
        except BaseException:
            self._end_reference_searches(raising=False)
            raise
        self._end_reference_searches(raising=True)
        return True

    def _home_send(self, command: int, type_: int, motor: int, value: int = 0) -> int:
        """One command of a home, refused once a stop has begun: the abort
        is read under the lock motor_stop's stop takes, so a home never
        starts an axis after a stop."""
        with self._lock:
            if self._home_abort.is_set():
                raise HardwareError('the home was stopped')
            return self._exchange(command, type_, motor, value)

    def _await(self, phase: str, done) -> None:
        """Poll until ``done()`` is true, reading the lid and the abort each time.

        Raises:
            MotionInterlockError: the lid was opened; all three axes stopped.
            HardwareError: a stop ended the home, or ``phase`` took longer
                than ``HOME_PHASE_TIMEOUT_S``.
        """
        deadline = time.monotonic() + HOME_PHASE_TIMEOUT_S
        while True:
            time.sleep(HOME_POLL_S)
            with self._lock:
                if self._home_abort.is_set():
                    raise HardwareError(f'the home was stopped during {phase}')
                if self._lid_open():
                    self._refuse_for_lid(moved=True)
                if done():
                    return
            if time.monotonic() >= deadline:
                raise HardwareError(f'homing {phase} did not complete in {HOME_PHASE_TIMEOUT_S} s')

    def _end_reference_searches(self, *, raising: bool) -> None:
        """``RFS 1`` to all three axes, on every exit from a home, as Classic's
        did. On a home that failed, the failure is what is reported: a stop
        that fails too is logged beside it."""
        for motor in MOTORS.values():
            try:
                self._exchange(RFS, RFS_STOP, motor)
            except HardwareError as e:
                if raising:
                    raise
                logger.error(f'[TMCM-6110 ] Ending the reference search on motor {motor}: {e}')

    def _set_switch_polarities(self) -> None:
        """The switches read uninverted with their pull-ups on, and no TMCL
        program starts on its own."""
        self._home_send(SGP, *GP_AUTO_START_MODE, 0)
        self._home_send(SGP, *GP_END_SWITCH_POLARITY, 0)
        self._home_send(SIO, *SWITCH_PULLUPS_OUTPUT, 1)

    def _init_axis(self, axis: str, overrides: Mapping[int, int] | None = None) -> None:
        """Write the axis's parameters, with ``overrides`` by TMCL number."""
        params = self.motorconfig.axis_parameters(axis)
        overrides = overrides or {}
        for name, number in INIT_PARAMETERS:
            self._home_send(SAP, number, MOTORS[axis], overrides.get(number, params[name]))

    def _stopped(self, axis: str):
        return lambda: self._exchange(GAP, AP_ACTUAL_VELOCITY, MOTORS[axis]) == 0

    def _home_z(self) -> None:
        """Z to its switch, then up off it, where the position becomes 0:
        a move back down meets the switch a few micrometres above the 0 the
        search sets, so a 0 on the switch would stop Z short of every move
        to 0."""
        search = self.motorconfig.homing('Z')['Switch Search']
        motor = MOTORS['Z']
        self._set_switch_polarities()
        self._init_axis('Z')
        self._home_send(SAP, AP_REFERENCE_SEARCH_MODE, motor, search['Reference Search Mode'])
        self._home_send(SAP, AP_REFERENCE_SEARCH_SPEED, motor, search['Reference Search Speed'])
        self._home_send(RFS, RFS_START, motor)
        self._await('Z to its switch', self._stopped('Z'))
        self._home_send(MVP, MVP_REL, motor, -search['Back-off Microsteps'])
        self._await(
            'Z off its switch',
            lambda: self._exchange(GAP, AP_TARGET_REACHED, motor) == 1,
        )
        self._home_send(SAP, AP_ACTUAL_POSITION, motor, 0)

    def _home_x_to_switch(self) -> None:
        """X to its switch fast, so the Y search starts with X out of the way."""
        pre = self.motorconfig.homing('X')['Switch Pre-move']
        motor = MOTORS['X']
        self._init_axis('X', {AP_MAX_POSITIONING_SPEED: pre['Max Positioning Speed']})
        self._home_send(SAP, AP_REFERENCE_SEARCH_MODE, motor, pre['Reference Search Mode'])
        self._home_send(SAP, AP_REFERENCE_SEARCH_SPEED, motor, pre['Reference Search Speed'])
        self._home_send(SAP, AP_REFERENCE_SWITCH_SPEED, motor, pre['Reference Switch Speed'])
        self._home_send(RFS, RFS_START, motor)
        self._await('X to its switch', self._stopped('X'))

    def _home_y(self) -> None:
        self._set_switch_polarities()
        self._init_axis('Y')
        self._index_search('Y')

    def _home_x(self) -> None:
        self._init_axis('X')
        self._index_search('X')

    def _index_search(self, axis: str) -> None:
        """Onto the right switch, off it, back onto it slowly, then the
        index pulse, where the position becomes 0."""
        index = self.motorconfig.homing(axis)['Index Search']
        params = self.motorconfig.axis_parameters(axis)
        motor = MOTORS[axis]
        fast, slow = index['Approach Speeds']

        def at_switch():
            return self._exchange(GAP, AP_RIGHT_SWITCH, motor) == 1

        if self._home_send(GAP, AP_RIGHT_SWITCH, motor) != 1:
            self._home_send(ROR, 0, motor, fast)
            self._await(f'{axis} onto its switch', at_switch)
        self._home_send(MVP, MVP_REL, motor, -index['Back-off Microsteps'])
        self._await(
            f'{axis} off its switch',
            lambda: self._exchange(GAP, AP_TARGET_REACHED, motor) == 1,
        )
        self._home_send(ROR, 0, motor, slow)
        self._await(f'{axis} back onto its switch', at_switch)
        # The speed is written before every acceleration change: the board
        # runs an axis anomalously slowly otherwise.
        self._home_send(SAP, AP_MAX_POSITIONING_SPEED, motor, params['Max Positioning Speed'])
        self._home_send(SAP, AP_MAX_ACCELERATION, motor, index['Max Acceleration'])
        self._home_send(SAP, AP_REFERENCE_SEARCH_MODE, motor, index['Reference Search Mode'])
        self._home_send(SAP, AP_REFERENCE_SEARCH_SPEED, motor, index['Reference Search Speed'])
        self._home_send(SAP, AP_REFERENCE_SWITCH_SPEED, motor, index['Reference Switch Speed'])
        self._home_send(RFS, RFS_START, motor)
        self._await(
            f'{axis} to its index pulse',
            lambda: self._exchange(RFS, RFS_STATUS, motor) == 0,
        )
        self._home_send(SAP, AP_MAX_POSITIONING_SPEED, motor, params['Max Positioning Speed'])
        self._home_send(SAP, AP_MAX_ACCELERATION, motor, params['Max Acceleration'])
        self._home_send(SAP, AP_ACTUAL_POSITION, motor, 0)

    def thome(self) -> bool:
        raise HardwareError('the TMCM-6110 stage has no turret')

    def has_homed(self) -> bool:
        """False: the board keeps no record of a home, as ``detect_homed_axes``
        answers none. The API's axis state is the one record of which axes
        are homed; a latch here disagreed with it both ways."""
        return False

    def has_turret(self) -> bool:
        return False

    def has_thomed(self) -> bool:
        return False

    def detect_present_axes(self) -> list:
        return list(MOTORS)

    def detect_homed_axes(self) -> list:
        """None: the board keeps no record of a home, so every axis starts unknown."""
        return []

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------

    _ACCELERATION_AXES = ('X', 'Y')
    _ACCELERATION_PARAMETERS = ('acceleration', 'deceleration')

    def _check_acceleration(self, axis: str, parameter: str) -> None:
        if axis not in self._ACCELERATION_AXES:
            raise NotImplementedError(
                f'Support for acceleration limit on axis {axis} not implemented'
            )
        if parameter not in self._ACCELERATION_PARAMETERS:
            raise NotImplementedError(
                f'Support for acceleration limit parameter {parameter} not implemented.'
            )

    def acceleration_limit(self, axis: str, parameter: str) -> int:
        """The axis's full acceleration in TMCL units. The TMC429's ramps are
        symmetric, so acceleration and deceleration are the one value."""
        self._check_acceleration(axis, parameter)
        return self.motorconfig.axis_parameters(axis)['Max Acceleration']

    def acceleration_limits(self) -> dict:
        return {
            axis: {p: self.acceleration_limit(axis, p) for p in self._ACCELERATION_PARAMETERS}
            for axis in self._ACCELERATION_AXES
        }

    def set_acceleration_limit(self, axis: str, parameter: str, val_pct: int) -> None:
        """Set the axis's acceleration (and so its deceleration) to
        ``val_pct`` of full.

        The speed is written again first: the board runs an axis
        anomalously slowly when its acceleration changes without it.

        The range is the API's: it refuses a value outside it before any
        board is commanded; the section's full value is checked at load to
        be large enough that no admitted percentage rounds to 0.

        Raises:
            HardwareError: the board did not take it.
        """
        self._check_acceleration(axis, parameter)
        params = self.motorconfig.axis_parameters(axis)
        motor = MOTORS[axis]
        with self._lock:
            self._exchange(SAP, AP_MAX_POSITIONING_SPEED, motor, params['Max Positioning Speed'])
            self._exchange(
                SAP, AP_MAX_ACCELERATION, motor, round(params['Max Acceleration'] * val_pct / 100)
            )

    def set_acceleration_limits(self, val_pct: int) -> None:
        for axis in self._ACCELERATION_AXES:
            self.set_acceleration_limit(axis, 'acceleration', val_pct)

    def set_precision_mode(self, axis: str, enabled: bool) -> None:
        """No precision mode on the TMCM-6110: nothing to set."""

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def _ustep2um(self, axis: str, ustep: int) -> float:
        return ustep * 1000 / self.motorconfig.usteps_per_mm(axis)

    def _um2ustep(self, axis: str, um: float) -> int:
        return round(self.motorconfig.usteps_per_mm(axis) * um / 1000)

    def z_ustep2um(self, ustep: int) -> float:
        return self._ustep2um('Z', ustep)

    def z_um2ustep(self, um: float) -> int:
        return self._um2ustep('Z', um)

    def xy_ustep2um(self, ustep: int) -> float:
        return self._ustep2um('X', ustep)

    def xy_um2ustep(self, um: float) -> int:
        return self._um2ustep('X', um)

    def t_ustep2deg(self, ustep: int) -> float:
        raise RuntimeError('the TMCM-6110 stage has no turret to convert for')

    def t_ustep2pos(self, ustep: int) -> int:
        raise RuntimeError('the TMCM-6110 stage has no turret to convert for')

    def t_deg2ustep(self, degrees: float) -> int:
        raise RuntimeError('the TMCM-6110 stage has no turret to convert for')

    def t_pos2ustep(self, position: int) -> int:
        raise RuntimeError('the TMCM-6110 stage has no turret to convert for')

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_microscope_model(self) -> str:
        """'LS720': the only Lumascope with this board."""
        return 'LS720'

    def get_serial_number(self) -> None:
        """None: the board carries no Etaluma serial number."""
        return None

    def get_current_firmware(self) -> str | None:
        return self.firmware_version

    def fullinfo(self) -> dict:
        homed = self.has_homed()
        return {
            'model': self.get_microscope_model(),
            'serial_number': None,
            'firmware_version': self.firmware_version,
            'x_homed': homed,
            'x_present': True,
            'y_homed': homed,
            'y_present': True,
            'z_homed': homed,
            'z_present': True,
            't_homed': False,
            't_present': False,
        }

    def get_axes_config(self) -> Mapping:
        return self.axes_config

    def get_axis_limits(self, axis: str) -> Mapping[str, float] | None:
        if axis not in self.axes_config:
            raise HardwareError(f'Unsupported axis ({axis})')
        return self.axes_config[axis]['limits']

    # ------------------------------------------------------------------
    # What the board has no part in
    # ------------------------------------------------------------------

    def supports_motor_stop(self) -> bool:
        return True

    def supports_fan(self) -> bool:
        return False

    def supports_diagnostics(self) -> bool:
        return False

    def spi_read(self, axis: str, addr: int) -> str | None:
        raise RuntimeError('the TMCM-6110 gives no SPI register access')

    def spi_write(self, axis: str, addr: int, payload: int | str) -> str:
        raise RuntimeError('the TMCM-6110 gives no SPI register access')

    def exchange_command(
        self, command: str, response_numlines: int = 1, timeout: float | None = None
    ) -> None:
        """None: the board takes no text commands, so no text has a reply."""
        logger.warning(f'[TMCM-6110 ] Text command {command!r} not sent: the board speaks TMCL')
        return None
