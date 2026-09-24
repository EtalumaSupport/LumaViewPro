# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The two TMC5072 motor drivers of an EL-0940, as the firmware sees them over
# SPI. Runs INSIDE the MicroPython child behind the `machine.SPI` stub, and
# imports under CPython too, so the kinematics are unit-tested in the suite.
# Pure Python: no dataclasses, no typing imports, nothing host-only.
#
# What is modelled: the register file each motor exposes, pipelined reads
# (a datagram answers the register the previous datagram addressed), the
# ramp generator (a trapezoid from AMAX / VMAX / DMAX, which is what the INI
# files configure: A1 and V1 are zero), the reference switch inputs with
# SW_MODE's enable / polarity / swap bits, hard stops at a switch,
# RAMP_STAT's status bits, and DRV_STATUS's standstill and open-load bits.
# Not modelled: StallGuard, chopper and current, encoders, latching.
#
# Hardware faults a test can switch on per motor (`FAULTS`): a reference
# switch that never trips, a motor that stalls, and a motor that is not
# connected. A stalled or disconnected motor leaves the chip's ramp running
# as an open-loop driver does: XACTUAL reaches the target while the stage,
# and so the switch, stays where it was.
#
# Two frames per motor. The PHYSICAL position (`phys`, microsteps from the
# reference flag's edge) is where the ramp has driven the motor and never
# changes on a register write; XACTUAL is physical plus an offset, and
# writing XACTUAL, as the firmware does in hold mode during homing, moves
# the offset, not the stage. The switches see the stage, which is `slip`
# behind the physical position; the two differ only once a stalled or
# disconnected motor has been driven.
#
# Positions are integers, in 1/_SUB of a microstep, as the chip's own are
# integers: the MicroPython this runs in computes floats in single precision,
# as the board's does, and a float position there cannot hold a small ramp
# step at a large position, nor the read-ahead past a switch edge.
# Velocities and accelerations stay floats; their error is relative.

try:
    from collections.abc import Callable
except ImportError:
    # MicroPython has no collections.abc and never evaluates annotations, so
    # the name only has to exist for the annotations to parse.
    Callable = None

# Per-motor register offsets (M1 at 0x20, M2 at 0x40).
RAMPMODE = 0x00
XACTUAL = 0x01
VACTUAL = 0x02
VSTART = 0x03
A1 = 0x04
V1 = 0x05
AMAX = 0x06
VMAX = 0x07
DMAX = 0x08
D1 = 0x0A
VSTOP = 0x0B
TZEROWAIT = 0x0C
XTARGET = 0x0D
IHOLD_IRUN = 0x10
SW_MODE = 0x14
RAMP_STAT = 0x15
XLATCH = 0x16

MOTOR_BASE = (0x20, 0x40)
MOTOR_SPAN = 0x20
DRV_STATUS = (0x6F, 0x7F)

# RAMPMODE values.
MODE_POSITION = 0
MODE_VELOCITY_POS = 1
MODE_VELOCITY_NEG = 2
MODE_HOLD = 3

# SW_MODE bits.
STOP_L_ENABLE = 1 << 0
STOP_R_ENABLE = 1 << 1
POL_STOP_L = 1 << 2
POL_STOP_R = 1 << 3
SWAP_LR = 1 << 4

# RAMP_STAT bits.
STATUS_STOP_L = 1 << 0
STATUS_STOP_R = 1 << 1
EVENT_STOP_L = 1 << 4
EVENT_STOP_R = 1 << 5
EVENT_POS_REACHED = 1 << 7
VELOCITY_REACHED = 1 << 8
POSITION_REACHED = 1 << 9
VZERO = 1 << 10

# DRV_STATUS bits.
DRV_OLA = 1 << 29
DRV_OLB = 1 << 30
DRV_STST = 1 << 31

# The turret's flag is a slot about this wide (the firmware backs off 12000
# microsteps to clear it).
TURRET_FLAG_WIDTH = 10000

AXES = ('X', 'Y', 'Z', 'T')

SWITCH_NEVER_TRIPS = 'switch_never_trips'
STALL = 'stall'
ABSENT = 'absent'
FAULTS = (SWITCH_NEVER_TRIPS, STALL, ABSENT)

# The integrator's largest step: the firmware polls every millisecond during
# a wait, so a longer gap only happens when nobody is asking.
_MAX_STEP_US = 1000

# The fraction of a microstep positions are kept in.
_SUB = 1 << 16

# How far ahead of the motor a switch is read before it moves: just past
# the position, so a motor resting on a flag edge sees the state it is
# about to enter and a motor approaching an edge is not stopped short of it.
_AHEAD = 1

_TWO_32 = 1 << 32
_TWO_31 = 1 << 31


def _signed32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - _TWO_32 if v >= _TWO_31 else v


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _usteps(pos: int) -> int:
    """A position in whole microsteps, rounded half up."""
    return (pos + _SUB // 2) // _SUB


def _sub(distance: float) -> int:
    """A distance travelled, in microsteps, as a position increment."""
    return int(distance * _SUB)


class Motor:
    """One motor of a chip: registers, physical position, ramp, switch."""

    def __init__(
        self,
        present: bool,
        usteps_per_mm: float,
        travel_mm: float,
        flag_active_low: bool,
        start_usteps: float,
        wrap_usteps: int = 0,
    ) -> None:
        self.present = present
        self.usteps_per_mm = usteps_per_mm
        self.travel_span = int(travel_mm * usteps_per_mm) * _SUB
        self.flag_active_low = flag_active_low
        # A rotary axis wraps and its flag is a slot; a linear axis has its
        # flag over everything left of the reference edge at physical 0.
        self.wrap = wrap_usteps * _SUB
        # `pos` is the position the ramp has driven the motor to. The stage
        # is `slip` behind it: nonzero only once a stalled or disconnected
        # motor has been driven without the stage following.
        self.pos = int(start_usteps) * _SUB
        self.slip = 0
        self.faults = set()
        self.offset = -_usteps(self.pos)
        self.velocity = 0.0  # microsteps per second, signed
        self.fclk = 16_000_000.0
        self.regs = {}
        self.events = 0
        self.last_us = None

    # --- reference switch inputs ---------------------------------------

    def set_fault(self, name: str, on: bool) -> None:
        if name not in FAULTS:
            raise ValueError(f'unknown fault {repr(name)}; faults are {FAULTS}')
        if on:
            self.faults.add(name)
        else:
            self.faults.discard(name)

    @property
    def phys(self) -> float:
        """The physical position in microsteps."""
        return self.pos / _SUB

    def stage_follows(self) -> bool:
        return STALL not in self.faults and ABSENT not in self.faults

    def _move_to(self, pos: int) -> None:
        if not self.stage_follows():
            self.slip += pos - self.pos
        self.pos = pos

    def on_flag(self, pos: int) -> bool:
        if SWITCH_NEVER_TRIPS in self.faults:
            return False
        pos -= self.slip
        if self.wrap:
            return 0 <= (pos % self.wrap) < TURRET_FLAG_WIDTH * _SUB
        return pos <= 0

    def refl_level(self, pos: int | None = None) -> int:
        if pos is None:
            pos = self.pos
        return 1 if self.on_flag(pos) != self.flag_active_low else 0

    def refr_level(self) -> int:
        # No switch on the right input: it rests at the off-flag level (a
        # pull-up on the active-low boards, ground on the active-high ones).
        return 1 if self.flag_active_low else 0

    def stop_status(self, pos: int | None = None) -> tuple[int, int]:
        """(status_stop_l, status_stop_r) as SW_MODE makes the chip read them."""
        sw = self.regs.get(SW_MODE, 0)
        left, right = self.refl_level(pos), self.refr_level()
        if sw & SWAP_LR:
            left, right = right, left
        return (left ^ (1 if sw & POL_STOP_L else 0), right ^ (1 if sw & POL_STOP_R else 0))

    def blocked(self, direction: int, pos: int | None = None) -> bool:
        """Whether a switch stop holds the motor from moving in `direction`
        from `pos` (the current position by default). The switch is read
        just ahead of the position: a motor sitting exactly on a flag edge
        with the polarity flipped stops the moment it leaves the edge, as the
        real one does, instead of running to its target."""
        sw = self.regs.get(SW_MODE, 0)
        if pos is None:
            pos = self.pos + direction * _AHEAD
        stop_l, stop_r = self.stop_status(pos)
        if direction < 0:
            return bool(sw & STOP_L_ENABLE) and stop_l == 1
        if direction > 0:
            return bool(sw & STOP_R_ENABLE) and stop_r == 1
        return False

    def next_edge(self, direction: int) -> int | None:
        """The nearest position ahead where the flag state changes, or None
        when the stage cannot reach one: it is not following the motor, or
        the switch never changes."""
        if not self.stage_follows() or SWITCH_NEVER_TRIPS in self.faults:
            return None
        stage = self.pos - self.slip
        if self.wrap:
            w = self.wrap
            base = (stage // w) * w
            edges = [base + k * w + e for k in (-1, 0, 1) for e in (0, TURRET_FLAG_WIDTH * _SUB)]
        else:
            edges = [0]
        edges = [e + self.slip for e in edges]
        ahead = [e for e in edges if (e - self.pos) * direction > 0]
        if not ahead:
            return None
        return min(ahead) if direction > 0 else max(ahead)

    # --- registers ------------------------------------------------------

    def actual(self) -> int:
        return _signed32(_usteps(self.pos) + self.offset)

    def target(self) -> int:
        return _signed32(self.regs.get(XTARGET, 0))

    def write(self, reg: int, value: int) -> None:
        value &= 0xFFFFFFFF
        if reg == XACTUAL:
            self.offset = _signed32(value) - _usteps(self.pos)
        elif reg == VACTUAL:
            return
        self.regs[reg] = value

    def read(self, reg: int) -> int:
        if reg == XACTUAL:
            return self.actual() & 0xFFFFFFFF
        if reg == VACTUAL:
            return round(self.velocity * (1 << 24) / self.fclk) & 0xFFFFFF
        if reg == RAMP_STAT:
            value = self.ramp_stat() | self.events
            self.events = 0
            return value
        return self.regs.get(reg, 0)

    def ramp_stat(self) -> int:
        stop_l, stop_r = self.stop_status()
        value = stop_l * STATUS_STOP_L | stop_r * STATUS_STOP_R
        if self.velocity == 0.0:
            value |= VZERO
        if self.regs.get(RAMPMODE, 0) == MODE_POSITION and self.actual() == self.target():
            value |= POSITION_REACHED
        if self.velocity != 0.0 and abs(self.velocity) >= self.vmax_usteps_s():
            value |= VELOCITY_REACHED
        return value

    def drv_status(self) -> int:
        value = 0
        if not self.present or ABSENT in self.faults:
            value |= DRV_OLA | DRV_OLB
        if self.velocity == 0.0:
            value |= DRV_STST
        ihold_irun = self.regs.get(IHOLD_IRUN, 0)
        current = ((ihold_irun >> 8) if self.velocity else ihold_irun) & 0x1F
        return value | (current << 16)

    # --- the ramp -------------------------------------------------------

    def vmax_usteps_s(self) -> float:
        return self.regs.get(VMAX, 0) * self.fclk / (1 << 24)

    def _accel(self, reg: int) -> float:
        return self.regs.get(reg, 0) * self.fclk * self.fclk / (1 << 41)

    def stop_point(self) -> tuple[int | None, int]:
        """Where the current ramp is heading, in the physical frame, and the
        direction of travel; (None, 0) when the motor has no reason to move."""
        mode = self.regs.get(RAMPMODE, 0)
        if mode == MODE_HOLD:
            return None, 0
        if mode == MODE_POSITION:
            goal = (self.target() - self.offset) * _SUB
            direction = _sign(goal - self.pos)
            if direction == 0:
                return None, 0
            return goal, direction
        direction = 1 if mode == MODE_VELOCITY_POS else -1
        return None, direction

    def advance(self, now_us: int, ticks_diff: 'Callable[[int, int], int]', instant: bool) -> None:
        """Bring the motor up to `now_us`."""
        if self.last_us is None:
            self.last_us = now_us
            if instant:
                self._jump()
            return
        elapsed = ticks_diff(now_us, self.last_us)
        self.last_us = now_us
        if instant:
            self._jump()
            return
        while elapsed > 0:
            step = elapsed if elapsed < _MAX_STEP_US else _MAX_STEP_US
            self._step(step / 1e6)
            elapsed -= step

    def _jump(self) -> None:
        """Instant mode: the move completes now, unless a switch catches it."""
        self.velocity = 0.0
        goal, direction = self.stop_point()
        if direction == 0:
            return
        if self.blocked(direction):
            if not self.blocked(direction, self.pos):
                self._switch_stop(direction)
            return
        if goal is None:
            # Velocity mode has no destination: park at the next edge, or a
            # travel's length away when there is none.
            edge = self.next_edge(direction)
            goal = edge if edge is not None else self.pos + direction * self.travel_span
        while True:
            edge = self.next_edge(direction)
            if edge is not None and (edge - self.pos) * direction < (goal - self.pos) * direction:
                self._move_to(edge)
                if self.blocked(direction, edge + direction * _AHEAD):
                    self._switch_stop(direction, edge)
                    return
                continue
            self._move_to(goal)
            self._reached()
            return

    def _step(self, dt: float) -> None:
        goal, direction = self.stop_point()
        if direction == 0:
            self._decelerate_to_rest(dt)
            return
        if self.blocked(direction) and self.velocity * direction >= 0:
            if self.velocity != 0.0 or not self.blocked(direction, self.pos):
                self._switch_stop(direction)
            self.velocity = 0.0
            return
        amax, dmax, vmax = self._accel(AMAX), self._accel(DMAX), self.vmax_usteps_s()
        if dmax <= 0.0:
            dmax = amax
        speed = abs(self.velocity)
        if self.velocity * direction < 0:
            # Reversing: bleed the old direction's speed first.
            speed = max(0.0, speed - dmax * dt)
            self.velocity = -direction * speed if speed else 0.0
            return
        edge = self.next_edge(direction)
        remaining = None if goal is None else abs(goal - self.pos)
        if remaining is not None and edge is not None and abs(edge - self.pos) < remaining:
            # A switch may intervene before the goal: do not brake for the goal.
            remaining = None
        if remaining is not None and speed * speed / (2.0 * dmax) >= remaining / _SUB:
            speed = max(0.0, speed - dmax * dt)
        else:
            speed = min(vmax, speed + amax * dt)
        travel = _sub(speed * dt)
        if edge is not None and abs(edge - self.pos) <= travel:
            travel -= abs(edge - self.pos)
            self._move_to(edge)
            if self.blocked(direction, edge + direction * _AHEAD):
                self._switch_stop(direction, edge)
                return
        if goal is not None and abs(goal - self.pos) <= travel:
            self._move_to(goal)
            self._reached()
            return
        self._move_to(self.pos + direction * travel)
        self.velocity = direction * speed

    def _decelerate_to_rest(self, dt: float) -> None:
        if self.velocity == 0.0:
            return
        dmax = self._accel(DMAX) or self._accel(AMAX)
        speed = max(0.0, abs(self.velocity) - dmax * dt)
        self._move_to(self.pos + _sign(self.velocity) * _sub(speed * dt))
        self.velocity = _sign(self.velocity) * speed if speed else 0.0

    def _switch_stop(self, direction: int, edge: int | None = None) -> None:
        """A hard stop at a switch. The motor comes to rest where the switch
        reads active at rest: on the edge when the edge itself is inside the
        stopping region, else one microstep past it, as a real motor runs a
        fraction of a step into its switch."""
        self.velocity = 0.0
        if edge is None:
            edge = self.pos
        self._move_to(edge if self.blocked(direction, edge) else edge + direction * _SUB)
        self.events |= EVENT_STOP_L if direction < 0 else EVENT_STOP_R

    def _reached(self) -> None:
        self.velocity = 0.0
        if self.regs.get(RAMPMODE, 0) == MODE_POSITION:
            self.events |= EVENT_POS_REACHED


class Chip:
    """A TMC5072: two motors, one SPI shift register."""

    def __init__(self, motors: tuple, fclk: float) -> None:
        self.motors = motors  # (M1, M2)
        for motor in motors:
            motor.fclk = fclk
        self.global_regs = {}
        self.last_addr = 0

    def _locate(self, addr: int) -> tuple:
        for index, base in enumerate(MOTOR_BASE):
            if base <= addr < base + MOTOR_SPAN:
                return self.motors[index], addr - base
        if addr in DRV_STATUS:
            return self.motors[DRV_STATUS.index(addr)], 'drv'
        return None, None

    def read(self, addr: int) -> int:
        motor, reg = self._locate(addr)
        if motor is None:
            return self.global_regs.get(addr, 0)
        if reg == 'drv':
            return motor.drv_status()
        return motor.read(reg)

    def write(self, addr: int, value: int) -> None:
        motor, reg = self._locate(addr)
        if motor is None:
            self.global_regs[addr] = value
        elif reg != 'drv':
            motor.write(reg, value)

    def datagram(self, buf: bytes) -> bytes:
        """One 40-bit transfer: answer the previous datagram's read, apply
        this one's write, and queue this one's address."""
        out = self.read(self.last_addr)
        addr = buf[0] & 0x7F
        if buf[0] & 0x80:
            self.write(addr, (buf[1] << 24) | (buf[2] << 16) | (buf[3] << 8) | buf[4])
        self.last_addr = addr
        return bytes((0, (out >> 24) & 0xFF, (out >> 16) & 0xFF, (out >> 8) & 0xFF, out & 0xFF))


class Board:
    """The XY and ZT chips of an EL-0940, built from the unit config the
    firmware itself reads plus the simulator's own settings."""

    def __init__(
        self,
        motorconfig: dict,
        sim: dict,
        ticks_us: 'Callable[[], int]',
        ticks_diff: 'Callable[[int, int], int]',
    ) -> None:
        self.ticks_us = ticks_us
        self.ticks_diff = ticks_diff
        self.instant = sim.get('timing', 'instant') == 'instant'
        fclk = float(sim.get('fclk_hz', 16_000_000))
        present = motorconfig['Axis Present']
        usteps = motorconfig['Axis Microsteps per mm / Objective']
        travel = motorconfig['Axis Travel Limit']
        invert = motorconfig.get('Axis Flag Invert', {})
        slots = motorconfig.get('TurretPosition', {})
        start = sim.get('start_usteps', {})
        motors = {}
        for axis in AXES:
            wrap = 0
            if axis == 'T' and slots:
                wrap = int(usteps['T']) * len(slots)
            motors[axis] = Motor(
                present=bool(int(present.get(axis, 0))),
                usteps_per_mm=float(usteps[axis]),
                travel_mm=float(travel[axis]),
                flag_active_low=bool(int(invert.get(axis, 0))),
                start_usteps=float(start.get(axis, 0)),
                wrap_usteps=wrap,
            )
        self.motors = motors
        self.chips = {
            'XY': Chip((motors['X'], motors['Y']), fclk),
            'ZT': Chip((motors['T'], motors['Z']), fclk),
        }

    def set_fault(self, axis: str, name: str, on: bool) -> None:
        if axis not in self.motors:
            raise ValueError(f'unknown axis {repr(axis)}')
        self.motors[axis].set_fault(name, on)

    def register_name(self, chip_name: str, addr: int) -> tuple:
        """(axis, register offset) of an address on a chip, or (None, the
        address) for a register no single motor owns."""
        motor, reg = self.chips[chip_name]._locate(addr)
        if motor is None or reg == 'drv':
            return None, addr
        for axis, candidate in self.motors.items():
            if candidate is motor:
                return axis, reg
        return None, addr

    def advance(self) -> None:
        now = self.ticks_us()
        for motor in self.motors.values():
            motor.advance(now, self.ticks_diff, self.instant)

    def datagram(self, chip_name: str, buf: bytes) -> bytes:
        self.advance()
        return self.chips[chip_name].datagram(buf)
