# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated serial backend: the boards a simulated scope has, at the wire.

Handed to a board driver in place of pyserial, it answers discovery with
the simulated scope's boards and opens an `EmulatedPort` to each. The
boards themselves (`EmulatedBoard`, the real firmware running) belong to
the backend and outlive every port opened to them, as a powered board
outlives its USB connection. The driver above it is the production
driver, unchanged; only the hardware is simulated.

Which boards the scope has, and which axes, is decided by the caller from
the scope model and passed in. A scope with no motor axes has no motor
board at all, so this backend offers none and a driver looking for one
finds nothing, as on a manual scope; a scope whose LEDs are not on an LED
board of their own has no LED board here either.
"""

import json
import pathlib
import platform
import sys
from dataclasses import dataclass

from serial.serialutil import SerialException
from serial.tools.list_ports_common import ListPortInfo

from drivers.sim_wire.mp.tmc5072 import AXES, FAULTS
from drivers.sim_wire.port import BoardImage, EmulatedBoard, EmulatedPort

_PACKAGE = pathlib.Path(__file__).resolve().parent
_REPO = _PACKAGE.parent.parent

MOTOR_VID, MOTOR_PID = 0x2E8A, 0x0005
MOTOR_DEVICE = 'simwire:motor'
LED_VID, LED_PID = 0x0424, 0x704C
LED_DEVICE = 'simwire:led'

TIMINGS = ('instant', 'realistic')

# The clock the TMC5072s run at, which sets the ramp's velocity and
# acceleration units. At this clock, with the field INI's registers, the
# model reproduces the Stage 0 bench moves to within the API's own poll
# overhead (X 96 mm: 2735 ms modelled, 2947 ms by API wait with a 220 ms
# poll plateau). A fit from the shipped INI's steeper acceleration landed
# at 14.5 MHz and was wrong for that reason.
FCLK_HZ = 16_000_000

# Where each axis physically sits when the simulated board powers up, in
# microsteps from its reference flag's edge: mid-travel, off the flag, so a
# home finds the flag by moving as a real stage does.
START_USTEPS = {'X': 600_000, 'Y': 400_000, 'Z': 600_000, 'T': 150_000}

# A complete unit config from the board bring-up template, built by the
# Firmware repo's tools/build_sim_firmware.py. The firmware reads all of it
# at boot; the simulator sets the model and the axes.
_MOTORCONFIG_BASE = _PACKAGE / 'firmware' / 'motorconfig-base.json'


def _runtime_tags() -> dict[str, str]:
    """Firmware dialect -> the MicroPython tag its boards run. The pin file is
    the one place this lives; the runtime build reads it too."""
    tags = {}
    for line in (_PACKAGE / 'runtime' / 'MICROPYTHON_PIN').read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith('#'):
            tags[fields[0]] = fields[1]
    return tags


RUNTIME_TAGS = _runtime_tags()
DIALECTS = tuple(RUNTIME_TAGS)
# The firmware a simulated scope's board runs unless a caller names another:
# the one the shipped boards run.
DEFAULT_DIALECT = 'field'


def runtime_platform() -> str | None:
    """The runtime directory this machine's MicroPython builds live in, or
    None where no runtime is built for it. The platforms are the ones the
    runtime build script produces; a machine outside them can still run
    the fast tier, so this answers rather than raises."""
    if sys.platform == 'darwin':
        return 'darwin'
    if sys.platform.startswith('linux') and platform.machine() == 'x86_64':
        return 'linux-x86_64'
    return None


def runtime_path(dialect: str) -> pathlib.Path:
    """The MicroPython runtime a dialect runs on, for this machine, or a
    refusal naming why."""
    platform_tag = runtime_platform()
    if platform_tag is None:
        raise SerialException(
            f'no simulator runtime for {sys.platform}/{platform.machine()}; '
            'the firmware-backed simulator runs on macOS and Linux x86_64'
        )
    path = _PACKAGE / 'runtime' / platform_tag / f'micropython-{RUNTIME_TAGS[dialect]}'
    if not path.exists():
        raise SerialException(f'{path} missing: run scripts/build_sim_runtime.sh')
    return path


def runtime_missing(dialect: str) -> str | None:
    """Why this machine cannot run a dialect's runtime, or None when it
    can. The Linux runtime is built where it runs rather than committed,
    so a supported platform can still be without one."""
    try:
        runtime_path(dialect)
    except SerialException as ex:
        return str(ex)
    return None


def _module_path(timing: str) -> tuple[str, ...]:
    """Where a board's firmware finds the simulated hardware modules, which
    shadow the runtime's own; in instant timing, a clock that never waits
    shadows `time` ahead of them."""
    module_path = (str(_PACKAGE / 'mp'),)
    if timing == 'instant':
        module_path = (str(_PACKAGE / 'mp' / 'instant'), *module_path)
    return module_path


@dataclass(frozen=True)
class MotorBoardSpec:
    """The simulated motor board: which scope, which axes, which firmware,
    which clock."""

    model: str
    axes: frozenset[str]
    dialect: str = DEFAULT_DIALECT
    timing: str = 'instant'
    # Whether the board reports every register write (`EmulatedPort.take_writes`).
    # Off unless a test reads them: nothing else does, and unread writes fill
    # the port's bounded store.
    oracle: bool = False
    fclk_hz: float = FCLK_HZ
    start_usteps: tuple[tuple[str, int], ...] = tuple(START_USTEPS.items())
    # A real unit's own motorconfig.json, as its CONFIG command answers it.
    # None boots the bring-up template with the model and axes set on it; a
    # unit's config is booted exactly as it is, so a simulated board has that
    # unit's scale, offsets and turret positions. The register tables it names
    # still come from the dialect's own set, so a table the set lacks fails
    # the image build by name.
    unit_config: dict | None = None

    def __post_init__(self):
        if not self.axes:
            raise ValueError(f'{self.model}: a motor board with no axes is not a motor board')
        unknown = set(self.axes) - set(AXES)
        if unknown:
            raise ValueError(f'{self.model}: unknown axes {sorted(unknown)}')
        if self.dialect not in DIALECTS:
            raise ValueError(f'firmware dialect {self.dialect!r} is not one of {DIALECTS}')
        if self.timing not in TIMINGS:
            raise ValueError(f'timing mode {self.timing!r} is not one of {TIMINGS}')
        if self.unit_config is not None:
            unit_model = self.unit_config['Microscope']
            unit_axes = {
                axis for axis, present in self.unit_config['Axis Present'].items() if present
            }
            if (unit_model, unit_axes) != (self.model, set(self.axes)):
                raise ValueError(
                    f'the unit config is a {unit_model} with axes {sorted(unit_axes)}, '
                    f'not a {self.model} with axes {sorted(self.axes)}'
                )

    def motorconfig(self) -> dict:
        if self.unit_config is not None:
            return json.loads(json.dumps(self.unit_config))
        config = json.loads(_MOTORCONFIG_BASE.read_text())
        config['Microscope'] = self.model
        config['Axis Present'] = {axis: int(axis in self.axes) for axis in AXES}
        return config

    def image(self) -> BoardImage:
        config = self.motorconfig()
        # The register tables each dialect's boards carry: a 3.0 board has the
        # ones LumaViewPro ships; a field board has its own, because the field
        # parser reads the comments in the current ones as registers and fails.
        ini_dir = _REPO / 'data' / 'firmware'
        if self.dialect == 'field':
            ini_dir = _PACKAGE / 'firmware' / 'field-ini'
        files = {name: (ini_dir / name).read_bytes() for name in config['IniFiles'].values()}
        files['motorconfig.json'] = json.dumps(config).encode()
        files['sim_chip.json'] = json.dumps(
            {
                'timing': self.timing,
                'fclk_hz': self.fclk_hz,
                'start_usteps': dict(self.start_usteps),
                'oracle': self.oracle,
            }
        ).encode()
        return BoardImage(
            runtime=str(runtime_path(self.dialect)),
            firmware_mpy=str(_PACKAGE / 'firmware' / f'motor-{self.dialect}.mpy'),
            files=files,
            module_path=_module_path(self.timing),
            fault_targets=AXES,
            fault_names=FAULTS,
            label=f'[sim motor {self.model} fw {self.dialect} {self.timing}]',
            oracle=self.oracle,
        )


# The firmware the LED boards run: only the field one exists in any unit.
LED_DIALECT = 'field'


@dataclass(frozen=True)
class LedBoardSpec:
    """The simulated LED board: which scope, which clock."""

    model: str
    timing: str = 'instant'

    def __post_init__(self):
        if self.timing not in TIMINGS:
            raise ValueError(f'timing mode {self.timing!r} is not one of {TIMINGS}')

    def image(self) -> BoardImage:
        return BoardImage(
            runtime=str(runtime_path(LED_DIALECT)),
            firmware_mpy=str(_PACKAGE / 'firmware' / f'led-{LED_DIALECT}.mpy'),
            files={},
            module_path=_module_path(self.timing),
            # The firmware detects no fault in its DAC, so there is none to
            # inject: a fault no firmware reaction shows tests nothing.
            fault_targets=(),
            fault_names=(),
            label=f'[sim led {self.model} fw {LED_DIALECT} {self.timing}]',
        )


class SimWireBackend:
    """Discovery and open for a simulated scope's boards."""

    def __init__(self, motor: MotorBoardSpec | None, led: LedBoardSpec | None = None):
        # The simulated boards, which a test reaches for faults, the oracle
        # and the USB link. Each is its own firmware process behind its own
        # port, as each is its own USB device on the scope; its firmware
        # starts with the first port opened to it.
        self.motor_board = None if motor is None else EmulatedBoard(motor.image())
        self.led_board = None if led is None else EmulatedBoard(led.image())

    def _boards(self) -> tuple[tuple[str, int, int, str, EmulatedBoard | None], ...]:
        """(device, VID, PID, description, board) for each board a scope can have."""
        return (
            (MOTOR_DEVICE, MOTOR_VID, MOTOR_PID, 'Simulated EL-0940 motor board', self.motor_board),
            (LED_DEVICE, LED_VID, LED_PID, 'Simulated EL-0940 LED board', self.led_board),
        )

    def comports(self) -> list[ListPortInfo]:
        ports = []
        for device, vid, pid, description, board in self._boards():
            if board is None or not board.plugged:
                continue
            info = ListPortInfo(device)
            info.vid, info.pid = vid, pid
            info.description = description
            ports.append(info)
        return ports

    def open(self, **kwargs) -> EmulatedPort:
        port = kwargs.get('port')
        for device, _vid, _pid, _description, board in self._boards():
            if board is not None and port == device:
                return EmulatedPort(board, **kwargs)
        raise SerialException(f'no simulated board at {port!r}')
