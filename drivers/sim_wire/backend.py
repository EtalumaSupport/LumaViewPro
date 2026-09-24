# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated serial backend: the boards a simulated scope has, at the wire.

Handed to a board driver in place of pyserial, it answers discovery with
the simulated scope's boards and opens each one as an `EmulatedPort`
running the real firmware. The driver above it is the production driver,
unchanged; only the port is simulated.

Which axes the scope has is decided by the caller from the scope model and
passed in. A scope with no motor axes has no motor board at all, so this
backend offers none and a driver looking for one finds nothing, as on a
manual scope.
"""

import json
import pathlib
import platform
import sys
from dataclasses import dataclass

from serial.serialutil import SerialException
from serial.tools.list_ports_common import ListPortInfo

from drivers.sim_wire.port import BoardImage, EmulatedPort

_PACKAGE = pathlib.Path(__file__).resolve().parent
_REPO = _PACKAGE.parent.parent

MOTOR_VID, MOTOR_PID = 0x2E8A, 0x0005
MOTOR_DEVICE = 'simwire:motor'

TIMINGS = ('instant', 'realistic')
AXES = ('X', 'Y', 'Z', 'T')

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


@dataclass(frozen=True)
class MotorBoardSpec:
    """The simulated motor board: which scope, which axes, which firmware,
    which clock."""

    model: str
    axes: frozenset[str]
    dialect: str = '3.0'
    timing: str = 'instant'

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

    def motorconfig(self) -> dict:
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
        module_path = [str(_PACKAGE / 'mp')]
        if self.timing == 'instant':
            module_path.insert(0, str(_PACKAGE / 'mp' / 'instant'))
        return BoardImage(
            runtime=str(runtime_path(self.dialect)),
            firmware_mpy=str(_PACKAGE / 'firmware' / f'motor-{self.dialect}.mpy'),
            files=files,
            module_path=tuple(module_path),
            label=f'[sim motor {self.model} fw {self.dialect} {self.timing}]',
        )


class SimWireBackend:
    """Discovery and open for a simulated scope's boards."""

    def __init__(self, motor: MotorBoardSpec | None):
        self._motor = motor

    def comports(self) -> list[ListPortInfo]:
        if self._motor is None:
            return []
        info = ListPortInfo(MOTOR_DEVICE)
        info.vid, info.pid = MOTOR_VID, MOTOR_PID
        info.description = 'Simulated EL-0940 motor board'
        return [info]

    def open(self, **kwargs) -> EmulatedPort:
        port = kwargs.get('port')
        if self._motor is None or port != MOTOR_DEVICE:
            raise SerialException(f'no simulated board at {port!r}')
        return EmulatedPort(self._motor.image(), **kwargs)
