# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The DAC80508 of the EL-0940's LED board (U50), as the field LED firmware
# sees it over SPI. Runs INSIDE the MicroPython child behind the `machine.SPI`
# stub, and imports under CPython too, so it is unit-tested in the suite.
# Pure Python: no dataclasses, no typing imports, nothing host-only.
#
# A transfer is one 24-bit frame, [address, MSB, LSB], written with chip
# select held low; the address byte's top bit set would make it a read. What
# is modelled is the register file the firmware writes: CONFIG's per-channel
# power-down bits, GAIN, TRIGGER's soft reset, and the eight channel codes.
# The firmware never reads the DAC, so a read frame is refused and MISO
# answers zeros. The firmware detects no DAC fault either, so the model has
# none to inject.
#
# What each channel is driving is the model's state: whether its enable
# switch is closed (the board's enable pin for that channel, high), whether
# the channel is powered, and its code. The enable pins are GPIOs, not SPI,
# so the model is told which pins they are and reads them when asked.

try:
    from collections.abc import Callable
except ImportError:
    # MicroPython has no collections.abc and never evaluates annotations, so
    # the name only has to exist for the annotations to parse.
    Callable = None

FRAME_BYTES = 3
READ = 0x80

CONFIG = 0x03
GAIN = 0x04
TRIGGER = 0x05
DAC0 = 0x08

# TRIGGER's SOFT-RESET field: this value in its low four bits resets the
# device.
SOFT_RESET = 0b1010

# The registers after a reset: every channel's code zero and every channel
# powered. The firmware rewrites CONFIG and GAIN straight after its soft
# reset, so no other default is ever observed.
_RESET = {CONFIG: 0, GAIN: 0}


class Board:
    """The LED board's one SPI chip, and the enable pin in front of each of
    its channels."""

    def __init__(self, enable_pins: tuple) -> None:
        self.state_pins = tuple(enable_pins)
        self.regs = dict(_RESET)

    def datagram(self, chip_name: str, buf: bytes) -> bytes:
        if len(buf) != FRAME_BYTES:
            raise ValueError(
                f'{chip_name}: a {len(buf)}-byte transfer; the DAC80508 takes 3-byte frames'
            )
        if buf[0] & READ:
            raise ValueError(f'{chip_name}: a read frame; the firmware never reads the DAC')
        addr, value = buf[0], (buf[1] << 8) | buf[2]
        if addr == TRIGGER and value & 0xF == SOFT_RESET:
            self.regs = dict(_RESET)
        else:
            self.regs[addr] = value
        return bytes(FRAME_BYTES)

    def written(self, chip_name: str, buf: bytes) -> tuple:
        """(None, register, value) of the write a frame makes: the DAC has no
        axis, and every frame the model takes is a write."""
        return None, buf[0], (buf[1] << 8) | buf[2]

    def state(self, pin_value: 'Callable[[int], int]') -> dict:
        """{chip: [enabled, powered, code] per channel}."""
        config = self.regs[CONFIG]
        return {
            'DAC': [
                [bool(pin_value(pin)), not config >> ch & 1, self.regs.get(DAC0 + ch, 0)]
                for ch, pin in enumerate(self.state_pins)
            ]
        }


def build(
    sim: dict, ticks_us: 'Callable[[], int]', ticks_diff: 'Callable[[int, int], int]'
) -> Board:
    """The board's chips, as `machine` builds them at the first transfer."""
    return Board(tuple(sim['enable_pins']))
