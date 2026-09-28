# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The DAC80508 of the EL-0940's LED board (U50), as the field LED firmware
# sees it over SPI. Runs INSIDE the MicroPython child behind the `machine.SPI`
# stub, and imports under CPython too, so it is unit-tested in the suite.
# Pure Python: no dataclasses, no typing imports, nothing host-only.
#
# A transfer is one 24-bit frame, [address, MSB, LSB], written with chip
# select held low. The firmware never reads the DAC, so nothing drives MISO
# and every transfer answers zeros. The firmware detects no DAC fault
# either, so the model has none to inject.

try:
    from collections.abc import Callable
except ImportError:
    # MicroPython has no collections.abc and never evaluates annotations, so
    # the name only has to exist for the annotations to parse.
    Callable = None

FRAME_BYTES = 3


class Board:
    """The LED board's one SPI chip."""

    def datagram(self, chip_name: str, buf: bytes) -> bytes:
        if len(buf) != FRAME_BYTES:
            raise ValueError(
                f'{chip_name}: a {len(buf)}-byte transfer; the DAC80508 takes 3-byte frames'
            )
        return bytes(FRAME_BYTES)


def build(
    sim: dict, ticks_us: 'Callable[[], int]', ticks_diff: 'Callable[[int, int], int]'
) -> Board:
    """The board's chips, as `machine` builds them at the first transfer."""
    return Board()
