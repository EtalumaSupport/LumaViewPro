# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The keys of a stock unit's fluorescence channels, read from the release.

A stock unit's fluorescence LEDs are addresses 0, 1 and 2, the LEDs the old
names ``Blue``, ``Green`` and ``Red`` always were; the release's old-name
table (``OldChannelNames``) names the key each address reads as. A test that
lights, sets or names one of a stock unit's fluorescence channels takes its
key from here, so it follows the catalogue rather than restating it.
"""

from modules.layer_record import release_old_names


def _stock_key(address: int) -> str:
    (key,) = [old.stock_key for old in release_old_names() if old.led_address == address]
    return key


# The stock key on each fluorescence LED address: 405, 488 and 589 nm.
LED0, LED1, LED2 = (_stock_key(address) for address in (0, 1, 2))
STOCK_FLUORESCENCE = (LED0, LED1, LED2)
