"""The Classic (FX2) LED current-to-byte mapping (#800).

The FX2 LED peripheral takes a 3-byte frame: ``0xFF`` preamble, channel,
brightness. The brightness byte is linear in current with full scale at
the driver's ``_MAX_MA``. Two defects lived in the mapping:

* the divisor was 200 on an 840 mA board, so every request under 200 mA
  drove the LED at 4.2x its number and every request at or above 200 mA
  saturated;
* the saturated byte was ``0xFF`` -- byte-identical to the frame preamble,
  so the peripheral dropped the frame and the channel stayed dark.

These tests pin the corrected scale and the ceiling that keeps the value
byte from ever repeating the preamble.
"""

from __future__ import annotations

from drivers import fx2driver


def _led() -> fx2driver.FX2LEDController:
    # No USB handle is needed to exercise the pure conversion.
    return object.__new__(fx2driver.FX2LEDController)


def test_full_scale_is_the_classic_board_ceiling():
    assert fx2driver.FX2LEDController._MAX_MA == 840


def test_zero_current_is_byte_zero():
    assert _led()._ma_to_brightness(0) == 0x00


def test_half_scale_is_byte_128():
    # 420 mA on an 840 mA board is exactly half of 255, rounded.
    assert _led()._ma_to_brightness(420) == 0x80


def test_full_scale_stops_one_below_the_preamble():
    assert _led()._ma_to_brightness(840) == 0xFE


def test_no_current_in_range_produces_the_preamble_byte():
    """The test that would have failed for the whole life of #800.

    ``0xFF`` is the frame preamble; a value byte equal to it is dropped by
    the peripheral and the LED goes dark instead of bright.
    """
    led = _led()
    colliding = [mA for mA in range(0, 10001) if led._ma_to_brightness(mA) == 0xFF]
    assert colliding == [], f'these requests would send the preamble as a value: {colliding[:5]}...'
