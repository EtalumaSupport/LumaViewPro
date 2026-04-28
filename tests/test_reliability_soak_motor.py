"""Regression tests for motor reliability-soak path.

The motor soak shares the LED soak's structure but uses driver-method
dispatch so it works on both FW4.0 and v3.0.x firmware. Tests pin:

* `cmd_reliability_soak` dispatches on `args.board`.
* `_soak_motor` is **read-only** — never calls motion-causing methods
  (``move``, ``home``, ``zhome``, ``thome``, ``home_axis``,
  ``xycenter``, ``stop``). Verified by source inspection. Bench safety:
  the soak runs against any motor board including stages with broken
  hardware (e.g. SN 7162-19 has a known-bad XY chip) and must not move.
* Argparse exposes `--board {led, motor}`.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.firmware_tools import cmd_reliability_soak


def _read_source() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / 'tools/firmware_tools.py').read_text()


class TestDispatch:
    def test_board_led_routes_to_soak_led(self):
        args = MagicMock()
        args.board = 'led'
        with patch('tools.firmware_tools._soak_led') as soak_led, \
             patch('tools.firmware_tools._soak_motor') as soak_motor:
            cmd_reliability_soak(args)
        soak_led.assert_called_once_with(args)
        soak_motor.assert_not_called()

    def test_board_motor_routes_to_soak_motor(self):
        args = MagicMock()
        args.board = 'motor'
        with patch('tools.firmware_tools._soak_led') as soak_led, \
             patch('tools.firmware_tools._soak_motor') as soak_motor:
            cmd_reliability_soak(args)
        soak_motor.assert_called_once_with(args)
        soak_led.assert_not_called()


class TestMotorSoakIsReadOnly:
    """Source-inspection: motor soak body must never reference motion-
    causing methods. Mocking would technically be more thorough but
    this catches accidental motion-call introductions at edit time
    without needing the full test harness."""

    @pytest.fixture
    def soak_motor_body(self):
        src = _read_source()
        start = src.find('def _soak_motor(')
        assert start != -1, '_soak_motor not found in firmware_tools.py'
        # Body ends at the next top-level def.
        end = src.find('\ndef ', start + 1)
        if end == -1:
            end = len(src)
        return src[start:end]

    @pytest.mark.parametrize('forbidden', [
        '.move(',
        '.home(',
        '.zhome(',
        '.thome(',
        '.home_axis(',
        '.xycenter(',
        '.stop(',
        "'cmd': 'POS_WRITE'",
        "'cmd': 'HOME'",
        "'cmd': 'STOP'",
        'TARGET_W',
    ])
    def test_no_motion_call(self, soak_motor_body, forbidden):
        """Each forbidden pattern would cause physical motion if invoked.
        The motor soak must never call any of them — bench safety."""
        assert forbidden not in soak_motor_body, (
            f'_soak_motor body contains motion-causing pattern {forbidden!r}; '
            f'motor reliability soak must remain read-only.'
        )

    def test_uses_pos_read_and_limit_sw(self, soak_motor_body):
        """Sanity check: the soak does the right reads."""
        assert "'POS_READ'" in soak_motor_body or 'POS_READ' in soak_motor_body
        assert "'LIMIT_SW'" in soak_motor_body or 'LIMIT_SW' in soak_motor_body


class TestArgparse:
    def test_board_choice_exposes_both(self):
        src = _read_source()
        # Argparse line shape (resilient to formatting):
        # '--board', choices=['led', 'motor']
        assert "'--board'" in src
        assert "choices=['led', 'motor']" in src
