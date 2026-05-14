# Copyright Etaluma, Inc.
"""Regression tests for 2026-05-14 PM bench-surfaced cluster.

Both bugs surfaced in the same SN12062 bench session and are bundled
here because they fit the same pattern (firmware response shape that
the LVP-side code didn't tolerate):

1. `LEDBoard.led_on(block=True)` retried forever when the EL-0925 Gen3
   firmware (2024-06-05ESWEA) acked an LED write with an empty string.
   The old polling loop required either the original command in the
   response OR specific substring matches; empty string matched
   neither, so the loop kept re-sending until the 5s deadline,
   halting any protocol that called led_on with block=True
   (issue #651).

2. TSR bundler emitted `SN<sn>-<timestamp>.zip` -- indistinguishable
   from the SNlogs user-dump bundle. The TSR token in the filename
   makes the two trivially sortable (issue #652).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLedOnBlockAcceptsEmptyAck:
    """Empty-string firmware response counts as an ack; no retry storm."""

    def _make_led(self, response_sequence):
        """Build a real LEDBoard with exchange_command monkeypatched."""
        from drivers.ledboard import LEDBoard
        led = LEDBoard.__new__(LEDBoard)  # bypass __init__ (needs serial)
        # Minimal state for led_on path
        led._validate_and_build_led_cmd = lambda ch, mA: ('BF', f'LED{ch}_{int(mA)}')
        led._update_state_cache = lambda color, mA: None
        # exchange_command returns the next response in the sequence
        responses = list(response_sequence)
        led.exchange_command = MagicMock(side_effect=lambda cmd: responses.pop(0) if responses else None)
        return led

    def test_empty_string_response_breaks_polling_loop(self):
        # EL-0925 Gen3 firmware (2024-06-05ESWEA) shape: empty string ack.
        led = self._make_led([''])
        led.led_on(channel=3, mA=2, block=True, timeout=5.0)
        # Only one call to exchange_command -- the initial send.
        # If the old code path ran, we'd see ~20+ retries.
        assert led.exchange_command.call_count == 1

    def test_echoed_command_response_breaks_polling_loop(self):
        # Older firmware shape: command echoed back.
        led = self._make_led(['LED3_2'])
        led.led_on(channel=3, mA=2, block=True, timeout=5.0)
        assert led.exchange_command.call_count == 1

    def test_none_response_retries_until_timeout(self):
        # Firmware completely unresponsive (None means no bytes back).
        # The driver must keep retrying until the deadline -- this is
        # the "actually broken serial" path the polling loop still
        # needs to guard.
        led = self._make_led([None] * 200)
        import time
        t0 = time.monotonic()
        led.led_on(channel=3, mA=2, block=True, timeout=0.1)
        elapsed = time.monotonic() - t0
        # Should have polled for ~0.1s, not bailed immediately.
        assert 0.08 < elapsed < 0.3, f"elapsed={elapsed:.3f}s out of band"
        # Multiple retries during the 0.1s window.
        assert led.exchange_command.call_count >= 3


class TestLedDriverNoSubstringCheck:
    """The brittle substring-match block on the polling loop is retired.

    The old code path used `check_each_substr(['LED', ch, mA], response)`
    plus `command not in response` to decide whether to keep polling.
    That shape rejected the EL-0925 Gen3 empty-string ack. The fix
    removed the substring check entirely -- ANY non-None response means
    the firmware received the command. The retired pattern must not
    creep back via a future merge.
    """

    def test_substring_match_helper_retired(self):
        src = (REPO_ROOT / 'drivers' / 'ledboard.py').read_text()
        assert 'check_each_substr' not in src
        # The old `command not in response` rejection logic also leaves.
        assert 'command not in response' not in src


class TestTsrFilenameHasTsrToken:
    """TSR zip filename has 'TSR' between SN and timestamp so it's
    visually distinct from SNlogs-... user-dump bundles."""

    def test_filename_pattern_in_source(self):
        src = (REPO_ROOT / 'modules' / 'tech_support_report.py').read_text()
        # The TSR token must be present in the f-string that builds the
        # zip path.
        assert 'SN{clean_sn}-TSR-{ts}.zip' in src
        # And the old pattern must not survive.
        assert 'SN{clean_sn}-{ts}.zip' not in src
