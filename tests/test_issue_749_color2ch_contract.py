# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#749 regression: colours a scope cannot drive must not light other LEDs.

Bug class: an undrivable colour represented in-band. `color2ch` returned a
real channel number (3 on RP2040-family boards, -1 on FX2) for colours the
scope has no LED for, so a luminescence acquire silently lit the brightfield
LED on EL-0940 boards and raised a range error on FX2 -- and the dark-floor
capture guard, keyed on `illumination > 0` alone, would reject a genuinely
dark luminescence frame as a failed capture once the LED correctly stays
dark.

This module grows with the fix stages:
- Stage 1 (here): the three dark-floor sites key on "the scope drives an
  LED for this channel", not illumination alone, and the composite
  luminescence grab gates its led_on on the same predicate.
- Stage 2 adds the driver contract tests (color2ch -> None) and the
  named-colour seam errors.
- Stage 3 adds the task-failure notification wording tests.
"""

from __future__ import annotations

import pathlib
import sys
from typing import ClassVar


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pytest

from modules import common_utils


def _driver_classes():
    from drivers.fx2driver import FX2LEDController
    from drivers.ledboard import LEDBoard
    from drivers.null_ledboard import NullLEDBoard
    from drivers.simulated_ledboard import SimulatedLEDBoard

    return [LEDBoard, NullLEDBoard, SimulatedLEDBoard, FX2LEDController]


class TestColor2chContract:
    """None is the only representation of 'this scope cannot drive it'."""

    def test_color2ch_unknown_is_none_all_drivers(self):
        for cls in _driver_classes():
            board = cls.__new__(cls)
            assert board.color2ch('NoSuchColour') is None, cls.__name__
            assert board.ch2color(99) is None, cls.__name__

    def test_known_colours_still_map(self):
        for cls in _driver_classes():
            board = cls.__new__(cls)
            assert board.color2ch('BF') == 3, cls.__name__
            assert board.ch2color(0) == 'Blue', cls.__name__


class _FourColourRecordingBoard:
    """LED-board double shaped like the FX2 (no PC/DF/Lumi channels),
    recording every command that reaches the driver."""

    _COLOR_TO_CH: ClassVar[dict] = {'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3}
    _CH_TO_COLOR: ClassVar[dict] = {v: k for k, v in _COLOR_TO_CH.items()}

    def __init__(self):
        self.commands = []

    def color2ch(self, color):
        return self._COLOR_TO_CH.get(color)

    def ch2color(self, channel):
        return self._CH_TO_COLOR.get(channel)

    def available_channels(self):
        return (0, 1, 2, 3)

    def available_colors(self):
        return tuple(self._COLOR_TO_CH)

    def led_on(self, channel, mA, **kwargs):
        self.commands.append(('on', channel, mA))
        return True

    def led_off(self, channel, **kwargs):
        self.commands.append(('off', channel))
        return True


@pytest.fixture
def sim_scope():
    from modules.lumascope_api import Lumascope

    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    yield s
    s.disconnect()


class TestSeamBehaviour:
    """The illumination API is the one place a colour becomes a channel."""

    def test_led_on_unknown_colour_names_the_colour(self, sim_scope):
        from modules.exceptions import ConfigError

        board = _FourColourRecordingBoard()
        sim_scope._led_driver = board
        with pytest.raises(ConfigError, match='Lumi'):
            sim_scope.illumination.led_on(channel='Lumi', mA=50)
        assert board.commands == [], 'no command may reach the driver'

    def test_led_off_unknown_colour_is_noop(self, sim_scope):
        board = _FourColourRecordingBoard()
        sim_scope._led_driver = board
        sim_scope.illumination.led_off(channel='PC')
        assert board.commands == [], 'off of an absent channel is already done'

    def test_numeric_none_channel_still_rejected(self, sim_scope):
        board = _FourColourRecordingBoard()
        sim_scope._led_driver = board
        with pytest.raises(ValueError):
            sim_scope.illumination.led_off(channel=None)


class TestTaskFailureNotificationWording:
    """The failure popup names the failed action; a protocol is blamed only
    when the task actually came off the protocol queue."""

    def _fire(self, monkeypatch, *, protocol: bool):
        from modules import notification_center
        from modules.sequential_io_executor import IOTask, SequentialIOExecutor

        executor = SequentialIOExecutor(max_workers=1, name='TEST_WORDING')
        try:
            calls = []
            monkeypatch.setattr(
                notification_center.notifications,
                'error',
                lambda title, subject, body, **kw: calls.append(body),
            )

            def sample_operation():
                pass

            task = IOTask(action=sample_operation, callback=lambda *a, **k: None)
            task.protocol = protocol
            # _on_task_done balances task_done() against whichever queue the
            # task's protocol flag says it came from.
            queue = executor.protocol_queue if protocol else executor.queue
            queue.put(task)
            queue.get_nowait()
            executor._on_task_done(task, None, RuntimeError('boom'))
            assert len(calls) == 1
            return calls[0]
        finally:
            executor.shutdown(wait=False)

    def test_live_task_names_action_not_protocol(self, monkeypatch):
        body = self._fire(monkeypatch, protocol=False)
        assert 'sample_operation' in body
        assert 'protocol' not in body.lower()

    def test_protocol_task_may_blame_the_protocol(self, monkeypatch):
        body = self._fire(monkeypatch, protocol=True)
        assert 'sample_operation' in body
        assert 'protocol' in body.lower()


class TestLayersWithLedSemantics:
    """The predicate source of truth: which channels drive an LED."""

    def test_luminescence_layers_have_no_led(self):
        with_led = common_utils.get_layers_with_led()
        for lumi_layer in common_utils.get_luminescence_layers():
            assert lumi_layer not in with_led

    def test_transmitted_and_fluorescence_layers_have_leds(self):
        with_led = common_utils.get_layers_with_led()
        for layer in (
            *common_utils.get_transmitted_layers(),
            *common_utils.get_fluorescence_layers(),
        ):
            assert layer in with_led


class TestDarkFloorKeysOnLedDrivability:
    """Seam guards: all three capture sites must key dark-floor rejection
    (and, for the composite loop, the led_on itself) on LED drivability,
    never on illumination alone. A source pin per site -- the behavioral
    seam tests ride the contract stage where the seam is directly drivable.
    """

    def test_protocol_writer_predicate(self):
        src = (REPO / 'modules' / 'protocol_image_writer.py').read_text()
        assert (
            "dark_floor_check=step['Illumination'] > 0\n"
            "                        and step['Color'] in common_utils.get_layers_with_led()"
        ) in src, 'protocol capture must exempt LED-less channels from dark-floor rejection'

    def test_composite_loop_gates_led_and_dark_floor_together(self):
        src = (REPO / 'ui' / 'composite_capture.py').read_text()
        assert (
            'led_driven = layer in common_utils.get_layers_with_led() and illumination > 0'
        ) in src, 'composite loop must compute LED drivability once'
        assert 'dark_floor_check=led_driven' in src, (
            'composite grab must expect a dark frame exactly when no LED was driven'
        )
        assert 'if layer not in common_utils.get_transmitted_layers():' not in src, (
            'the vacuously-true not-transmitted guard must not gate led_on'
        )

    def test_live_capture_predicate(self):
        src = (REPO / 'ui' / 'composite_capture.py').read_text()
        assert (
            'layer in common_utils.get_layers_with_led()\n'
            "            and layer_configs[layer]['illumination_ma'] > 0"
        ) in src, 'manual live capture must exempt LED-less channels from dark-floor rejection'
