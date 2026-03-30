import threading
import time

import numpy as np
import pytest

from modules.video_capture import StimulationController, VideoCaptureSession


class FakeFrameValidity:
    def frames_until_valid(self):
        return 0

    def count_frame(self):
        pass


class FakeScope:
    def __init__(self):
        self.frame_validity = FakeFrameValidity()
        self.calls = []

    def color2ch(self, color):
        return {
            "Blue": 0,
            "Green": 1,
            "Red": 2,
            "BF": 3,
            "PC": 4,
            "DF": 5,
        }[color]

    def led_on_fast(self, channel, mA):
        self.calls.append(("on", channel, mA))

    def led_off_fast(self, channel):
        self.calls.append(("off", channel, None))

    def led_on(self, channel, mA):
        self.calls.append(("on_slow", channel, mA))

    def led_off(self, channel):
        self.calls.append(("off_slow", channel, None))

    def get_image(self, force_to_8bit=True, force_new_capture=False):
        return np.zeros((2, 2), dtype=np.uint8)

    def set_auto_gain(self, state, settings):
        pass

    def auto_gain_once(self, **kwargs):
        pass


def test_build_edge_schedule_single_channel_produces_two_edges_per_pulse():
    scheduler = StimulationController(
        FakeScope(),
        {
            "Red": {
                "enabled": True,
                "illumination": 100,
                "frequency": 2.0,
                "pulse_width": 50,
                "pulse_count": 3,
            }
        },
    )

    assert len(scheduler._edges) == 6
    assert scheduler._edges[0].action == "on"
    assert scheduler._edges[1].action == "off"
    assert scheduler._edges[0].target_offset_s == 0.0


def test_build_edge_schedule_sorts_simultaneous_edges_off_before_on():
    scheduler = StimulationController(
        FakeScope(),
        {
            "Red": {
                "enabled": True,
                "illumination": 100,
                "frequency": 1.0,
                "pulse_width": 500,
                "pulse_count": 1,
            },
            "Green": {
                "enabled": True,
                "illumination": 100,
                "frequency": 2.0,
                "pulse_width": 50,
                "pulse_count": 2,
            },
        },
    )

    matching_edges = [
        edge for edge in scheduler._edges
        if abs(edge.target_offset_s - 0.5) < 1e-9
    ]
    assert [edge.action for edge in matching_edges] == ["off", "on"]
    assert [edge.color for edge in matching_edges] == ["Red", "Green"]


def test_invalid_channel_is_skipped_without_aborting_valid_channels():
    scheduler = StimulationController(
        FakeScope(),
        {
            "Red": {
                "enabled": True,
                "illumination": 100,
                "frequency": 0,
                "pulse_width": 10,
                "pulse_count": 3,
            },
            "Blue": {
                "enabled": True,
                "illumination": 80,
                "frequency": 5.0,
                "pulse_width": 20,
                "pulse_count": 2,
            },
        },
    )

    assert len(scheduler._edges) == 4
    assert all(edge.color == "Blue" for edge in scheduler._edges)


def test_pulse_width_is_clamped_to_ninety_percent_of_period():
    scheduler = StimulationController(
        FakeScope(),
        {
            "Blue": {
                "enabled": True,
                "illumination": 90,
                "frequency": 10.0,
                "pulse_width": 200,
                "pulse_count": 1,
            }
        },
    )

    on_edge, off_edge = scheduler._edges
    assert on_edge.action == "on"
    assert off_edge.action == "off"
    assert off_edge.target_offset_s == pytest.approx(0.09)


def test_scheduler_stop_exits_cleanly_and_turns_off_channels():
    scope = FakeScope()
    scheduler = StimulationController(
        scope,
        {
            "Red": {
                "enabled": True,
                "illumination": 100,
                "frequency": 100.0,
                "pulse_width": 2,
                "pulse_count": 50,
            },
            "Green": {
                "enabled": True,
                "illumination": 120,
                "frequency": 100.0,
                "pulse_width": 2,
                "pulse_count": 50,
            },
        },
    )

    start_event = threading.Event()
    stop_event = threading.Event()
    thread = threading.Thread(target=scheduler.run, args=(start_event, stop_event))
    thread.start()
    start_event.set()
    time.sleep(0.01)
    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert ("off", 2, None) in scope.calls
    assert ("off", 1, None) in scope.calls


def test_video_capture_session_creates_one_stim_thread(monkeypatch):
    created_threads = []

    class RecordingThread:
        def __init__(self, target, name, args):
            self.target = target
            self.name = name
            self.args = args
            self._alive = False
            created_threads.append(self)

        def start(self):
            self._alive = True

        def join(self, timeout=None):
            self._alive = False

        def is_alive(self):
            return self._alive

    monkeypatch.setattr("modules.video_capture.threading.Thread", RecordingThread)

    scope = FakeScope()
    step = {
        "Exposure": 10,
        "Auto_Gain": False,
        "Video Config": {"duration": 0.03},
        "Color": "BF",
        "False_Color": False,
        "Stim_Config": {
            "Red": {
                "enabled": True,
                "illumination": 100,
                "frequency": 5.0,
                "pulse_width": 20,
                "pulse_count": 3,
            },
            "Green": {
                "enabled": True,
                "illumination": 110,
                "frequency": 5.0,
                "pulse_width": 20,
                "pulse_count": 3,
            },
            "Blue": {
                "enabled": False,
                "illumination": 90,
                "frequency": 5.0,
                "pulse_width": 20,
                "pulse_count": 3,
            },
        },
    }

    session = VideoCaptureSession(
        scope=scope,
        step=step,
        autogain_settings={},
        is_protocol_running_fn=lambda: True,
        callbacks={},
        leds_off_fn=lambda: None,
    )
    result = session.capture()

    assert result is not None
    assert len(created_threads) == 1
    assert created_threads[0].name == "stim-scheduler"
