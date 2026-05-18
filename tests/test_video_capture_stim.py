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

    # LAYER-F: production code now reads frame validity through the
    # Lumascope API rather than reaching into self.frame_validity
    # directly. Mirror the delegating accessors here.
    def frames_until_valid(self):
        return self.frame_validity.frames_until_valid()

    def count_frame(self):
        self.frame_validity.count_frame()

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


class TimestampingScope:
    """FakeScope variant that records perf_counter timestamps of led_on/off_fast.

    Used by the pulse-width-jitter regression test. The StimulationController
    detects `led_on_fast`/`led_off_fast` via hasattr and prefers them, so
    recording them here is equivalent to recording the scheduler's actual
    edge dispatch times.
    """

    def __init__(self):
        self.frame_validity = FakeFrameValidity()
        self.events = []  # list of (action, channel, mA, t_perf)

    # LAYER-F: production code now reads frame validity through the
    # Lumascope API rather than reaching into self.frame_validity
    # directly. Mirror the delegating accessors here.
    def frames_until_valid(self):
        return self.frame_validity.frames_until_valid()

    def count_frame(self):
        self.frame_validity.count_frame()

    def color2ch(self, color):
        return {
            "Blue": 0, "Green": 1, "Red": 2, "BF": 3, "PC": 4, "DF": 5,
        }[color]

    def led_on_fast(self, channel, mA):
        self.events.append(("on", channel, mA, time.perf_counter()))

    def led_off_fast(self, channel):
        self.events.append(("off", channel, None, time.perf_counter()))

    def led_on(self, channel, mA):
        self.events.append(("on_slow", channel, mA, time.perf_counter()))

    def led_off(self, channel):
        self.events.append(("off_slow", channel, None, time.perf_counter()))


@pytest.mark.timing_sensitive
def test_scheduler_pulse_width_jitter_within_tolerance():
    """Regression test for the stim scheduler spin-yield fix.

    Guards against reintroducing `time.sleep(~100us)` (or any GIL-yielding
    call) inside `StimulationController._wait_until`'s final pre-edge spin
    window. A 100us sleep yields to the OS scheduler, whose wake-up latency
    (100us to 20+ms on macOS/Linux desktops) inflates OFF-edge timing for
    short pulses.

    Historical measurement (dev bench, 10 pulses @ 0.8 Hz, 10 ms pulse width,
    single channel):
        yielding spin  -> pulse-width stddev 5.91 ms, worst-case err 16.27 ms
        busy-wait spin -> pulse-width stddev 1.70 ms, worst-case err  3.88 ms

    Thresholds below are intentionally well above the busy-wait numbers but
    well below the yielding-spin numbers so the test distinguishes the two
    regimes without being flaky on a lightly loaded dev machine. Gated
    behind `--run-timing-sensitive` to keep CI/default runs deterministic.
    """
    scope = TimestampingScope()
    scheduler = StimulationController(
        scope,
        {
            "Green": {
                "enabled": True,
                "illumination": 500,
                "frequency": 0.8,
                "pulse_width": 10,   # ms — the regime the bug affected
                "pulse_count": 10,
            },
        },
    )

    start_event = threading.Event()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=scheduler.run, args=(start_event, stop_event),
        name="stim-scheduler-regression-test",
    )
    thread.start()
    start_event.set()

    # Schedule horizon: 10 pulses at 0.8 Hz = 9 * 1.25 s + 10 ms ~= 11.26 s.
    # Allow a generous join margin.
    thread.join(timeout=15.0)
    assert not thread.is_alive(), "Scheduler thread did not exit in time"

    green_ch = scope.illumination.color2ch("Green")
    ons = [t for (a, c, _mA, t) in scope.events if a == "on" and c == green_ch]
    offs = [t for (a, c, _mA, t) in scope.events if a == "off" and c == green_ch]
    # Scheduler issues a final cleanup `led_off_fast` in its finally block, so
    # OFF count is 10 (scheduled) + 1 (cleanup) = 11. Pair each ON with the
    # first OFF after it.
    assert len(ons) == 10, f"Expected 10 ON edges, got {len(ons)}"
    assert len(offs) >= 10, f"Expected >=10 OFF edges, got {len(offs)}"

    paired_offs = []
    off_iter = iter(offs)
    next_off = next(off_iter, None)
    for on in ons:
        while next_off is not None and next_off < on:
            next_off = next(off_iter, None)
        assert next_off is not None, (
            f"Could not find a matching OFF edge for ON at {on}"
        )
        paired_offs.append(next_off)
        next_off = next(off_iter, None)

    pulse_widths_ms = [(off - on) * 1000.0 for on, off in zip(ons, paired_offs)]
    widths = np.asarray(pulse_widths_ms, dtype=float)
    stddev_ms = float(widths.std(ddof=0))
    worst_err_ms = float(np.max(np.abs(widths - 10.0)))
    mean_ms = float(widths.mean())

    # One-run record: printed so the run log shows what this machine measured
    # on the day the test was added (see DAILY_LOG 2026-04-19 / 2026-04-20).
    print(
        f"\n[pulse-width-jitter] n=10 target=10.0 ms "
        f"mean={mean_ms:.3f} ms stddev={stddev_ms:.3f} ms "
        f"worst_err={worst_err_ms:.3f} ms "
        f"widths={[round(w, 3) for w in pulse_widths_ms]}"
    )

    # Thresholds: headline pass line from the post-bench TODO. A GIL-yielding
    # spin produces stddev ~5.9 ms and worst-case ~16 ms, so these catch the
    # regression with ~2x margin on the busy-wait baseline.
    assert stddev_ms < 3.0, (
        f"Pulse-width stddev {stddev_ms:.2f} ms exceeds 3.0 ms — suggests "
        f"_wait_until's final spin is yielding the GIL (regression of the "
        f"busy-wait fix). Widths: {pulse_widths_ms}"
    )
    assert worst_err_ms < 10.0, (
        f"Worst-case pulse-width error {worst_err_ms:.2f} ms exceeds 10.0 ms "
        f"— suggests _wait_until's final spin is yielding the GIL (regression "
        f"of the busy-wait fix). Widths: {pulse_widths_ms}"
    )


def test_video_capture_session_creates_one_stim_thread(monkeypatch):
    created_threads = []

    class RecordingThread:
        def __init__(self, target, name, args, daemon=False):
            self.target = target
            self.name = name
            self.args = args
            self.daemon = daemon
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
    assert created_threads[0].daemon is True, (
        "stim_thread must be daemon=True so app exit reaps it without "
        "hang on an in-flight scheduler iteration (Rule 41 + LVP "
        "f4920c8 daemon-flag fix)"
    )
