# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the per-spike frame-drop attribution logger.

The "uneven video" instrument emits one WARN when the gap between two live
frames drawn spikes above its recent-window baseline, with a breakdown so a
bench bundle can attribute each stutter. It is judged on the UI thread at
each live frame drawn (``ScopeDisplay._judge_live_draw``), by the record the
display thread adds to at each frame it picks up. The contract these tests
lock:

  * The interval is between live frames DRAWN -- duplicate/empty loop
    iterations never collapse it, and a UI thread too busy to draw is a
    stutter even when the display thread kept picking frames up.
  * The reported grab/proc/eng are the display thread's work on the frames
    it picked up since the last draw -- the display-path work that ran
    during the interval. gap = interval - that work.
  * It fires only when the interval beats BOTH the floor AND ratio-x-median,
    so a uniformly slow stream (median rises to match) does not fire.
  * A fast->slow transition is rate-limited so it does not log every frame
    until the median catches up.
  * Intended non-render time (protocol/derived-image holds, user pause) is
    subtracted before judging AND before feeding the median window: a
    deliberately held frame is not a stutter, and a hold chain must not
    inflate the baseline. A genuine residue past the hold still fires, with
    held= reported on the line.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.live_view_driver import LiveView, SlowFrameLines, live_view_widget
from ui.scope_display import (
    FRAME_SPIKE_FLOOR_MS,
    FRAME_SPIKE_LOG_MIN_GAP_S,
    FRAME_SPIKE_MIN_SAMPLES,
    FRAME_SPIKE_RATIO,
    FRAME_SPIKE_WINDOW,
    ScopeDisplay,
)


def _settled_at(baseline_ms, **compute):
    """A live view that has drawn a full window of frames ``baseline_ms`` apart."""
    view = LiveView(live_view_widget())
    view.frames(FRAME_SPIKE_WINDOW + 1, baseline_ms, **compute)
    return view


def _next(view, interval_ms, *, compute=(2.0, 8.0, 51.0)):
    """The next frame, picked up and drawn ``interval_ms`` after the last draw; the lines it wrote."""
    with SlowFrameLines() as lines:
        view.pick_up(after_ms=interval_ms, compute=compute)
        view.draw()
    return lines.lines


def test_the_first_draw_is_silent():
    view = LiveView(live_view_widget())
    with SlowFrameLines() as lines:
        view.frames(1, 5000.0)
        view.frames(1, 5000.0)
    assert lines.lines == []


def test_silent_until_window_has_min_samples():
    # A clear 250ms spike, but only a handful of baseline samples -> the median
    # is not yet meaningful, so no warn.
    view = LiveView(live_view_widget())
    view.frames(6, 33.0)
    assert FRAME_SPIKE_MIN_SAMPLES > 6
    assert _next(view, 250.0) == []


def test_fires_and_attributes_the_display_threads_work_since_the_last_draw():
    view = _settled_at(33.0)  # 2x = 66ms < the floor -> threshold = FRAME_SPIKE_FLOOR_MS
    lines = _next(view, 250.0, compute=(2.0, 8.0, 51.0))
    assert len(lines) == 1
    msg = lines[0]
    assert 'interval=250ms' in msg
    assert 'median=33ms' in msg
    assert 'grab=2.0ms proc=8.0ms eng=51.0ms' in msg
    # gap = 250 - (2 + 8 + 51) = 189ms -> the non-display remainder (upstream).
    assert 'gap=189ms' in msg


def test_the_work_is_every_frame_picked_up_since_the_last_draw_and_none_before_it():
    # Work on a frame drawn before this interval is not this interval's; work
    # on a frame picked up and never drawn (its post replaced) is.
    view = _settled_at(33.0, compute=(99.0, 99.0, 99.0))
    with SlowFrameLines() as lines:
        view.pick_up(after_ms=125.0, compute=(1.0, 4.0, 25.0))
        view.pick_up(after_ms=125.0, compute=(1.0, 4.0, 26.0))
        view.draw()
    assert len(lines.lines) == 1
    assert 'grab=2.0ms proc=8.0ms eng=51.0ms' in lines.lines[0]
    assert '99.0ms' not in lines.lines[0]


def test_a_ui_thread_too_busy_to_draw_is_a_slow_frame():
    # The camera delivers every 33 ms and the display thread picks every
    # frame up; the UI thread draws nothing for 330 ms. The viewer sees a
    # 330 ms stutter. Judged where the display thread picks up, it was none.
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        for _ in range(10):
            view.pick_up(after_ms=33.0)
        view.draw()
    assert len(lines.lines) == 1
    assert 'interval=330ms' in lines.lines[0]


def test_silent_just_below_threshold():
    view = _settled_at(33.0)  # 2x median is under the floor, so the floor is the threshold
    assert _next(view, FRAME_SPIKE_FLOOR_MS - 1.0) == []


def test_floor_suppresses_fast_stream_jitter():
    # Fast stream: median 10ms, 2x = 20ms, but the floor dominates, so a
    # 30ms blip (3x median) does NOT fire -- it is not a real stutter.
    assert FRAME_SPIKE_RATIO * 10.0 < FRAME_SPIKE_FLOOR_MS
    assert _next(_settled_at(10.0), 30.0) == []
    # ...but a genuine 200ms stall on the same fast stream does fire.
    assert len(_next(_settled_at(10.0), 200.0)) == 1


def test_uniformly_slow_stream_does_not_fire():
    # A steadily slow stream raises the median to match, so no single frame
    # beats 2x median. This is the case capture_fps owns, not this instrument.
    view = _settled_at(2000.0)  # ~0.5 fps, the old fps-bug symptom
    assert _next(view, 2000.0) == []
    assert _next(view, 2100.0) == []


def test_rate_limit_caps_a_fast_to_slow_burst():
    view = _settled_at(33.0)
    # First spike fires.
    assert len(_next(view, 250.0)) == 1
    # A second spike within FRAME_SPIKE_LOG_MIN_GAP_S is suppressed (the median
    # has not yet caught up to the new slow rate, but we don't flood the log).
    assert _next(view, 250.0) == []
    # Once the min gap has elapsed, a distinct later stutter logs again.
    view.frames(round(FRAME_SPIKE_LOG_MIN_GAP_S * 1000 / 33.0) + 1, 33.0)
    assert len(_next(view, 300.0)) == 1


def test_intentional_hold_fully_accounting_for_excess_is_silent():
    # A protocol hold chains a saved frame on screen: the 550ms wall-clock span
    # is 517ms of deliberate hold + a normal 33ms frame. Not a stutter.
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        view.held(517.0)
        view.pick_up(after_ms=33.0)
        view.draw()
    assert lines.lines == []
    # The median window got the ADJUSTED interval, so a hold chain does not
    # inflate the baseline and mask real stutters afterwards: a 160 ms
    # stutter is still judged against 33 ms.
    for _ in range(FRAME_SPIKE_WINDOW // 2):
        view.held(517.0)
        view.frames(1, 33.0)
    assert len(_next(view, FRAME_SPIKE_FLOOR_MS + 10.0)) == 1


def test_a_hold_with_no_draw_of_its_own_then_a_live_draw_is_silent():
    # The pause, or a hold whose saved frame is drawn by the held path, not
    # as a live frame: no live draw ends it, so the next live draw's
    # interval spans it, and its time is taken out.
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        view.pick_up(after_ms=33.0)
        view.held(2000.0)
        view.draw(after_ms=10.0)
    assert lines.lines == []


def test_hold_with_genuine_residue_fires_and_reports_held():
    # 750ms span = 500ms hold + 250ms real gap: the residue is a genuine spike
    # and must still fire, with the hold reported so the line accounts for the
    # full wall-clock span.
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        view.held(500.0)
        view.pick_up(after_ms=250.0, compute=(2.0, 8.0, 51.0))
        view.draw()
    assert len(lines.lines) == 1
    msg = lines.lines[0]
    assert 'interval=250ms' in msg
    assert 'held=500ms' in msg
    # gap arithmetic stays on the judged interval: 250 - (2+8+51) = 189ms.
    assert 'gap=189ms' in msg


def test_no_hold_line_has_no_held_note():
    lines = _next(_settled_at(33.0), 250.0)
    assert len(lines) == 1
    assert 'held=' not in lines[0]


# --- The line names who was late ---------------------------------------------
#
# camera= is the longest gap between the arrivals of two frames the display
# picked up in a row, and skipped= the frames the camera stored between them
# that it never picked up: a long gap with nothing skipped is the camera; one
# with frames skipped is the display thread or the process. ui_wait= is how
# long the oldest waiting live frame waited for the UI thread to draw it.


def test_a_ui_thread_stall_is_named_by_its_wait():
    # The camera steady at 33 ms, every frame picked up; the UI thread draws
    # nothing for 330 ms. The first frame posted after the last draw waited
    # 297 ms of it.
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        for _ in range(10):
            view.pick_up(after_ms=33.0)
        view.draw()
    assert len(lines.lines) == 1
    assert 'camera=33ms skipped=0 ui_wait=297ms' in lines.lines[0]


def test_a_camera_that_delivered_late_is_named_by_its_gap():
    lines = _next(_settled_at(33.0), 330.0)
    assert len(lines) == 1
    assert 'camera=330ms skipped=0 ui_wait=0ms' in lines[0]


def test_a_display_thread_stall_is_named_by_the_frames_it_skipped():
    # The camera steady at 33 ms, the display thread held 330 ms: the frames
    # stored meanwhile were never picked up.
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        view.pick_up(after_ms=330.0, skipped=9)
        view.draw()
    assert len(lines.lines) == 1
    assert 'camera=330ms skipped=9 ui_wait=0ms' in lines.lines[0]


def test_a_live_frame_a_hold_kept_off_the_screen_did_not_wait_for_the_ui_thread():
    view = _settled_at(33.0)
    with SlowFrameLines() as lines:
        view.pick_up(after_ms=33.0)
        view.widget._drop_waiting_post()
        view.pick_up(after_ms=300.0)
        view.draw(after_ms=20.0)
    assert len(lines.lines) == 1
    assert 'ui_wait=20ms' in lines.lines[0]


@pytest.mark.parametrize(
    ('minimized', 'focused', 'shown'),
    [(False, True, 'focused'), (False, False, 'unfocused'), (True, False, 'minimized')],
)
def test_the_line_states_the_windows_state_when_drawn(monkeypatch, minimized, focused, shown):
    # parametrize-justified: three window states, each its own word on the line.
    import ui.scope_display as scope_display_module

    # a stand-in by design: Kivy's window never runs here; the display reads
    # only its focus.
    monkeypatch.setattr(scope_display_module, 'Window', SimpleNamespace(focus=focused))
    view = _settled_at(33.0)
    view.widget.window_minimized = lambda: minimized
    lines = _next(view, 250.0)
    assert len(lines) == 1
    assert f'window={shown} ' in lines[0]


# --- [FRAME INTERVAL] states the intervals between live frames drawn ---------
#
# The metrics line's percentiles were taken over the display loop's passes,
# so a loop polling every few milliseconds read its own cadence, not what the
# viewer saw. They are the intervals the slow-frame judge measures at each
# live draw, holds and stops taken out.


def test_the_frame_interval_is_the_interval_between_live_frames_drawn():
    view = LiveView(live_view_widget())
    view.frames(41, 33.0)
    percentiles = view.widget.frame_interval_percentiles_ms()
    assert (percentiles['n'], round(percentiles['p50'])) == (40, 33)


def test_the_frame_interval_takes_a_hold_out():
    view = LiveView(live_view_widget())
    view.frames(5, 38.0)
    view.held(500.0)
    view.frames(1, 38.0)
    assert round(view.widget.frame_interval_percentiles_ms()['max']) == 38


# --- The floor is a perceptible stutter, not the display quantum ---------------
#
# At the 30 fps cap the displayed-frame median is one ~31 ms period, and the
# display loop is scheduled in 15.6 ms Windows ticks, so a single missed tick
# is a 63 ms interval. A floor at the quantum fires on scheduler jitter and on
# the engineering-stats hitch every 0.5 s; the corpus on the bench machine held
# 49,749 such lines. The floor is the shortest gap a person watching the live
# view could see as a stutter; the ratio term still carries long exposures.


def test_single_missed_tick_on_a_30fps_stream_is_silent():
    for interval_ms in (63.0, 94.0, 125.0):
        lines = _next(_settled_at(31.0), interval_ms, compute=(0.0, 0.0, 94.0))
        assert lines == [], f'{interval_ms:.0f}ms on a 31ms median must not warn'


def test_perceptible_stutter_on_a_30fps_stream_fires():
    interval_ms = FRAME_SPIKE_FLOOR_MS + 10.0
    lines = _next(_settled_at(31.0), interval_ms)
    assert len(lines) == 1
    assert f'interval={interval_ms:.0f}ms' in lines[0]


def test_floor_is_the_perception_bound_not_the_display_quantum():
    assert FRAME_SPIKE_FLOOR_MS >= 100.0
    # A long-exposure layer: 100 ms median. The ratio governs there, and a
    # 250 ms interval is still a spike.
    assert len(_next(_settled_at(100.0), 250.0)) == 1


# --- The engineering stats run on a stride-4 view -----------------------------


def test_engineering_stats_use_a_stride_4_view():
    frame = np.zeros((1900, 1900), dtype=np.uint8)
    frame[::4, ::4] = 200
    # The full-frame mean is 12.5; the stride-4 lattice is uniformly 200.
    mean, stddev = ScopeDisplay._engineering_stats(frame)
    assert (mean, stddev) == (200.0, 0.0)
    assert type(mean) is float and type(stddev) is float
    colour = np.full((1900, 1900, 3), 7, dtype=np.uint8)
    mean_c, std_c = ScopeDisplay._engineering_stats(colour)
    assert (mean_c, std_c) == (7.0, 0.0)


# --- A stretch in which no thread of the process ran is not a slow frame -----
#
# A sim run stopped for 10 s (kill -STOP) logged "[SLOW FRAME] interval=10033ms":
# the camera and the display were not late, the process did not run. The
# session's heartbeat says that stretch; the judge and the histogram take it
# out, as they take out a hold.


@pytest.fixture
def stopped_10_s(sim_session, monkeypatch):
    """The session's heartbeat having recorded a stop from 1000.0 to 1010.0 (time.monotonic)."""
    import types

    import _lumascope.modules.app_context as app_context
    import _lumascope.modules.metrics_logger as metrics_logger_module

    # a stand-in by design: no test can stop its own process for a stretch
    # and go on; the heartbeat reads only time.monotonic, and this one jumps
    # as a standby does.
    clock = types.SimpleNamespace(now=999.0)
    monkeypatch.setattr(
        metrics_logger_module, 'time', types.SimpleNamespace(monotonic=lambda: clock.now)
    )
    monkeypatch.setattr(app_context, 'ctx', app_context.AppContext.over_session(sim_session))
    watch = sim_session.metrics_logger
    while clock.now < 1000.0:
        watch.tick_heartbeat()
        clock.now += metrics_logger_module.HEARTBEAT_INTERVAL_S
    watch.tick_heartbeat()
    clock.now = 1010.0
    watch.tick_heartbeat()


def _settled_before_the_stop():
    """A live view settled at 33 ms whose last draw is at 1000.0, where the stop begins."""
    view = LiveView(live_view_widget(), now=1000.0 - (FRAME_SPIKE_WINDOW + 1) * 0.033)
    view.frames(FRAME_SPIKE_WINDOW + 1, 33.0)
    view.now = 1000.0
    return view


def test_a_frame_gap_the_process_did_not_run_is_not_a_slow_frame(stopped_10_s):
    view = _settled_before_the_stop()
    assert _next(view, 10033.0) == []
    # The stop did not reach the baseline: a 160 ms stutter is still one.
    assert len(_next(view, FRAME_SPIKE_FLOOR_MS + 10.0)) == 1


def test_a_stutter_beside_a_stop_fires_and_reports_the_stop(stopped_10_s):
    lines = _next(_settled_before_the_stop(), 10250.0)
    assert len(lines) == 1, lines
    assert 'interval=250ms' in lines[0] and 'stopped=10000ms' in lines[0], lines


def test_the_frame_interval_takes_a_stop_out(stopped_10_s):
    view = _settled_before_the_stop()
    _next(view, 10038.0)
    assert round(view.widget.frame_interval_percentiles_ms()['max']) == 38


def test_the_bullseye_shown_with_debug_on_keeps_no_waits_for_the_summary(monkeypatch):
    # The [PERF] summary is written, and the waits it reads cleared, only
    # in the mono view; a bullseye shown for an hour with debug on kept one
    # per draw. The kept waits are the summary's only input and the summary
    # a debug line, so the list is read here.
    from _lumascope.modules import app_context

    # a stand-in by design: only debug_mode is read, and no session runs.
    monkeypatch.setattr(
        app_context, 'ctx', SimpleNamespace(settings={'debug_mode': True}, session=None)
    )
    view = LiveView(live_view_widget())
    view.widget.use_bullseye = True
    view.frames(50, 66.0)
    assert view.widget._perf_ui_waits == []
    view.widget.use_bullseye = False
    view.frames(1, 33.0)
    assert len(view.widget._perf_ui_waits) == 1
