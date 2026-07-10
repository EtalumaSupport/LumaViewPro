import threading
import time

import numpy as np
import pytest

from modules.video_capture import (
    STIM_END_CAPTURE_FAULT,
    STIM_END_EMPTY_SCHEDULE,
    STIM_END_INCOMPLETE,
    STIM_END_JOIN_TIMEOUT,
    STIM_END_SCHEDULE_COMPLETE,
    STIM_END_STOP_EVENT_SET,
    _CLEAN_STIM_END_REASONS,
    StimulationController,
    VideoCaptureSession,
)


class FakeFrameValidity:
    def frames_until_valid(self):
        return 0

    def count_frame(self):
        pass


class FakeScope:
    def __init__(self):
        self.frame_validity = FakeFrameValidity()
        self.calls = []
        # Wave 7 Phase 3f: production code now reaches LED methods via
        # `self._scope.illumination.<method>`. Aliasing illumination to
        # self lets the existing led_on/led_off/color2ch methods serve
        # both call shapes.
        self.illumination = self
        self.imaging = self
        # Production reads this when a stim run starts, to attribute pulses
        # to whichever run owns the LED lease. None == no run owns the LEDs.
        self.led_lease_owner = None

    # LAYER-F: production code now reads frame validity through the
    # Lumascope API rather than reaching into self.frame_validity
    # directly. Mirror the delegating accessors here.
    def frames_until_valid(self):
        return self.frame_validity.frames_until_valid()

    def count_frame(self):
        self.frame_validity.count_frame()

    def color2ch(self, color):
        return {
            'Blue': 0,
            'Green': 1,
            'Red': 2,
            'BF': 3,
            'PC': 4,
            'DF': 5,
        }[color]

    def led_on_fast(self, channel, mA):
        self.calls.append(('on', channel, mA))

    def led_off_fast(self, channel):
        self.calls.append(('off', channel, None))

    def led_on(self, channel, mA):
        self.calls.append(('on_slow', channel, mA))

    def led_off(self, channel):
        self.calls.append(('off_slow', channel, None))

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
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 2.0,
                'pulse_width': 50,
                'pulse_count': 3,
            }
        },
    )

    assert len(scheduler._edges) == 6
    assert scheduler._edges[0].action == 'on'
    assert scheduler._edges[1].action == 'off'
    assert scheduler._edges[0].target_offset_s == 0.0


def test_build_edge_schedule_sorts_simultaneous_edges_off_before_on():
    scheduler = StimulationController(
        FakeScope(),
        {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 1.0,
                'pulse_width': 500,
                'pulse_count': 1,
            },
            'Green': {
                'enabled': True,
                'illumination': 100,
                'frequency': 2.0,
                'pulse_width': 50,
                'pulse_count': 2,
            },
        },
    )

    matching_edges = [edge for edge in scheduler._edges if abs(edge.target_offset_s - 0.5) < 1e-9]
    assert [edge.action for edge in matching_edges] == ['off', 'on']
    assert [edge.color for edge in matching_edges] == ['Red', 'Green']


def test_invalid_channel_is_skipped_without_aborting_valid_channels():
    scheduler = StimulationController(
        FakeScope(),
        {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 0,
                'pulse_width': 10,
                'pulse_count': 3,
            },
            'Blue': {
                'enabled': True,
                'illumination': 80,
                'frequency': 5.0,
                'pulse_width': 20,
                'pulse_count': 2,
            },
        },
    )

    assert len(scheduler._edges) == 4
    assert all(edge.color == 'Blue' for edge in scheduler._edges)


def test_pulse_width_is_clamped_to_ninety_percent_of_period():
    scheduler = StimulationController(
        FakeScope(),
        {
            'Blue': {
                'enabled': True,
                'illumination': 90,
                'frequency': 10.0,
                'pulse_width': 200,
                'pulse_count': 1,
            }
        },
    )

    on_edge, off_edge = scheduler._edges
    assert on_edge.action == 'on'
    assert off_edge.action == 'off'
    assert off_edge.target_offset_s == pytest.approx(0.09)


def test_scheduler_stop_exits_cleanly_and_turns_off_channels():
    scope = FakeScope()
    scheduler = StimulationController(
        scope,
        {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 100.0,
                'pulse_width': 2,
                'pulse_count': 50,
            },
            'Green': {
                'enabled': True,
                'illumination': 120,
                'frequency': 100.0,
                'pulse_width': 2,
                'pulse_count': 50,
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
    assert ('off', 2, None) in scope.calls
    assert ('off', 1, None) in scope.calls


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
        # Wave 7 Phase 3f: production code now reaches LED methods via
        # `self._scope.illumination.<method>`.
        self.illumination = self
        self.led_lease_owner = None

    # LAYER-F: production code now reads frame validity through the
    # Lumascope API rather than reaching into self.frame_validity
    # directly. Mirror the delegating accessors here.
    def frames_until_valid(self):
        return self.frame_validity.frames_until_valid()

    def count_frame(self):
        self.frame_validity.count_frame()

    def color2ch(self, color):
        return {
            'Blue': 0,
            'Green': 1,
            'Red': 2,
            'BF': 3,
            'PC': 4,
            'DF': 5,
        }[color]

    def led_on_fast(self, channel, mA):
        self.events.append(('on', channel, mA, time.perf_counter()))

    def led_off_fast(self, channel):
        self.events.append(('off', channel, None, time.perf_counter()))

    def led_on(self, channel, mA):
        self.events.append(('on_slow', channel, mA, time.perf_counter()))

    def led_off(self, channel):
        self.events.append(('off_slow', channel, None, time.perf_counter()))


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
            'Green': {
                'enabled': True,
                'illumination': 500,
                'frequency': 0.8,
                'pulse_width': 10,  # ms -- the regime the bug affected
                'pulse_count': 10,
            },
        },
    )

    start_event = threading.Event()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=scheduler.run,
        args=(start_event, stop_event),
        name='stim-scheduler-regression-test',
    )
    thread.start()
    start_event.set()

    # Schedule horizon: 10 pulses at 0.8 Hz = 9 * 1.25 s + 10 ms ~= 11.26 s.
    # Allow a generous join margin.
    thread.join(timeout=15.0)
    assert not thread.is_alive(), 'Scheduler thread did not exit in time'

    green_ch = scope.illumination.color2ch('Green')
    ons = [t for (a, c, _mA, t) in scope.events if a == 'on' and c == green_ch]
    offs = [t for (a, c, _mA, t) in scope.events if a == 'off' and c == green_ch]
    # Scheduler issues a final cleanup `led_off_fast` in its finally block, so
    # OFF count is 10 (scheduled) + 1 (cleanup) = 11. Pair each ON with the
    # first OFF after it.
    assert len(ons) == 10, f'Expected 10 ON edges, got {len(ons)}'
    assert len(offs) >= 10, f'Expected >=10 OFF edges, got {len(offs)}'

    paired_offs = []
    off_iter = iter(offs)
    next_off = next(off_iter, None)
    for on in ons:
        while next_off is not None and next_off < on:
            next_off = next(off_iter, None)
        assert next_off is not None, f'Could not find a matching OFF edge for ON at {on}'
        paired_offs.append(next_off)
        next_off = next(off_iter, None)

    pulse_widths_ms = [(off - on) * 1000.0 for on, off in zip(ons, paired_offs, strict=False)]
    widths = np.asarray(pulse_widths_ms, dtype=float)
    stddev_ms = float(widths.std(ddof=0))
    worst_err_ms = float(np.max(np.abs(widths - 10.0)))
    mean_ms = float(widths.mean())

    # One-run record: printed so the run log shows what this machine measured
    # on the day the test was added (see DAILY_LOG 2026-04-19 / 2026-04-20).
    print(
        f'\n[pulse-width-jitter] n=10 target=10.0 ms '
        f'mean={mean_ms:.3f} ms stddev={stddev_ms:.3f} ms '
        f'worst_err={worst_err_ms:.3f} ms '
        f'widths={[round(w, 3) for w in pulse_widths_ms]}'
    )

    # Thresholds: headline pass line from the post-bench TODO. A GIL-yielding
    # spin produces stddev ~5.9 ms and worst-case ~16 ms, so these catch the
    # regression with ~2x margin on the busy-wait baseline.
    assert stddev_ms < 3.0, (
        f'Pulse-width stddev {stddev_ms:.2f} ms exceeds 3.0 ms -- suggests '
        f"_wait_until's final spin is yielding the GIL (regression of the "
        f'busy-wait fix). Widths: {pulse_widths_ms}'
    )
    assert worst_err_ms < 10.0, (
        f'Worst-case pulse-width error {worst_err_ms:.2f} ms exceeds 10.0 ms '
        f"-- suggests _wait_until's final spin is yielding the GIL (regression "
        f'of the busy-wait fix). Widths: {pulse_widths_ms}'
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

    monkeypatch.setattr('modules.video_capture.threading.Thread', RecordingThread)

    scope = FakeScope()
    step = {
        'Exposure': 10,
        'Auto_Gain': False,
        'Video Config': {'duration': 0.03},
        'Color': 'BF',
        'False_Color': False,
        'Stim_Config': {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 5.0,
                'pulse_width': 20,
                'pulse_count': 3,
            },
            'Green': {
                'enabled': True,
                'illumination': 110,
                'frequency': 5.0,
                'pulse_width': 20,
                'pulse_count': 3,
            },
            'Blue': {
                'enabled': False,
                'illumination': 90,
                'frequency': 5.0,
                'pulse_width': 20,
                'pulse_count': 3,
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
    assert created_threads[0].name == 'stim-scheduler'
    assert created_threads[0].daemon is True, (
        'stim_thread must be daemon=True so app exit reaps it without '
        'hang on an in-flight scheduler iteration (Rule 41 + LVP '
        'f4920c8 daemon-flag fix)'
    )


def test_stim_edge_not_refused_while_protocol_owns_lease():
    """A stim pulse fired during a protocol must reach the hardware.

    The protocol owns the LED lease while a video step captures, so a stim
    edge written with the default empty owner would be refused. The
    controller captures the lease owner at construction and attributes its
    pulses to it, so the edge is allowed.
    """
    from modules.lumascope_api import Lumascope
    from modules.video_capture import StimEdge, StimulationController

    scope = Lumascope(simulate=True)
    scope._led_driver.set_timing_mode('fast')
    scope.illumination.acquire_led_lease('protocol', alive=lambda: True)

    stim_configs = {
        'Blue': {
            'enabled': True,
            'illumination': 50,
            'frequency': 1,
            'pulse_width': 100,
            'pulse_count': 1,
        }
    }
    controller = StimulationController(scope, stim_configs)
    assert controller._lease_owner == 'protocol'

    ch = scope.illumination.color2ch('Blue')
    controller._dispatch_edge(
        StimEdge(target_offset_s=0.0, action='on', channel=ch, mA=50.0, color='Blue')
    )
    assert scope.illumination.led_enabled('Blue'), 'stim pulse was refused under the protocol lease'


def _one_frame_result(stim_end_reason):
    import datetime
    import queue as _queue

    from modules.video_capture import VideoCaptureResult

    frames = _queue.Queue()
    frames.put((np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now()))
    return VideoCaptureResult(
        captured_frames=1,
        calculated_fps=1,
        video_images=frames,
        duration_sec=1.0,
        stim_end_reason=stim_end_reason,
    )


def test_failed_stim_writes_status_sidecar(tmp_path):
    """A stim schedule that ended on a dispatch fault must leave a stim-status
    sidecar next to the saved recording, so the incomplete stimulation is on disk
    rather than the run looking like a normal stim recording."""
    from modules.video_capture import write_video

    write_video(
        result=_one_frame_result('dispatch_error'),
        save_folder=tmp_path,
        name='rec',
        video_as_frames=True,
        step={'Color': 'Blue', 'False_Color': False},
        callbacks={},
        save_encoding='8bit',
        capture_depth=8,
    )
    sidecar = tmp_path / 'rec_stim_status.txt'
    assert sidecar.exists(), 'a failed stim run must leave a status sidecar'
    assert 'dispatch_error' in sidecar.read_text()


def test_clean_and_intentional_stim_runs_write_no_sidecar(tmp_path):
    from modules.video_capture import write_video

    for reason in ('schedule_complete', 'stop_event_set', 'stop_event_set_before_start', None):
        write_video(
            result=_one_frame_result(reason),
            save_folder=tmp_path,
            name=f'rec_{reason}',
            video_as_frames=True,
            step={'Color': 'Blue', 'False_Color': False},
            callbacks={},
            save_encoding='8bit',
            capture_depth=8,
        )
        assert not (tmp_path / f'rec_{reason}_stim_status.txt').exists(), (
            f'end_reason={reason!r} is clean/intentional and must not write a sidecar'
        )


def test_dropped_frames_are_logged_not_a_modal(tmp_path, monkeypatch):
    """Dropped frames during a protocol video are logged, never a modal:
    write_video runs only on the protocol path, and an unattended protocol must
    not pop a non-fatal dialog."""
    import modules.notification_center as nc
    import modules.video_capture as vc
    from modules.video_capture import write_video

    result = _one_frame_result('schedule_complete')
    result.dropped_frames = 3

    notified = []
    monkeypatch.setattr(nc.notifications, 'warning', lambda *a, **k: notified.append(a))
    warnings = []
    monkeypatch.setattr(vc.logger, 'warning', lambda msg, *a, **k: warnings.append(str(msg)))

    write_video(
        result=result,
        save_folder=tmp_path,
        name='rec',
        video_as_frames=True,
        step={'Color': 'Blue', 'False_Color': False},
        callbacks={},
        save_encoding='8bit',
        capture_depth=8,
    )
    assert not notified, 'dropped frames must not pop a modal during a protocol'
    assert any('dropped' in w for w in warnings), 'the dropped-frame count must be logged'


# --- Incomplete-stim classification: the clean state must be earned, not the
# default. Each test below drives one way a schedule can fail to finish and
# asserts it does NOT classify clean. They fail on the prior code, where
# _end_reason defaulted to 'schedule_complete' and was only republished by
# run()'s finally -- so an early return, a wedged thread, a zero-frame
# recording, or a camera-fault stop all read clean and wrote no sidecar.


def _stim_step(duration_sec):
    """A one-channel video step with stim enabled, for capture()-driven tests."""
    return {
        'Exposure': 10,
        'Auto_Gain': False,
        'Video Config': {'duration': duration_sec},
        'Color': 'Red',
        'False_Color': False,
        'Stim_Config': {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 5.0,
                'pulse_width': 20,
                'pulse_count': 5,
            },
        },
    }


def test_empty_schedule_classifies_incomplete_not_clean():
    """Face: an enabled stim that builds zero edges (misconfigured pulse_count)
    returns before run()'s try/finally. It delivers no pulses, so it must not be
    read as a clean schedule_complete."""
    scheduler = StimulationController(
        FakeScope(),
        {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 5.0,
                'pulse_width': 20,
                'pulse_count': 0,  # builds no edges
            }
        },
    )
    assert scheduler._edges == []

    scheduler.run(threading.Event(), threading.Event())

    assert scheduler._end_reason == STIM_END_EMPTY_SCHEDULE
    assert scheduler._end_reason not in _CLEAN_STIM_END_REASONS


def test_camera_fault_stop_classifies_incomplete_not_clean():
    """Face: a camera disconnect stops the schedule via the same stop_event an
    intentional stop uses. The fault is threaded through so the truncated
    schedule classifies incomplete instead of a clean stop_event_set."""
    scope = FakeScope()
    scheduler = StimulationController(
        scope,
        {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 100.0,
                'pulse_width': 2,
                'pulse_count': 200,  # long enough to still be running at stop
            }
        },
    )

    start_event = threading.Event()
    stop_event = threading.Event()
    fault_event = threading.Event()
    thread = threading.Thread(target=scheduler.run, args=(start_event, stop_event, fault_event))
    thread.start()
    start_event.set()
    time.sleep(0.02)
    # Order mirrors capture(): mark the fault, then trip the shared stop.
    fault_event.set()
    stop_event.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert scheduler._end_reason == STIM_END_CAPTURE_FAULT
    assert scheduler._end_reason not in _CLEAN_STIM_END_REASONS


def test_normal_stop_without_fault_stays_clean():
    """Guard for the disambiguation: a stop with no fault (short video / cancel)
    stays clean, so the fault path does not over-flag intentional stops."""
    scope = FakeScope()
    scheduler = StimulationController(
        scope,
        {
            'Red': {
                'enabled': True,
                'illumination': 100,
                'frequency': 100.0,
                'pulse_width': 2,
                'pulse_count': 200,
            }
        },
    )

    start_event = threading.Event()
    stop_event = threading.Event()
    fault_event = threading.Event()
    thread = threading.Thread(target=scheduler.run, args=(start_event, stop_event, fault_event))
    thread.start()
    start_event.set()
    time.sleep(0.02)
    stop_event.set()  # no fault
    thread.join(timeout=2.0)

    assert scheduler._end_reason == STIM_END_STOP_EVENT_SET
    assert scheduler._end_reason in _CLEAN_STIM_END_REASONS


def test_join_timeout_marks_wedged_stim_incomplete(monkeypatch):
    """Face: stim_thread.join can return with the thread still alive (a wedged
    dispatch). run()'s finally never ran, so the reason must be forced to a
    join-timeout marker rather than left at the clean default."""

    class WedgedThread:
        def __init__(self, target, name, args, daemon=False):
            self.target = target
            self.name = name
            self.args = args
            self.daemon = daemon

        def start(self):
            pass  # never runs run(): _end_reason stays the constructor default

        def join(self, timeout=None):
            pass  # returns without the thread having exited

        def is_alive(self):
            return True

    monkeypatch.setattr('modules.video_capture.threading.Thread', WedgedThread)

    session = VideoCaptureSession(
        scope=FakeScope(),
        step=_stim_step(0.03),
        autogain_settings={},
        is_protocol_running_fn=lambda: True,
        callbacks={},
        leds_off_fn=lambda: None,
    )
    result = session.capture()

    assert result is not None
    assert result.stim_end_reason == STIM_END_JOIN_TIMEOUT
    assert result.stim_end_reason not in _CLEAN_STIM_END_REASONS


def test_zero_frame_capture_with_incomplete_stim_writes_sidecar(tmp_path, monkeypatch):
    """Face: a recording that captures zero frames returns before write_video, so
    the sidecar must be written from the capture path. An incomplete stim under a
    frame-less recording still dosed the sample wrong and must not be silent."""

    class IdleThread:
        def __init__(self, target, name, args, daemon=False):
            self.target = target
            self.name = name
            self.args = args
            self.daemon = daemon

        def start(self):
            pass  # never runs run(): _end_reason stays the INCOMPLETE default

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr('modules.video_capture.threading.Thread', IdleThread)

    class DisconnectedScope(FakeScope):
        def get_image(self, force_to_8bit=True, force_new_capture=False):
            # Non-array => camera disconnected => capture loop breaks with zero
            # frames captured.
            return None

    session = VideoCaptureSession(
        scope=DisconnectedScope(),
        step=_stim_step(0.5),
        autogain_settings={},
        is_protocol_running_fn=lambda: True,
        callbacks={},
        leds_off_fn=lambda: None,
        save_folder=tmp_path,
        name='rec',
    )
    result = session.capture()

    assert result is None  # zero frames -> no video result
    sidecar = tmp_path / 'rec_stim_status.txt'
    assert sidecar.exists(), 'a frame-less recording with an incomplete stim must leave a sidecar'


class _LateFinallyScheduler(StimulationController):
    """Simulates the scheduler thread's finally running LATE.

    The capture thread, on a join timeout, classifies the recording JOIN_TIMEOUT.
    The real risk is that the wedged thread then unwedges and its finally
    publishes a clean reason onto the shared _end_reason field BEFORE the capture
    thread reads its result. This subclass reproduces that exact clobber: the
    moment a JOIN_TIMEOUT is written to the field, it overwrites it with a clean
    reason, as the late finally would. The hardened capture path must not read
    this back -- it owns the decision in a local.
    """

    @property
    def _end_reason(self):
        return self.__dict__.get('_end_reason_value', STIM_END_INCOMPLETE)

    @_end_reason.setter
    def _end_reason(self, value):
        if value == STIM_END_JOIN_TIMEOUT:
            # The late finally wins the race and stamps the field clean.
            self.__dict__['_end_reason_value'] = STIM_END_SCHEDULE_COMPLETE
        else:
            self.__dict__['_end_reason_value'] = value


def test_late_scheduler_finally_cannot_clobber_join_timeout(tmp_path, monkeypatch):
    """Race guard: a join-timeout classification owns the recording even if the
    wedged scheduler thread's finally runs late and publishes a clean reason.

    On the unhardened code the capture thread forced JOIN_TIMEOUT onto the shared
    field, then re-read that field for the result -- so a late finally that
    overwrote it with schedule_complete flipped the recording to clean and wrote
    no sidecar, hiding an under-dosed sample. The capture thread must snapshot the
    join-timeout decision into a local and never re-read the cross-thread field.
    """
    from modules.video_capture import write_video

    class WedgedThread:
        def __init__(self, target, name, args, daemon=False):
            self.target = target
            self.name = name
            self.args = args
            self.daemon = daemon

        def start(self):
            pass  # never runs run(): _end_reason stays the constructor default

        def join(self, timeout=None):
            pass  # returns without the thread having exited

        def is_alive(self):
            return True

    monkeypatch.setattr('modules.video_capture.threading.Thread', WedgedThread)
    # The session builds the scheduler internally; swap in the late-clobber
    # variant so any write of JOIN_TIMEOUT to the shared field is overwritten
    # clean, exactly as a late finally on the real thread would.
    monkeypatch.setattr('modules.video_capture.StimulationController', _LateFinallyScheduler)

    session = VideoCaptureSession(
        scope=FakeScope(),
        step=_stim_step(0.03),
        autogain_settings={},
        is_protocol_running_fn=lambda: True,
        callbacks={},
        leds_off_fn=lambda: None,
        save_folder=tmp_path,
        name='rec',
    )
    result = session.capture()

    assert result is not None
    assert result.stim_end_reason == STIM_END_JOIN_TIMEOUT
    assert result.stim_end_reason not in _CLEAN_STIM_END_REASONS

    write_video(
        result=result,
        save_folder=tmp_path,
        name='rec',
        video_as_frames=True,
        step={'Color': 'Red', 'False_Color': False},
        callbacks={},
        save_encoding='8bit',
        capture_depth=8,
    )
    sidecar = tmp_path / 'rec_stim_status.txt'
    assert sidecar.exists(), (
        'a join-timeout recording must leave a sidecar even when the wedged '
        'thread later publishes a clean reason'
    )
    assert STIM_END_JOIN_TIMEOUT in sidecar.read_text()
