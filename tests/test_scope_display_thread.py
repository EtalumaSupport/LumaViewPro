# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stage B1 regression tests -- ScopeDisplayThread lifecycle, pacing,
publish/read, pause/resume, listener fan-out.

Headless-only (no Kivy). The thread is GUI-agnostic per Rule 15 so
these tests drive the production code path directly.
"""
import threading
import time
from collections import deque

import pytest

from modules import app_context as _app_ctx
from modules.scope_display_thread import (
    ScopeDisplayThread,
    STATUS_OK,
    STATUS_EMPTY,
    STATUS_DUPLICATE,
    STATUS_NOT_READY,
)


class _FakeCtx:
    """Minimal ctx for the thread's ctx_provider lookups."""
    def __init__(self):
        self.scope = object()           # truthy
        self.scope_display = None       # set later
        self.engineering_mode = False


class _FakeWidget:
    """Stand-in for ScopeDisplay. Records calls and lets tests control
    the status code returned by _render_one_frame."""
    def __init__(self, status_sequence=None):
        self.status_sequence = status_sequence or [STATUS_OK]
        self.calls = []                       # (cycle_start, generation)
        self._status_index = 0
        self._last_rendered_frame = (b'fake_bytes', (10, 10), 0.0)
        self._frame_interval_history = deque(maxlen=200)

    def _render_one_frame(self, *, active_layer, active_layer_config,
                          open_layer, dispatch_time, generation):
        self.calls.append({
            'active_layer': active_layer,
            'active_layer_config': active_layer_config,
            'open_layer': open_layer,
            'dispatch_time': dispatch_time,
            'generation': generation,
        })
        if self._status_index < len(self.status_sequence):
            s = self.status_sequence[self._status_index]
            self._status_index += 1
        else:
            s = self.status_sequence[-1]
        if s == STATUS_OK:
            self._last_rendered_frame = (
                b'frame_bytes', (10, 10), time.monotonic(),
            )
        return s


def _make_thread(*, status_sequence=None, fps=30):
    ctx = _FakeCtx()
    widget = _FakeWidget(status_sequence=status_sequence)
    ctx.scope_display = widget
    t = ScopeDisplayThread(ctx_provider=lambda: ctx)
    return t, ctx, widget


def test_thread_starts_and_stops_cleanly():
    t, _, _ = _make_thread()
    t.start(fps=60)
    assert t.is_running
    time.sleep(0.1)
    t.stop(timeout=2.0)
    assert not t.is_running


def test_pause_and_resume_does_not_restart_thread():
    t, _, widget = _make_thread()
    t.start(fps=30)
    time.sleep(0.05)
    initial_gen = t.generation
    thread_id_before = t._thread.ident

    t.pause()
    assert t.is_paused
    time.sleep(0.05)
    calls_at_pause = len(widget.calls)
    time.sleep(0.1)
    # Calls should not increase during pause (within tolerance for
    # an in-flight iteration that started before pause took effect).
    assert len(widget.calls) <= calls_at_pause + 1

    t.resume()
    assert not t.is_paused
    time.sleep(0.1)
    assert len(widget.calls) > calls_at_pause + 1

    # Thread did not respawn; generation did not bump.
    assert t._thread.ident == thread_id_before
    assert t.generation == initial_gen
    t.stop()


def test_set_fps_changes_cadence():
    t, _, widget = _make_thread()
    t.start(fps=10)
    time.sleep(0.3)
    n_at_10fps = len(widget.calls)
    t.set_fps(50)
    time.sleep(0.3)
    n_after = len(widget.calls)
    # At 10 fps we expect ~3 frames in 300ms; at 50 fps the delta in
    # the next 300ms should be substantially higher (>=8 frames). Loose
    # check so CI noise doesn't false-fail.
    delta_fast = n_after - n_at_10fps
    assert n_at_10fps <= 6
    assert delta_fast >= 8
    t.stop()


def test_update_layer_config_publishes_to_loop():
    t, _, widget = _make_thread()
    t.update_layer_config('BF', {'gain': 1.0}, 'BF')
    t.start(fps=60)
    time.sleep(0.1)
    t.stop()
    seen = [c for c in widget.calls if c['active_layer'] == 'BF']
    assert seen, 'thread did not pick up the published layer config'
    assert seen[-1]['active_layer_config'] == {'gain': 1.0}
    assert seen[-1]['open_layer'] == 'BF'


def test_bump_protocol_hold_pauses_rendering():
    t, _, widget = _make_thread()
    t.start(fps=60)
    time.sleep(0.05)
    calls_before_hold = len(widget.calls)
    t.bump_protocol_hold(0.3)
    time.sleep(0.2)
    # Hold still active; few/no new calls.
    delta_during_hold = len(widget.calls) - calls_before_hold
    assert delta_during_hold <= 2, (
        f'expected near-zero calls during hold; got {delta_during_hold}'
    )
    time.sleep(0.2)
    # Hold expired; calls resumed.
    assert len(widget.calls) - calls_before_hold > 2
    t.stop()


def test_stop_during_long_hold_returns_within_timeout():
    t, _, _ = _make_thread()
    t.start(fps=60)
    t.bump_protocol_hold(5.0)
    time.sleep(0.05)
    t0 = time.monotonic()
    t.stop(timeout=0.5)
    elapsed = time.monotonic() - t0
    # Event.wait(timeout=) returns immediately on stop_event set;
    # bound generous to absorb thread scheduling jitter.
    assert elapsed < 0.5, f'stop took {elapsed:.3f}s during hold'
    assert not t.is_running


def test_generation_counter_increments_on_start():
    t, _, _ = _make_thread()
    assert t.generation == 0
    t.start(fps=30)
    g1 = t.generation
    assert g1 == 1
    t.stop()
    t.start(fps=30)
    g2 = t.generation
    assert g2 == g1 + 1
    t.stop()


def test_layer_config_thread_safe_under_high_publish_rate():
    t, _, widget = _make_thread()
    t.start(fps=30)
    stop = threading.Event()

    def publisher():
        i = 0
        while not stop.is_set():
            t.update_layer_config(f'L{i}', {'i': i}, f'L{i}')
            i += 1

    th = threading.Thread(target=publisher, daemon=True)
    th.start()
    time.sleep(0.3)
    stop.set()
    th.join(timeout=1.0)
    t.stop()
    # Just verify no exceptions / torn reads -- presence of any call
    # means the lock didn't deadlock.
    assert widget.calls, 'thread ran zero iterations under load'


def test_add_frame_listener_called_per_frame():
    t, _, _ = _make_thread(status_sequence=[STATUS_OK] * 10)
    received = []

    def listener(data, shape, gen, ts):
        received.append((data, shape, gen, ts))

    t.add_frame_listener(listener)
    t.start(fps=60)
    time.sleep(0.2)
    t.stop()
    assert received, 'listener was never called'
    data, shape, gen, ts = received[0]
    assert data == b'frame_bytes'
    assert shape == (10, 10)
    assert gen == t.generation


def test_remove_frame_listener_stops_calls():
    t, _, _ = _make_thread(status_sequence=[STATUS_OK] * 50)
    received = []

    def listener(data, shape, gen, ts):
        received.append(ts)

    t.add_frame_listener(listener)
    t.start(fps=60)
    time.sleep(0.1)
    n_with_listener = len(received)
    t.remove_frame_listener(listener)
    time.sleep(0.2)
    # After removal, no further calls.
    assert len(received) == n_with_listener
    t.stop()


def test_status_not_ok_does_not_fan_out_to_listeners():
    t, _, _ = _make_thread(status_sequence=[STATUS_EMPTY] * 10)
    received = []
    t.add_frame_listener(lambda *a: received.append(a))
    t.start(fps=60)
    time.sleep(0.15)
    t.stop()
    assert received == [], (
        'listeners must only fire on STATUS_OK; '
        f'got {len(received)} calls on STATUS_EMPTY'
    )


def test_widget_unavailable_loop_retries_without_crash():
    ctx = _FakeCtx()
    ctx.scope_display = None   # widget not built yet
    t = ScopeDisplayThread(ctx_provider=lambda: ctx)
    t.start(fps=30)
    time.sleep(0.2)
    # No crash; loop kept retrying. Now wire widget up.
    widget = _FakeWidget()
    ctx.scope_display = widget
    time.sleep(0.1)
    t.stop()
    assert widget.calls, 'thread did not pick up widget after late wiring'


def test_widget_start_delegate_when_ctx_wired_runs_thread():
    """Mirror the lumaviewpro.build() start site: with both
    ctx.scope_display_thread and ctx.settings populated, the delegate
    used by ui/scope_display.py:start() must successfully start the
    thread. Regression for the kv-construction-time start race where
    ScopeDisplay.__init__ called self.start() before ExecutorRegistry
    created the thread, so getattr(ctx, 'scope_display_thread', None)
    returned None and the delegate silently no-opped."""
    saved_ctx = _app_ctx.ctx
    try:
        t, fake_ctx, _widget = _make_thread()
        fake_ctx.scope_display_thread = t
        fake_ctx.settings = {'live_view_fps': 30}
        _app_ctx.ctx = fake_ctx

        # Mirror ui/scope_display.py:start() body without importing Kivy.
        ctx_now = _app_ctx.ctx
        fps = ctx_now.settings['live_view_fps']
        thread = getattr(ctx_now, 'scope_display_thread', None)
        assert thread is t, (
            'scope_display_thread missing from ctx; lumaviewpro.build() '
            'must wire it before invoking widget.start()'
        )
        thread.start(fps=fps)
        try:
            assert t.is_running, (
                'thread.start() did not actually start the worker; '
                'this would manifest as silent live-preview death at bench'
            )
        finally:
            t.stop(timeout=2.0)
    finally:
        _app_ctx.ctx = saved_ctx


def test_widget_start_delegate_silently_noops_when_thread_missing():
    """Documents the defensive guard in ui/scope_display.py:start() that
    handles the case where ctx is None or ctx.scope_display_thread is
    not yet wired. Other early-call sites depend on this no-op behavior;
    if it ever raises, the new lumaviewpro.build() start contract breaks
    for those sites."""
    saved_ctx = _app_ctx.ctx
    try:
        # Case 1: ctx is None entirely
        _app_ctx.ctx = None
        ctx_now = _app_ctx.ctx
        thread = getattr(ctx_now, 'scope_display_thread', None) if ctx_now else None
        assert thread is None
        if thread is not None:           # mirrors widget code
            thread.start(fps=30)         # never reached

        # Case 2: ctx exists but scope_display_thread field absent
        _app_ctx.ctx = _FakeCtx()        # no scope_display_thread attribute
        ctx_now = _app_ctx.ctx
        thread = getattr(ctx_now, 'scope_display_thread', None) if ctx_now else None
        assert thread is None
        if thread is not None:
            thread.start(fps=30)         # never reached
    finally:
        _app_ctx.ctx = saved_ctx
