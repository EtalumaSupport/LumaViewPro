# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# GUI-agnostic: no Kivy imports.
"""ScopeDisplayThread -- dedicated long-lived thread that paces the
live-display refresh loop.

Replaces the queue-of-1 SequentialIOExecutor pattern that hosted Bug
E's orphan-submit retention. The thread owns the FPS-paced loop;
calls the widget's _render_one_frame(...) per iteration; sleeps via
_stop_event.wait(timeout=...) for responsive shutdown.

Public API:
  start(fps)            -- spawn thread, begin rendering
  stop(timeout)         -- signal stop, join with bound timeout
  pause() / resume()    -- loop continues but skips work; no Thread
                           teardown, no generation bump
  set_fps(fps)          -- runtime FPS-cap change
  update_layer_config(active_layer, open_layer)
                        -- UI-thread publishes Kivy widget state for
                           the next frame
  bump_protocol_hold(hold_seconds)
                        -- protocol-side bumps the hold deadline so
                           the saved frame stays on screen
  add_frame_listener(callback) / remove_frame_listener(callback)
                        -- per-frame fan-out hook for future REST
                           streaming / plugin live-processing.
                           callback(bytes, shape, generation, monotonic_ts)
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any


logger = logging.getLogger('LVP.modules.scope_display_thread')


# Status codes returned by _render_one_frame on the widget. Kept here
# so the widget contract is documented in one place.
STATUS_OK = 0
STATUS_EMPTY = 1  # no new frame in buffer
STATUS_DUPLICATE = 2  # same camera timestamp as last frame
STATUS_NOT_READY = 3  # ctx is None / scope disconnected / similar


# Idle back-off floor for the pacing wait. The loop's only pacing input is the
# FPS cap (min_frame_interval); uncapped (fps=0, the rate Etaluma runs so the
# preview never throttles below the sensor) that interval is 0, so an iteration
# that delivers NO fresh frame (empty buffer, duplicate timestamp, or
# not-ready) re-polls with no wait and pins ~a full core whenever the stream is
# idle, stalled, or between frames. Flooring the wait to this interval on a
# no-delivery iteration bounds that idle poll rate. A delivered frame
# (STATUS_OK) skips the floor entirely, so the achievable frame rate is never
# capped below the camera: the floor is far shorter than any real inter-frame
# gap (the sensor ceiling is ~31 fps = ~32 ms), so a fresh frame is still
# picked up within one floor interval of arriving.
IDLE_BACKOFF_SECONDS = 0.005


class ScopeDisplayThread:
    """Owns the live-display refresh loop.

    Construction takes:
      widget:        the ScopeDisplay widget. The thread calls
                     widget._render_one_frame(...) per iteration; the
                     widget owns the actual rendering state (texture,
                     frame interval history). Thread owns only the
                     loop, pacing, and the volatile publish state.
      ctx_provider:  callable() -> ctx (so we don't capture a stale
                     ref at construction). The thread looks up ctx
                     at each iteration; if ctx is None the loop
                     short-sleeps and retries.
    """

    def __init__(
        self,
        *,
        ctx_provider: Callable[[], Any] | None = None,
    ):
        # The widget (ScopeDisplay) is created by Kivy after this
        # registry runs. The thread looks it up each iteration via
        # ctx_provider().scope_display so we don't have to thread
        # through a setter at app-build time.
        self._ctx_provider = ctx_provider or (lambda: None)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()

        # All UI-published volatile state lives under this lock.
        # The loop reads the snapshot once per iteration; UI publishers
        # write whenever state changes. Lock is uncontended in practice
        # (publish rate ~30Hz, render rate <=30Hz).
        self._config_lock = threading.Lock()
        self._active_layer: str | None = None
        self._open_layer: str | None = None
        self._fps: int = 30
        self._min_frame_interval: float = 1.0 / 30
        self._protocol_hold_until: float = 0.0

        # Wall-clock the loop deliberately spent NOT rendering (protocol /
        # derived-image holds, user pause) since the last _render_one_frame
        # call. Handed to the widget so its timing instruments (slow-frame
        # spike WARN, frame-interval histogram) can exclude intended waits
        # instead of reporting a deliberate on-screen freeze as latency.
        # Measured actual wait, not requested hold -- re-bumped holds overlap.
        # Loop-thread-only state; no lock.
        self._intentional_wait_s: float = 0.0

        # Generation counter increments on start(); resume() does NOT
        # bump by default (so pause/resume preserves the visual frame).
        self._generation: int = 0

        # Frame listeners (REST / plugin fan-out). Snapshotted under
        # lock; called outside the lock so a slow listener can't stall
        # the render loop. Empty by default; zero-cost.
        self._listeners_lock = threading.Lock()
        self._frame_listeners: list[Callable] = []

    # ---- lifecycle ----

    def start(self, fps: int = 30) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.debug(
                'scope_display_thread already running; set_fps + resume instead of re-start'
            )
            self.set_fps(fps)
            self.resume()
            return
        self.set_fps(fps)
        self._stop_event.clear()
        self._paused.clear()
        self._generation += 1
        self._thread = threading.Thread(
            target=self._run_loop,
            name='scope_display_thread',
            daemon=True,
        )
        self._thread.start()
        logger.info(f'scope_display_thread started (fps={fps}, gen={self._generation})')

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        # Wake the loop if it's in a pause-wait or hold-wait.
        self._paused.clear()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(
                    f'scope_display_thread did not join within {timeout}s; '
                    f'daemon=True so process exit will reap it'
                )
        self._thread = None

    def pause(self) -> None:
        """Stop rendering iterations without teardown. Thread stays
        alive; resume() unblocks the loop. Texture freezes on the
        last rendered frame."""
        self._paused.set()

    def resume(self) -> None:
        """Resume rendering iterations after pause()."""
        self._paused.clear()

    # ---- runtime config ----

    def set_fps(self, fps: int) -> None:
        with self._config_lock:
            self._fps = fps
            self._min_frame_interval = 0.0 if fps == 0 else 1.0 / max(1, fps)

    def update_layer_config(
        self,
        active_layer: str | None,
        open_layer: str | None,
    ) -> None:
        with self._config_lock:
            self._active_layer = active_layer
            self._open_layer = open_layer

    def bump_protocol_hold(self, hold_seconds: float) -> None:
        """Set the protocol-hold deadline to monotonic() + hold_seconds.
        Loop honors the deadline and resumes after it expires."""
        with self._config_lock:
            self._protocol_hold_until = time.monotonic() + hold_seconds

    # ---- listeners ----

    def add_frame_listener(self, callback: Callable) -> None:
        """Register a per-frame listener. Called after the texture
        dispatch with (bytes, shape, generation, monotonic_ts).
        Listener exceptions are caught + logged; the loop continues."""
        with self._listeners_lock:
            if callback not in self._frame_listeners:
                self._frame_listeners.append(callback)

    def remove_frame_listener(self, callback: Callable) -> None:
        with self._listeners_lock:
            try:
                self._frame_listeners.remove(callback)
            except ValueError:
                pass

    # ---- inspection ----

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive() and not self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def fps(self) -> int:
        return self._fps

    # ---- loop ----

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            # Pause-wait: short tick so resume is responsive without
            # busy-looping. _stop_event.wait returns True if stop was
            # signalled; break out of pause + loop on that.
            if self._paused.is_set():
                pause_wait_start = time.monotonic()
                if self._stop_event.wait(timeout=0.05):
                    return
                self._intentional_wait_s += time.monotonic() - pause_wait_start
                continue

            # ctx / widget not ready (early boot, between disconnect /
            # reconnect). Short sleep + retry; respect stop signal.
            ctx = self._ctx_provider()
            widget = getattr(ctx, 'scope_display', None) if ctx else None
            if ctx is None or widget is None or getattr(ctx, 'scope', None) is None:
                if self._stop_event.wait(timeout=0.1):
                    return
                continue

            # Snapshot config under lock; release before doing work.
            with self._config_lock:
                active_layer = self._active_layer
                open_layer = self._open_layer
                min_frame_interval = self._min_frame_interval
                hold_until = self._protocol_hold_until

            # DISPLAY-1 protocol hold: if a saved frame should stay on
            # screen, wait until the hold expires. Wait is responsive
            # to stop_event (zero shutdown latency).
            now = time.monotonic()
            if hold_until > now:
                if self._stop_event.wait(timeout=hold_until - now):
                    return
                self._intentional_wait_s += time.monotonic() - now
                continue

            cycle_start = time.monotonic()

            # Drain the intended-wait accumulator into this render call; the
            # wait belongs to exactly one iteration's interval.
            intentional_wait_ms = self._intentional_wait_s * 1000.0
            self._intentional_wait_s = 0.0

            # Render one frame. The widget owns the rendering state;
            # we just call its body.
            try:
                status = widget._render_one_frame(
                    active_layer=active_layer,
                    open_layer=open_layer,
                    dispatch_time=cycle_start,
                    generation=self._generation,
                    intentional_wait_ms=intentional_wait_ms,
                )
            except Exception:
                logger.exception('scope_display_thread iteration error')
                status = STATUS_NOT_READY

            # Fan out to frame listeners on success.
            if status == STATUS_OK:
                self._dispatch_listeners(widget)

            # FPS pace. Event.wait(timeout=) returns False on timeout,
            # True when stop is signalled. Skip the wait entirely if
            # uncapped or already over budget.
            elapsed = time.monotonic() - cycle_start
            wait = max(0.0, min_frame_interval - elapsed)
            # Idle back-off: an iteration that produced no fresh frame floors the
            # wait so an idle/stalled/between-frames stream cannot busy-spin (see
            # IDLE_BACKOFF_SECONDS). STATUS_OK skips this, so a live stream is
            # never paced below the camera rate.
            if status != STATUS_OK:
                wait = max(wait, IDLE_BACKOFF_SECONDS)
            if wait > 0 and self._stop_event.wait(timeout=wait):
                return

        logger.info('scope_display_thread exiting')

    def _dispatch_listeners(self, widget) -> None:
        """Pull the last-rendered bytes/shape from the widget and fan
        out to listeners. Snapshot the listener list under lock so
        registrations during dispatch don't race; call listeners
        outside the lock so a slow listener can't stall others."""
        with self._listeners_lock:
            if not self._frame_listeners:
                return
            listeners = list(self._frame_listeners)
        last = getattr(widget, '_last_rendered_frame', None)
        if last is None:
            return
        data, shape, ts = last
        gen = self._generation
        for cb in listeners:
            try:
                cb(data, shape, gen, ts)
            except Exception:
                logger.exception(f'frame_listener {getattr(cb, "__name__", str(cb))} raised')
