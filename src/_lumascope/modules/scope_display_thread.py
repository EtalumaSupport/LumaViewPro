# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# GUI-agnostic: no Kivy imports.
"""ScopeDisplayThread -- dedicated long-lived thread that paces the
live-display refresh loop.

Replaces a queue-of-1 SequentialIOExecutor, which retained orphaned
submits. The thread owns the FPS-paced loop; calls the widget's
_render_one_frame(...) per iteration; renders a held frame handed to it
before each; and sleeps on one wake event that stop, a held frame and
resume set, so none waits out a pause, a hold or the pacing.

Public API:
  start(renderer, fps)  -- spawn thread, begin rendering through renderer
  stop(timeout)         -- signal stop, join with bound timeout
  pause() / resume()    -- loop continues but skips work; no Thread
                           teardown, no generation bump
  set_fps(fps)          -- runtime FPS-cap change
  update_channel_config(open_channel)
                        -- UI-thread publishes Kivy widget state for
                           the next frame
  bump_protocol_hold(hold_seconds)
                        -- protocol-side bumps the hold deadline so
                           the saved frame stays on screen
  render_held(render, category)
                        -- render a held frame on this thread, the
                           newest only, its failure reported once
  protocol_hold_in_force
                        -- whether that deadline is still ahead; a live
                           blit run during it draws nothing
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from _lumascope.modules.notification_center import notifications


logger = logging.getLogger('LVP.modules.scope_display_thread')


# Status codes returned by _render_one_frame on the widget. Kept here
# so the widget contract is documented in one place.
STATUS_OK = 0
STATUS_EMPTY = 1  # no new frame in buffer
STATUS_DUPLICATE = 2  # same camera timestamp as last frame
STATUS_NOT_READY = 3  # scope disconnected / similar


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

    The renderer it is started with (the ScopeDisplay widget) is called
    as renderer._render_one_frame(...) per iteration; the renderer owns
    the actual rendering state (texture, frame interval history), and
    answers STATUS_NOT_READY while it has nothing to render from. The
    thread owns only the loop, pacing, and the volatile publish state,
    and reads no application context.
    """

    def __init__(self):
        # The widget is created by Kivy after the session builds this
        # thread, so it arrives with start(), the thread's one starter.
        self._renderer: Any = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        # Set by stop(), render_held() and resume(), so every wait of the
        # loop ends at once for any of them.
        self._wake = threading.Event()

        # All UI-published volatile state lives under this lock.
        # The loop reads the snapshot once per iteration; UI publishers
        # write whenever state changes. Lock is uncontended in practice
        # (publish rate ~30Hz, render rate <=30Hz).
        self._config_lock = threading.Lock()
        self._open_channel: str | None = None
        self._fps: int = 30
        self._min_frame_interval: float = 1.0 / 30
        self._protocol_hold_until: float = 0.0
        # The newest held frame's render not yet run, and the category its
        # failure is reported under; a newer one replaces it.
        self._held_render: tuple[Callable[[], None], str] | None = None

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

    # ---- lifecycle ----

    def start(self, renderer: Any, fps: int = 30) -> None:
        """Render through *renderer* at up to *fps* (0 uncapped)."""
        self._renderer = renderer
        if self._thread is not None and self._thread.is_alive():
            # A pause is the user's frozen picture; a second start keeps it.
            logger.debug('scope_display_thread already running; set_fps instead of re-start')
            self.set_fps(fps)
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
        self._wake.set()
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
        # Cleared before the wake, so the woken loop reads it unpaused.
        self._paused.clear()
        self._wake.set()

    # ---- runtime config ----

    def set_fps(self, fps: int) -> None:
        with self._config_lock:
            self._fps = fps
            self._min_frame_interval = 0.0 if fps == 0 else 1.0 / max(1, fps)

    def update_channel_config(self, open_channel: str | None) -> None:
        with self._config_lock:
            self._open_channel = open_channel

    def bump_protocol_hold(self, hold_seconds: float) -> None:
        """Set the protocol-hold deadline to monotonic() + hold_seconds.
        Loop honors the deadline and resumes after it expires."""
        with self._config_lock:
            self._protocol_hold_until = time.monotonic() + hold_seconds

    def render_held(self, render: Callable[[], None], category: str) -> None:
        """Run ``render`` on this thread at its next turn, in place of any not yet run; from any thread.

        A held frame (a run's saved still) is rendered here, not on the
        thread that sends it, so a run's thread hands it over and goes on.
        A sender faster than the renders replaces the one waiting, so only
        the newest is rendered. It runs whether or not the live view is
        paused, and waits for ``start`` when the loop is not running.

        Args:
            render: Renders the frame and posts it to the UI thread.
            category: What a failure of ``render`` is reported as, once and
                unasked, through ``report_outcome``.
        """
        with self._config_lock:
            self._held_render = (render, category)
        self._wake.set()

    # ---- inspection ----

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive() and not self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def protocol_hold_in_force(self) -> bool:
        """Whether a held frame (a saved or derived image) is still due its time on screen."""
        with self._config_lock:
            return self._protocol_hold_until > time.monotonic()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def fps(self) -> int:
        return self._fps

    # ---- loop ----

    def _wait(self, timeout: float | None) -> bool:
        """Wait up to ``timeout`` s (None: until woken), ended early by a wake; True once stopped."""
        self._wake.wait(timeout=timeout)
        self._wake.clear()
        return self._stop_event.is_set()

    def _run_held_render(self) -> None:
        with self._config_lock:
            held, self._held_render = self._held_render, None
        if held is None:
            return
        render, category = held
        try:
            render()
        except Exception as ex:
            notifications.report_outcome(ex, solicited=False, category=category)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._run_held_render()

            # Pause-wait: sleep until resume, a held frame or stop wakes
            # the loop. _wait returns True if stop was signalled; break
            # out of pause + loop on that.
            if self._paused.is_set():
                pause_wait_start = time.monotonic()
                if self._wait(None):
                    return
                self._intentional_wait_s += time.monotonic() - pause_wait_start
                continue

            widget = self._renderer

            # Snapshot config under lock; release before doing work.
            with self._config_lock:
                open_channel = self._open_channel
                min_frame_interval = self._min_frame_interval
                hold_until = self._protocol_hold_until

            # Protocol hold: if a saved frame should stay on screen, wait
            # until the hold expires. A stop ends the wait at once, and a
            # held frame handed over ends it to be rendered.
            now = time.monotonic()
            if hold_until > now:
                if self._wait(hold_until - now):
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
                    open_channel=open_channel,
                    dispatch_time=cycle_start,
                    generation=self._generation,
                    intentional_wait_ms=intentional_wait_ms,
                )
            except Exception:
                logger.exception('scope_display_thread iteration error')
                status = STATUS_NOT_READY

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
            if wait > 0 and self._wait(wait):
                return

        logger.info('scope_display_thread exiting')
