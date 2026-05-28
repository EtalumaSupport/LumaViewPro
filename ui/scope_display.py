# Copyright Etaluma, Inc.
"""
ScopeDisplay -- pull-based image display loop.

Image Pipeline (sensor -> screen):
  1. Camera SDK callback -> ImageHandler._store_frame()     [1 copy: SDK buffer -> numpy]
  2. grab_latest() -> returns stored reference               [0 copies]
  3. get_image_from_buffer():
     - scale bar overlay (in-place on the reference)        [0 copies]
     - 12->8 bit LUT conversion (if force_to_8bit)           [1 copy: LUT indexing]
  4. Worker thread: contrast stretch / bullseye LUT          [1 copy: LUT indexing]
  5. image.tobytes() -> blit_buffer() to GPU texture          [1 copy: tobytes]

Copy budget:
  8-bit path:  SDK(1) + tobytes(1)                    = 2 copies
  12-bit path: SDK(1) + 12->8 LUT(1) + tobytes(1)     = 3 copies

Threading model (Stage B1):
  - Main thread (Kivy): create_and_set_texture() / create_and_set_bullseye_texture()
    blit textures dispatched from worker via Clock.schedule_once.
  - scope_display_thread (modules/scope_display_thread.py): owns the
    FPS-paced loop; calls _render_one_frame(...) per iteration.
  - Generation counter (owned by the thread; mirrored via _current_generation())
    prevents stale callbacks after stop()/start() cycles.
  - FPS pacing lives on the thread (Event.wait(timeout=...) per iteration).
"""

import logging
import threading
import time

import numpy as np
import skimage.draw

from kivy.clock import Clock
from kivy.graphics import InstructionGroup, Color, Line, Ellipse
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import BooleanProperty
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.input import MotionEvent

from modules.contrast_stretcher import ContrastStretcher
from modules import gui_logger
import modules.autofocus_functions as autofocus_functions
import modules.common_utils as common_utils
import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.scope_display')

# --- Display constants ---
BULLSEYE_FPS_CAP = 15  # Max FPS for CPU-intensive bullseye LUT rendering
VALIDITY_DOT_RADIUS = 10  # Engineering-mode validity indicator dot radius (px)
VALIDITY_DOT_MARGIN = 20  # Margin from image edge to dot center (px)


class ScopeDisplay(Image):
    play = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(ScopeDisplay, self).__init__(**kwargs)
        logger.debug('[LVP Main  ] ScopeDisplay.__init__()')
        self.play = True
        # paused / _display_running / _display_generation / _min_frame_interval /
        # _protocol_hold_until / _cycle_start_time all retired from this widget.
        # Lifecycle + pacing state now owned by ScopeDisplayThread
        # (modules/scope_display_thread.py). The widget keeps only render-side
        # state (texture cache, _last_rendered_frame for listener fan-out,
        # frame-interval history for metrics).

        self.use_bullseye = False
        self.use_crosshairs = False
        self.use_live_image_histogram_equalization = False
        self.camera_disconnected_display_set = False

        self._bullseye_rgb_buf = None
        self._bullseye_buf_shape = None

        # Reusable 8-bit LUT destination for the preview 12->8 conversion;
        # (re)allocated lazily to match the frame in _render_one_frame.
        self._display_8bit_buf = None

        # FPS tracking -- capture thread (frames grabbed from camera)
        self._capture_fps_count = 0
        self._capture_fps_last_time = time.monotonic()
        self._capture_fps_value = 0.0

        # Display FPS tracking -- main thread (frames actually rendered on screen)
        self._display_fps_count = 0
        self._display_fps_last_time = time.monotonic()
        self._display_fps_value = 0.0

        # Camera data rate (MB/s) -- computed from capture FPS and frame size
        self._camera_mbps = 0.0
        self._last_frame_nbytes = 0

        # Frame-interval rolling histogram for P50/P95/P99. Sized to ~60 s
        # at typical 15-30 fps. Worker thread appends; metrics-log thread
        # reads via `frame_interval_percentiles_ms()`. deque.append is
        # atomic in CPython so no lock needed for occasional snapshot reads.
        from collections import deque

        self._frame_interval_history = deque(maxlen=2000)
        self._last_frame_pull_time = None

        # Engineering stats timing (2x per second)
        self._eng_stats_last_time = 0.0

        # Performance instrumentation (enabled via settings.debug_mode)
        self._perf_log_interval = 5.0  # seconds between perf reports
        self._perf_log_last_time = 0.0
        self._perf_grab_times = []
        self._perf_process_times = []
        self._perf_blit_schedule_times = []
        self._perf_blit_delays = []
        self._debug_perf = None  # lazy-resolved from settings.debug_mode on first frame

        # Bullseye frame rate cap (15 FPS -- CPU-intensive LUT rendering)
        self._bullseye_min_interval = 1.0 / BULLSEYE_FPS_CAP
        self._bullseye_last_time = 0.0

        self._contrast_stretcher = ContrastStretcher(
            window_len=3,
            bottom_pct=0.3,
            top_pct=0.3,
        )

        self.use_full_pixel_depth = False

        # Counters (were module-level globals in lumaviewpro.py)
        self._debug_counter = 0
        self._display_update_counter = 0

        # Display loop state owned by ScopeDisplayThread. Widget keeps:
        self._last_frame_ts = None  # camera timestamp of last displayed frame (dup check)
        self._last_rendered_frame = (
            None  # (bytes, shape, monotonic_ts) for thread.add_frame_listener fan-out
        )
        self._PROTOCOL_HOLD_MS = (
            500  # hold deadline duration (bumped via thread.bump_protocol_hold)
        )

        # Crosshair canvas overlay (drawn on top of texture, not into pixels)
        self._crosshair_group = InstructionGroup()
        self.canvas.after.add(self._crosshair_group)
        self._crosshair_visible = False

        # Frame validity indicator (engineering mode only)
        # Green dot = frame is valid, red dot = settling after hardware change
        self._validity_group = InstructionGroup()
        self.canvas.after.add(self._validity_group)
        self._validity_dot_visible = False

        self.bind(
            size=self._on_size_changed, pos=self._on_size_changed, texture=self._on_size_changed
        )

        # Create a black texture to avoid white flash on startup
        self._create_default_black_texture()

        # Display thread start happens from lumaviewpro.py:build()
        # after ctx is fully wired. Starting from __init__ runs during
        # kv tree construction, before ctx.scope_display_thread exists,
        # so the delegate inside self.start() silently no-ops.

    def _create_default_black_texture(self):
        """Create a default black texture to display before camera feed starts."""
        # Create a small black image (will be stretched to fit)
        black_image = np.zeros((100, 100), dtype=np.uint8)
        texture = Texture.create(
            size=(black_image.shape[1], black_image.shape[0]), colorfmt='luminance'
        )
        texture.blit_buffer(black_image.tobytes(), colorfmt='luminance', bufferfmt='ubyte')
        self.texture = texture

    def _on_size_changed(self, *args):
        """Rebuild crosshair overlay when widget size or position changes."""
        if self._crosshair_visible:
            self._build_crosshair_overlay()

    def _get_displayed_image_bounds(self):
        """Compute the actual displayed image rectangle within the widget.

        With fit_mode='contain', the image is scaled to fit while maintaining
        aspect ratio. Returns (cx, cy, img_w, img_h) where cx/cy is the center
        and img_w/img_h is the displayed size in widget pixels.
        """
        norm_w, norm_h = self.norm_image_size
        cx = self.center_x
        cy = self.center_y
        return cx, cy, norm_w, norm_h

    def _build_crosshair_overlay(self):
        """Rebuild the crosshair canvas instructions to match current layout."""
        self._crosshair_group.clear()

        cx, cy, img_w, img_h = self._get_displayed_image_bounds()
        if img_w < 1 or img_h < 1:
            return

        min_dim = min(img_w, img_h)
        line_width = dp(1)

        # Semi-transparent white
        self._crosshair_group.add(Color(1, 1, 1, 0.6))

        # Vertical center line (full height of displayed image)
        self._crosshair_group.add(
            Line(
                points=[cx, cy - img_h / 2, cx, cy + img_h / 2],
                width=line_width,
            )
        )

        # Horizontal center line (full width of displayed image)
        self._crosshair_group.add(
            Line(
                points=[cx - img_w / 2, cy, cx + img_w / 2, cy],
                width=line_width,
            )
        )

        # 4 radiating circles, evenly spaced across half the minimum dimension
        num_circles = 4
        circle_spacing = min_dim / 2 / num_circles
        for i in range(num_circles):
            radius = (i + 1) * circle_spacing
            self._crosshair_group.add(
                Line(
                    circle=(cx, cy, radius),
                    width=line_width,
                )
            )

    def show_crosshairs(self, show):
        """Show or hide the crosshair overlay."""
        self._crosshair_visible = show
        if show:
            self._build_crosshair_overlay()
        else:
            self._crosshair_group.clear()

    def _update_validity_dot(self, is_valid: bool):
        """Draw a small green (valid) or red (settling) dot in the top-right corner.

        Only visible in engineering mode. Called from the main thread via
        Clock.schedule_once after each frame grab.
        """
        self._validity_group.clear()
        ctx = _app_ctx.ctx
        if not ctx.engineering_mode:
            return

        center_x, center_y, img_w, img_h = self._get_displayed_image_bounds()
        if img_w < 1 or img_h < 1:
            return

        dot_radius = VALIDITY_DOT_RADIUS
        margin = VALIDITY_DOT_MARGIN
        # Top-right corner of the displayed image
        cx = center_x + img_w / 2 - margin
        cy = center_y + img_h / 2 - margin

        if is_valid:
            self._validity_group.add(Color(0, 1, 0, 0.8))  # green
        else:
            self._validity_group.add(Color(1, 0, 0, 0.8))  # red
        self._validity_group.add(
            Ellipse(
                pos=(cx - dot_radius, cy - dot_radius),
                size=(dot_radius * 2, dot_radius * 2),
            )
        )

    def start(self, fps=None):
        logger.info('[LVP Main  ] ScopeDisplay.start()')
        ctx = _app_ctx.ctx
        if fps is not None:
            self.fps = fps
        elif ctx is not None and 'live_view_fps' in ctx.settings:
            self.fps = ctx.settings['live_view_fps']
        else:
            self.fps = 30
        thread = getattr(ctx, 'scope_display_thread', None) if ctx else None
        if thread is not None:
            thread.start(fps=self.fps)
            fps_label = 'uncapped' if self.fps == 0 else f'{self.fps} FPS cap'
            logger.info(f'[LVP Main  ] ScopeDisplay: thread started ({fps_label})')

    def stop(self):
        ctx = _app_ctx.ctx
        thread = getattr(ctx, 'scope_display_thread', None) if ctx else None
        if thread is not None:
            thread.stop()
        logger.info('[LVP Main  ] ScopeDisplay.stop()')

    def pause(self):
        """Pause rendering without tearing down the thread. The
        last-rendered frame stays on screen. Use resume() to continue.
        cam_toggle button (ui/main_display.py) uses pause/resume so
        the thread stays alive and the texture freezes."""
        ctx = _app_ctx.ctx
        thread = getattr(ctx, 'scope_display_thread', None) if ctx else None
        if thread is not None:
            thread.pause()

    def resume(self):
        ctx = _app_ctx.ctx
        thread = getattr(ctx, 'scope_display_thread', None) if ctx else None
        if thread is not None:
            thread.resume()

    def touch(self, target: Widget, event: MotionEvent):
        if event.is_touch and (event.device == 'mouse') and (event.button == 'right'):
            norm_texture_width, norm_texture_height = self.norm_image_size
            norm_texture_x_min = self.center_x - norm_texture_width / 2
            norm_texture_x_max = self.center_x + norm_texture_width / 2
            norm_texture_y_min = self.center_y - norm_texture_height / 2
            norm_texture_y_max = self.center_y + norm_texture_height / 2

            click_pos_x = event.pos[0]
            click_pos_y = event.pos[1]

            # Check if click occurred within texture
            if (
                (click_pos_x >= norm_texture_x_min)
                and (click_pos_x <= norm_texture_x_max)
                and (click_pos_y >= norm_texture_y_min)
                and (click_pos_y <= norm_texture_y_max)
            ):
                norm_texture_click_pos_x = click_pos_x - norm_texture_x_min
                norm_texture_click_pos_y = click_pos_y - norm_texture_y_min
                texture_width, texture_height = self.texture_size

                # Scale to image pixels
                texture_click_pos_x = norm_texture_click_pos_x * texture_width / norm_texture_width
                texture_click_pos_y = (
                    norm_texture_click_pos_y * texture_height / norm_texture_height
                )

                # Distance from center
                x_dist_pixel = (
                    texture_click_pos_x - texture_width / 2
                )  # Positive means to the right of center
                y_dist_pixel = (
                    texture_click_pos_y - texture_height / 2
                )  # Positive means above center

                from modules.config_ui_getters import (
                    get_current_objective_info,
                    get_binning_from_ui,
                )
                from ui.ui_helpers import move_relative_position

                _, objective = get_current_objective_info()
                pixel_size_um = common_utils.get_pixel_size(
                    focal_length=objective['focal_length'],
                    binning_size=get_binning_from_ui(),
                )

                x_dist_um = x_dist_pixel * pixel_size_um
                y_dist_um = y_dist_pixel * pixel_size_um

                gui_logger.button(
                    'SCOPE_CLICK_TO_CENTER',
                    f'dx_um={x_dist_um:.1f} dy_um={y_dist_um:.1f} '
                    f'pixel_um={pixel_size_um:.3f}',
                )
                move_relative_position(axis='X', um=x_dist_um)
                move_relative_position(axis='Y', um=y_dist_um)

    @staticmethod
    def add_crosshairs(image):
        height, width = image.shape[0], image.shape[1]

        if image.ndim == 3:
            is_color = True
        else:
            is_color = False

        center_x = round(width / 2)
        center_y = round(height / 2)

        # Crosshairs - 2 pixels wide
        if is_color:
            image[:, center_x - 1 : center_x + 1, :] = 255
            image[center_y - 1 : center_y + 1, :, :] = 255
        else:
            image[:, center_x - 1 : center_x + 1] = 255
            image[center_y - 1 : center_y + 1, :] = 255

        # Radiating circles
        num_circles = 4
        minimum_dimension = min(height, width)
        circle_spacing = round(minimum_dimension / 2 / num_circles)
        for i in range(num_circles):
            radius = (i + 1) * circle_spacing
            rr, cc = skimage.draw.circle_perimeter(
                center_y, center_x, radius=radius, shape=image.shape
            )
            image[rr, cc] = 255

            # To make circles 2 pixel wide...
            rr, cc = skimage.draw.circle_perimeter(
                center_y, center_x, radius=radius + 1, shape=image.shape
            )
            image[rr, cc] = 255

        return image

    # Pre-built 256-entry LUT for bullseye color mapping (built once, used every frame)
    _bullseye_lut = None

    @staticmethod
    def _build_bullseye_lut():
        """Build a 256x3 uint8 lookup table for the bullseye color map."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        # Pattern: 10-pixel-wide bands alternating black/green,
        # with blue at 125-135 and red at 245-255
        color_bands = [
            # (start_exclusive, end_inclusive, R, G, B)
            (5, 15, 0, 255, 0),
            (25, 35, 0, 255, 0),
            (45, 55, 0, 255, 0),
            (65, 75, 0, 255, 0),
            (85, 95, 0, 255, 0),
            (105, 115, 0, 255, 0),
            (125, 135, 0, 0, 255),
            (145, 155, 0, 255, 0),
            (165, 175, 0, 255, 0),
            (185, 195, 0, 255, 0),
            (205, 215, 0, 255, 0),
            (225, 235, 0, 255, 0),
            (245, 255, 255, 0, 0),
        ]
        for start, end, r, g, b in color_bands:
            lut[start + 1 : end + 1] = [r, g, b]
        return lut

    @staticmethod
    def transform_to_bullseye(image):
        if ScopeDisplay._bullseye_lut is None:
            ScopeDisplay._bullseye_lut = ScopeDisplay._build_bullseye_lut()
        return ScopeDisplay._bullseye_lut[image]

    def transform_to_bullseye_prealloc(self, image):
        if ScopeDisplay._bullseye_lut is None:
            ScopeDisplay._bullseye_lut = ScopeDisplay._build_bullseye_lut()
        target_shape = image.shape + (3,)
        if self._bullseye_rgb_buf is None or self._bullseye_buf_shape != image.shape:
            self._bullseye_rgb_buf = np.empty(target_shape, dtype=np.uint8)
            self._bullseye_buf_shape = image.shape
        np.take(ScopeDisplay._bullseye_lut, image, axis=0, out=self._bullseye_rgb_buf)
        return self._bullseye_rgb_buf

    def frame_interval_percentiles_ms(self):
        """Return P50/P95/P99 frame interval in ms over the rolling history.

        Used by log_system_metrics() to detect consumer stalls. Returns dict
        with keys p50/p95/p99/n; empty dict if no samples yet.
        """
        history = list(self._frame_interval_history)
        n = len(history)
        if n == 0:
            return {}
        history.sort()
        return {
            'p50': history[n // 2],
            'p95': history[min(n - 1, int(n * 0.95))],
            'p99': history[min(n - 1, int(n * 0.99))],
            'max': history[-1],
            'n': n,
        }

    # _pull_next_frame, update_scopedisplay, _schedule_next retired in Stage B1.
    # The dedicated scope_display_thread owns the FPS-paced loop; this widget
    # provides _render_one_frame as the loop body (one iteration = one frame).
    # FPS pacing, generation, hold-deadline all live on the thread.
    pass

    @staticmethod
    def _eng_stats_due(open_layer, use_bullseye, now, last_time, interval=0.5):
        """Whether engineering stats should be computed this frame.

        open_layer is None when every layer accordion is collapsed, so the
        stats are off-screen. Gate the whole compute on it -- not just the UI
        dispatch -- so mean / std / focus do not run on hidden frames.
        """
        if open_layer is None or use_bullseye:
            return False
        return now - last_time >= interval

    @staticmethod
    def _focus_score_enabled(ctx):
        """Whether the per-frame Vollath focus score is enabled.

        Read live each frame so the engineering-tab toggle takes effect
        without a restart; a value cached on the first frame would freeze at
        whatever the setting was when the first frame arrived.
        """
        return bool(ctx is not None and ctx.settings.get('focus_score_enabled', False))

    def set_engineering_ui(self, mean, stddev, af_score, open_layer):
        ctx = _app_ctx.ctx
        open_layer_obj = ctx.image_settings.layer_lookup(layer=open_layer)
        new_mean_text = f'Mean: {mean}'
        if open_layer_obj.ids['image_stats_mean_id'].text != new_mean_text:
            open_layer_obj.ids['image_stats_mean_id'].text = new_mean_text
        new_stddev_text = f'StdDev: {stddev}'
        if open_layer_obj.ids['image_stats_stddev_id'].text != new_stddev_text:
            open_layer_obj.ids['image_stats_stddev_id'].text = new_stddev_text
        new_af_text = f'AF Score: {af_score}'
        if open_layer_obj.ids['image_af_score_id'].text != new_af_text:
            open_layer_obj.ids['image_af_score_id'].text = new_af_text

    def set_camera_disconnected_display(self):
        self.source = './data/icons/camera_to_USB.png'
        self.camera_disconnected_display_set = True
        # Drop the bullseye RGB scratch buffer so a reconnect at a
        # different camera resolution doesn't retain the old allocation
        # (swapping 2K->4K->2K otherwise leaks ~60 MB per cycle).
        self._bullseye_rgb_buf = None
        self._bullseye_buf_shape = None
        self._display_8bit_buf = None
        return

    def source_clear(self):
        self.source = ''
        self.camera_disconnected_display_set = False
        return

    def _increment_display_counter(self, dt=None):
        """Increment display update counter on main thread."""
        self._display_update_counter += 1

    def _reset_display_counter(self, dt=None):
        """Reset display update counter on main thread."""
        self._display_update_counter = 0

    def _increment_debug_counter(self, dt=None):
        """Increment debug counter on main thread."""
        self._debug_counter += 1
        if self._debug_counter == 30:
            self._debug_counter = 0

    def _render_one_frame(
        self, *, active_layer, active_layer_config, open_layer, dispatch_time=0, generation=0
    ):
        """Render one display frame. Called by ScopeDisplayThread per iteration.

        Returns a status code from scope_display_thread (STATUS_OK /
        STATUS_EMPTY / STATUS_DUPLICATE / STATUS_NOT_READY); the loop uses
        the status to decide whether to fan out to frame listeners and how
        to pace the next iteration. Self-rearming via Clock.schedule_once
        is RETIRED -- the loop owns pacing now.
        """
        from modules.scope_display_thread import (
            STATUS_OK,
            STATUS_EMPTY,
            STATUS_DUPLICATE,
            STATUS_NOT_READY,
        )

        ctx = _app_ctx.ctx

        # SHUTDOWN-RACE-1: thread can dequeue here after ctx / ctx.scope
        # has been torn down. Early-return rather than NPE through
        # scope.camera_connected.
        if ctx is None or ctx.scope is None:
            return STATUS_NOT_READY

        # Frame-interval recording (was on _pull_next_frame; now per-iteration here).
        cycle_start = dispatch_time or time.monotonic()
        if self._last_frame_pull_time is not None:
            interval_ms = (cycle_start - self._last_frame_pull_time) * 1000.0
            self._frame_interval_history.append(interval_ms)
        self._last_frame_pull_time = cycle_start

        t_worker_start = cycle_start
        t_queue_wait = 0  # No queue under B1; preserve var for downstream perf code.

        # Snapshot counter value before scheduling increment on main thread
        display_counter = self._display_update_counter + 1
        Clock.schedule_once(self._increment_display_counter, 0)

        if not ctx.scope.camera_connected:
            if not self.camera_disconnected_display_set:
                Clock.schedule_once(lambda dt: self.set_camera_disconnected_display(), 0)
            return STATUS_NOT_READY

        if self.camera_disconnected_display_set:
            Clock.schedule_once(lambda dt: self.source_clear(), 0)

        # Update scale bar color based on active channel (black for transmitted, white for fluorescence)
        if active_layer is not None:
            ctx.scope.imaging.set_scale_bar(
                enabled=ctx.scope.imaging.scale_bar_enabled, color=active_layer
            )

        # Likely not an IO call as image will be stored in buffer
        t_grab_start = time.monotonic()
        # Reuse one 8-bit LUT buffer across frames so the 12->8 conversion
        # in get_image_from_buffer does not allocate a fresh ~W*H array
        # every frame on the 30 fps preview. tobytes() below copies before
        # the next frame overwrites it, so a single slot is safe (same
        # pattern as the bullseye buffer). Only this preview thread owns
        # and passes this buffer; the histogram (main thread) passes none.
        image, frame_ts = ctx.scope.imaging.get_image_from_buffer(
            force_to_8bit=True, out_8bit=self._display_8bit_buf
        )
        if image is None or image.size == 0:
            return STATUS_EMPTY

        # (Re)allocate the reusable buffer to match the frame so the NEXT
        # frame's conversion writes into it. The 8-bit camera path returns
        # its own buffer and never uses this; the cost is one idle buffer.
        if image.ndim == 2 and image.dtype == np.uint8 and (
            self._display_8bit_buf is None or self._display_8bit_buf.shape != image.shape
        ):
            self._display_8bit_buf = np.empty(image.shape, dtype=np.uint8)

        # Skip duplicate frames (same camera timestamp = same data)
        if frame_ts is not None and frame_ts == self._last_frame_ts:
            return STATUS_DUPLICATE
        self._last_frame_ts = frame_ts
        t_grab_end = time.monotonic()

        # Record queue wait for perf logging (settings.debug_mode only).
        # On the very first frame _debug_perf is None (resolved below); we
        # miss one queue-wait sample, which is irrelevant given the 5-second
        # log window.
        if self._debug_perf:
            self._perf_blit_schedule_times.append(t_queue_wait)

        # Capture FPS tracking + camera data rate
        # Use raw camera frame size (before 12->8 bit conversion) so the
        # displayed data rate reflects actual camera throughput, not the
        # post-conversion display throughput.
        self._capture_fps_count += 1
        fs = ctx.scope.imaging.camera_frame_size
        pixel_format = ctx.scope.imaging.camera_pixel_format
        bpp = 2 if pixel_format in ('Mono10', 'Mono10g40IDS', 'Mono12', 'Mono12g24IDS') else 1
        self._last_frame_nbytes = fs.get('width', 0) * fs.get('height', 0) * bpp
        now = time.monotonic()
        elapsed = now - self._capture_fps_last_time
        if elapsed >= 1.0:
            self._capture_fps_value = self._capture_fps_count / elapsed
            # EMA smoothing (alpha=0.3) -- without this the title bar bounces noisily
            # because each 1-second window gets a fresh hard-assigned value
            # (85 / 120 / 95 / 110 / 88 MB/s during a steady capture). EMA
            # converges to the real average over 3-4 seconds.
            new_mbps = (self._capture_fps_value * self._last_frame_nbytes) / (1024 * 1024)
            self._camera_mbps = 0.3 * new_mbps + 0.7 * self._camera_mbps
            self._capture_fps_count = 0
            self._capture_fps_last_time = now

        if display_counter % 10 == 0:
            Clock.schedule_once(self._reset_display_counter, 0)

            if active_layer_config is not None and active_layer_config['auto_gain']:
                from modules.sequential_io_executor import IOTask

                ctx.camera_executor.put(IOTask(action=self.get_true_gain_exp, args=(active_layer,)))

        t_eng_stats = 0
        if ctx.engineering_mode:
            # Frame validity indicator: update every frame (lightweight canvas op)
            fv_valid = ctx.scope.imaging.frame_is_valid
            Clock.schedule_once(lambda dt, v=fv_valid: self._update_validity_dot(v), 0)

            # Engineering stats: 2x per second (time-based, not frame-based)
            now_eng = time.monotonic()
            if self._eng_stats_due(
                open_layer, self.use_bullseye, now_eng, self._eng_stats_last_time
            ):
                self._eng_stats_last_time = now_eng
                t_eng_start = time.monotonic()
                mean = round(np.mean(a=image), 2)
                stddev = round(np.std(a=image), 2)
                # The Vollath focus score is the costly per-frame stat; the
                # engineering-tab "Focus Score" toggle suppresses it. Mean and
                # std stay (cheap). Display-only -- autofocus scores its own
                # frames independently and is unaffected.
                if self._focus_score_enabled(ctx):
                    af_score = autofocus_functions.focus_function(
                        image=image, skip_score_logging=True
                    )
                else:
                    af_score = 'off'
                t_eng_stats = time.monotonic() - t_eng_start

                Clock.schedule_once(
                    lambda dt: self.set_engineering_ui(mean, stddev, af_score, open_layer), 0
                )

        if self.use_bullseye:
            now_be = time.monotonic()
            if now_be - self._bullseye_last_time >= self._bullseye_min_interval:
                self._bullseye_last_time = now_be
                image_bullseye = self.transform_to_bullseye_prealloc(image=image)
                bullseye_bytes = image_bullseye.tobytes()
                bullseye_shape = image_bullseye.shape
                g = generation
                Clock.schedule_once(
                    lambda dt, b=bullseye_bytes, s=bullseye_shape, gen=g: (
                        self.create_and_set_bullseye_texture(b, s, gen)
                    ),
                    0,
                )
                # Publish for thread.add_frame_listener fan-out
                self._last_rendered_frame = (bullseye_bytes, bullseye_shape, time.monotonic())

        if not self.use_bullseye:
            t_process_start = time.monotonic()
            if self.use_live_image_histogram_equalization:
                image = self._contrast_stretcher.update(image)

            # Convert to bytes on worker thread, blit on main thread
            image_bytes = image.tobytes()
            t_process_end = time.monotonic()
            image_shape = image.shape
            t_blit_scheduled = time.monotonic()
            g = generation
            Clock.schedule_once(
                lambda dt, b=image_bytes, s=image_shape, ts=t_blit_scheduled, gen=g: (
                    self.create_and_set_texture(b, s, ts, gen)
                ),
                0,
            )
            # Publish for thread.add_frame_listener fan-out
            self._last_rendered_frame = (image_bytes, image_shape, t_blit_scheduled)

            # Performance instrumentation gated on settings.debug_mode, cached on
            # the first frame. This mirrors the debug_mode gate lvp_logger uses to
            # set the LVP logger level + lift DEBUG suppression, so the [PERF]
            # logger.debug below actually reaches the log when debug_mode is on.
            if self._debug_perf is None:
                ctx = _app_ctx.ctx
                self._debug_perf = bool(ctx is not None and ctx.settings.get('debug_mode', False))
            if self._debug_perf:
                self._perf_grab_times.append(t_grab_end - t_grab_start)
                self._perf_process_times.append(t_process_end - t_process_start)
                now_perf = time.monotonic()
                if now_perf - self._perf_log_last_time >= self._perf_log_interval:
                    self._perf_log_last_time = now_perf
                    n = len(self._perf_grab_times)
                    if n > 0:
                        avg_grab = sum(self._perf_grab_times) / n * 1000
                        avg_proc = sum(self._perf_process_times) / n * 1000
                        max_grab = max(self._perf_grab_times) * 1000
                        max_proc = max(self._perf_process_times) * 1000
                        avg_queue = (
                            sum(self._perf_blit_schedule_times)
                            / max(1, len(self._perf_blit_schedule_times))
                            * 1000
                        )
                        kivy_fps = Clock.get_fps()
                        kivy_rfps = Clock.get_rfps()
                        display_fps = self._display_fps_value
                        avg_blit_delay = (
                            sum(self._perf_blit_delays) / max(1, len(self._perf_blit_delays)) * 1
                            if self._perf_blit_delays
                            else 0
                        )
                        max_blit_delay = (
                            max(self._perf_blit_delays) if self._perf_blit_delays else 0
                        )
                        capture_fps = self._capture_fps_value
                        logger.debug(
                            f'[PERF] capture={capture_fps:.1f} display={display_fps:.1f} '
                            f'kivy={kivy_fps:.0f}/{kivy_rfps:.0f} FPS | '
                            f'queue={avg_queue:.1f}ms grab={avg_grab:.1f}ms(max {max_grab:.1f}) '
                            f'proc={avg_proc:.1f}ms(max {max_proc:.1f}) '
                            f'blit_delay={avg_blit_delay:.1f}ms(max {max_blit_delay:.0f}) eng={t_eng_stats * 1000:.1f}ms'
                        )
                    self._perf_grab_times.clear()
                    self._perf_process_times.clear()
                    self._perf_blit_schedule_times.clear()
                    self._perf_blit_delays.clear()

        return STATUS_OK

    # _schedule_next retired; ScopeDisplayThread loop owns pacing.

    def create_and_set_bullseye_texture(self, image_bytes, shape, generation=0):
        if generation != self._current_generation():
            return  # Stale callback from previous start/stop cycle
        size = (shape[1], shape[0])
        # Mirror the mono path's caching -- only allocate a new GDI texture
        # when the frame size changes; otherwise blit into the existing
        # one. Pre-cache fix: at the 15 fps bullseye cap this leaked
        # ~54k texture objects per hour.
        if (
            not hasattr(self, '_bullseye_texture')
            or self._bullseye_texture is None
            or self._bullseye_texture.size != size
        ):
            self._bullseye_texture = Texture.create(size=size, colorfmt='rgb')
        self._bullseye_texture.blit_buffer(image_bytes, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = self._bullseye_texture
        self.canvas.ask_update()
        self._count_display_fps()
        # _schedule_next retired; ScopeDisplayThread loop owns pacing.

    def _current_generation(self):
        """Return the active display generation from the thread, or 0
        if the thread hasn't spawned yet. Used by texture-blit callbacks
        to discard callbacks from a previous start/stop cycle."""
        ctx = _app_ctx.ctx
        thread = getattr(ctx, 'scope_display_thread', None) if ctx else None
        return thread.generation if thread is not None else 0

    def create_and_set_texture(self, image_bytes, shape, scheduled_time=0, generation=0):
        if generation != self._current_generation():
            return  # Stale callback from previous start/stop cycle
        if scheduled_time and self._debug_perf:
            blit_delay = (time.monotonic() - scheduled_time) * 1000
            self._perf_blit_delays.append(blit_delay)
            if blit_delay > 100:
                logger.debug(
                    f'[PERF] Blit callback delayed {blit_delay:.0f}ms (main thread congested)'
                )
        size = (shape[1], shape[0])
        if (
            not hasattr(self, '_mono_texture')
            or self._mono_texture is None
            or self._mono_texture.size != size
        ):
            self._mono_texture = Texture.create(size=size, colorfmt='luminance')
        self._mono_texture.blit_buffer(image_bytes, colorfmt='luminance', bufferfmt='ubyte')
        self.texture = self._mono_texture
        self.canvas.ask_update()
        self._count_display_fps()
        # _schedule_next retired; ScopeDisplayThread loop owns pacing.

    def hold_protocol_saved_image(self, image):
        """DISPLAY-1: show the most-recent protocol-saved image and hold it.

        Called from the protocol-image-writer thread immediately after a
        step finishes capturing. Pushes the captured frame to the
        display texture (so the user sees the actual saved frame, not
        a stale live grab) and bumps the hold deadline to ``now +
        PROTOCOL_HOLD_MS`` so the live preview's pull loop pauses long
        enough for the save to be visible. The next protocol save bumps
        the deadline forward again -- there's no added delay anywhere,
        only a minimum-visible-time floor on the most-recent saved
        frame.

        Args:
            image: numpy.ndarray, the captured frame in either 8-bit or
                12-bit grayscale (matching what was saved). The display
                is luminance-only; if the array is wider than 8 bits,
                we convert with the same LUT the live path uses.
        """
        if image is None or getattr(image, 'size', 0) == 0:
            return
        try:
            from kivy.clock import Clock as _Clock
            import modules.image_utils as _image_utils

            arr = image
            if arr.dtype != np.uint8:
                arr = _image_utils.convert_12bit_to_8bit(arr)
            shape = arr.shape
            data = arr.tobytes()
            gen = self._current_generation()
            # Bump hold deadline on the thread (was self._protocol_hold_until).
            ctx = _app_ctx.ctx
            thread = getattr(ctx, 'scope_display_thread', None) if ctx else None
            if thread is not None:
                thread.bump_protocol_hold(self._PROTOCOL_HOLD_MS / 1000.0)
            _Clock.schedule_once(
                lambda dt, b=data, s=shape, g=gen: self.create_and_set_texture(b, s, generation=g),
                0,
            )
        except Exception as e:
            logger.warning(f'[LVP Main  ] hold_protocol_saved_image failed: {e}')

    def _count_display_fps(self):
        """Track actual rendered frame rate (called on main thread after blit).

        Capped at capture FPS -- display cannot render more frames than
        the camera produces, any excess is measurement window jitter.
        """
        self._display_fps_count += 1
        now = time.monotonic()
        elapsed = now - self._display_fps_last_time
        if elapsed >= 1.0:
            raw_display_fps = self._display_fps_count / elapsed
            self._display_fps_value = (
                min(raw_display_fps, self._capture_fps_value)
                if self._capture_fps_value > 0
                else raw_display_fps
            )
            self._display_fps_count = 0
            self._display_fps_last_time = now

    def get_true_gain_exp(self, layer):
        ctx = _app_ctx.ctx
        actual_gain = ctx.scope.imaging.camera_gain
        actual_exp = ctx.scope.imaging.camera_exposure_ms
        Clock.schedule_once(lambda dt: self.update_auto_gain_ui(layer, actual_gain, actual_exp), 0)

    def update_auto_gain_ui(self, layer, actual_gain, actual_exp):
        ctx = _app_ctx.ctx
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        # Only update if values changed to prevent unnecessary ScrollView layout recalculation
        if abs(layer_obj.ids['gain_slider'].value - actual_gain) > 0.01:
            layer_obj.ids['gain_slider'].value = actual_gain
        if abs(layer_obj.ids['exp_slider'].value - actual_exp) > 0.01:
            layer_obj.ids['exp_slider'].value = actual_exp
