# Copyright Etaluma, Inc.
"""
ScopeDisplay — pull-based image display loop.

Image Pipeline (sensor → screen):
  1. Camera SDK callback → ImageHandler._store_frame()     [1 copy: SDK buffer → numpy]
  2. grab_latest() → returns stored reference               [0 copies]
  3. get_image_from_buffer():
     - scale bar overlay (in-place on the reference)        [0 copies]
     - 12→8 bit LUT conversion (if force_to_8bit)           [1 copy: LUT indexing]
  4. Worker thread: contrast stretch / bullseye LUT          [1 copy: LUT indexing]
  5. image.tobytes() → blit_buffer() to GPU texture          [1 copy: tobytes]

Copy budget:
  8-bit path:  SDK(1) + tobytes(1)                    = 2 copies
  12-bit path: SDK(1) + 12→8 LUT(1) + tobytes(1)     = 3 copies

Threading model:
  - Main thread (Kivy): _pull_next_frame(), create_and_set_texture(), _schedule_next()
  - Worker thread (scope_display_thread_executor): update_scopedisplay_thread()
  - Generation counter prevents stale callbacks after stop()/start() cycles
  - _schedule_next() enforces FPS cap via Clock.schedule_once delay
"""
import logging
import threading
import time

import numpy as np
import skimage.draw

from kivy.clock import Clock
from kivy.graphics import InstructionGroup, Color, Line
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import BooleanProperty
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.input import MotionEvent

from modules.contrast_stretcher import ContrastStretcher
import modules.common_utils as common_utils
import modules.autofocus_functions as autofocus_functions
import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.scope_display')


class ScopeDisplay(Image):
    play = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(ScopeDisplay,self).__init__(**kwargs)
        logger.debug('[LVP Main  ] ScopeDisplay.__init__()')
        self.play = True
        self.paused = threading.Event()
        self.paused.clear()

        self.use_bullseye = False
        self.use_crosshairs = False
        self.use_live_image_histogram_equalization = False
        self.camera_disconnected_display_set = False

        self._bullseye_rgb_buf = None
        self._bullseye_buf_shape = None

        # FPS tracking — worker thread (frames grabbed)
        self._fps_frame_count = 0
        self._fps_last_time = time.monotonic()
        self._fps_value = 0.0

        # Display FPS tracking — main thread (frames actually rendered on screen)
        self._display_fps_count = 0
        self._display_fps_last_time = time.monotonic()
        self._display_fps_value = 0.0

        # Engineering stats timing (2x per second)
        self._eng_stats_last_time = 0.0

        # Performance instrumentation (enabled via settings.debug_mode)
        self._perf_log_interval = 5.0   # seconds between perf reports
        self._perf_log_last_time = 0.0
        self._perf_grab_times = []
        self._perf_process_times = []
        self._perf_blit_schedule_times = []
        self._perf_blit_delays = []


        # Bullseye frame rate cap (15 FPS — CPU-intensive LUT rendering)
        self._bullseye_min_interval = 1.0 / 15
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

        # Pull-based display loop state
        self._display_running = False
        self._display_generation = 0  # Incremented on each start() to invalidate stale callbacks
        self._cycle_start_time = 0.0  # When _pull_next_frame() was called (full cycle timing)
        self._last_frame_ts = None  # Camera timestamp of last displayed frame
        self._min_frame_interval = 1.0 / 30  # derived from fps setting

        # Crosshair canvas overlay (drawn on top of texture, not into pixels)
        self._crosshair_group = InstructionGroup()
        self.canvas.after.add(self._crosshair_group)
        self._crosshair_visible = False
        self.bind(size=self._on_size_changed, pos=self._on_size_changed, texture=self._on_size_changed)

        # Create a black texture to avoid white flash on startup
        self._create_default_black_texture()

        self.start()

    def _create_default_black_texture(self):
        """Create a default black texture to display before camera feed starts."""
        # Create a small black image (will be stretched to fit)
        black_image = np.zeros((100, 100), dtype=np.uint8)
        texture = Texture.create(size=(black_image.shape[1], black_image.shape[0]), colorfmt='luminance')
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
        self._crosshair_group.add(Line(
            points=[cx, cy - img_h / 2, cx, cy + img_h / 2],
            width=line_width,
        ))

        # Horizontal center line (full width of displayed image)
        self._crosshair_group.add(Line(
            points=[cx - img_w / 2, cy, cx + img_w / 2, cy],
            width=line_width,
        ))

        # 4 radiating circles, evenly spaced across half the minimum dimension
        num_circles = 4
        circle_spacing = min_dim / 2 / num_circles
        for i in range(num_circles):
            radius = (i + 1) * circle_spacing
            self._crosshair_group.add(Line(
                circle=(cx, cy, radius),
                width=line_width,
            ))

    def show_crosshairs(self, show):
        """Show or hide the crosshair overlay."""
        self._crosshair_visible = show
        if show:
            self._build_crosshair_overlay()
        else:
            self._crosshair_group.clear()

    def start(self, fps = None):
        logger.info('[LVP Main  ] ScopeDisplay.start()')
        ctx = _app_ctx.ctx
        if fps is not None:
            self.fps = fps
        elif ctx is not None and 'live_view_fps' in ctx.settings:
            self.fps = ctx.settings['live_view_fps']
        else:
            self.fps = 30

        # fps=0 means uncapped — run as fast as the pipeline allows
        if self.fps == 0:
            self._min_frame_interval = 0
        else:
            self._min_frame_interval = 1.0 / max(1, self.fps)
        self._display_generation += 1
        self.paused.clear()

        if not self._display_running:
            self._display_running = True
            fps_label = "uncapped" if self.fps == 0 else f"{self.fps} FPS cap"
            logger.info(f'[LVP Main  ] ScopeDisplay: pull-based loop started ({fps_label})')
        Clock.schedule_once(self._pull_next_frame, 0)

    def stop(self):
        self.paused.set()
        self._display_running = False
        Clock.unschedule(self._pull_next_frame)
        logger.info('[LVP Main  ] ScopeDisplay.stop()')


    def touch(self, target: Widget, event: MotionEvent):
        if event.is_touch and (event.device == 'mouse') and (event.button == 'right'):
            norm_texture_width, norm_texture_height = self.norm_image_size
            norm_texture_x_min = self.center_x - norm_texture_width/2
            norm_texture_x_max = self.center_x + norm_texture_width/2
            norm_texture_y_min = self.center_y - norm_texture_height/2
            norm_texture_y_max = self.center_y + norm_texture_height/2

            click_pos_x = event.pos[0]
            click_pos_y = event.pos[1]

            # Check if click occurred within texture
            if (click_pos_x >= norm_texture_x_min) and (click_pos_x <= norm_texture_x_max) and \
               (click_pos_y >= norm_texture_y_min) and (click_pos_y <= norm_texture_y_max):
                norm_texture_click_pos_x = click_pos_x - norm_texture_x_min
                norm_texture_click_pos_y = click_pos_y - norm_texture_y_min
                texture_width, texture_height = self.texture_size

                # Scale to image pixels
                texture_click_pos_x = norm_texture_click_pos_x * texture_width / norm_texture_width
                texture_click_pos_y = norm_texture_click_pos_y * texture_height / norm_texture_height

                # Distance from center
                x_dist_pixel = texture_click_pos_x - texture_width/2 # Positive means to the right of center
                y_dist_pixel = texture_click_pos_y - texture_height/2 # Positive means above center

                from modules.config_ui_getters import get_current_objective_info, get_binning_from_ui
                from modules.ui_helpers import move_relative_position
                _, objective = get_current_objective_info()
                pixel_size_um = common_utils.get_pixel_size(
                    focal_length=objective['focal_length'],
                    binning_size=get_binning_from_ui(),
                )

                x_dist_um = x_dist_pixel * pixel_size_um
                y_dist_um = y_dist_pixel * pixel_size_um

                ctx = _app_ctx.ctx
                from modules.sequential_io_executor import IOTask
                ctx.io_executor.put(IOTask(move_relative_position, kwargs={'axis':'X', 'um':x_dist_um}))
                ctx.io_executor.put(IOTask(move_relative_position, kwargs={'axis':'Y', 'um':y_dist_um}))


    @staticmethod
    def add_crosshairs(image):
        height, width = image.shape[0], image.shape[1]

        if image.ndim == 3:
            is_color = True
        else:
            is_color = False

        center_x = round(width/2)
        center_y = round(height/2)

        # Crosshairs - 2 pixels wide
        if is_color:
            image[:,center_x-1:center_x+1,:] = 255
            image[center_y-1:center_y+1,:,:] = 255
        else:
            image[:,center_x-1:center_x+1] = 255
            image[center_y-1:center_y+1,:] = 255

        # Radiating circles
        num_circles = 4
        minimum_dimension = min(height, width)
        circle_spacing = round(minimum_dimension/ 2 / num_circles)
        for i in range(num_circles):
            radius = (i+1) * circle_spacing
            rr, cc = skimage.draw.circle_perimeter(center_y, center_x, radius=radius, shape=image.shape)
            image[rr, cc] = 255

            # To make circles 2 pixel wide...
            rr, cc = skimage.draw.circle_perimeter(center_y, center_x, radius=radius+1, shape=image.shape)
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
            (  5,  15,   0, 255,   0),
            ( 25,  35,   0, 255,   0),
            ( 45,  55,   0, 255,   0),
            ( 65,  75,   0, 255,   0),
            ( 85,  95,   0, 255,   0),
            (105, 115,   0, 255,   0),
            (125, 135,   0,   0, 255),
            (145, 155,   0, 255,   0),
            (165, 175,   0, 255,   0),
            (185, 195,   0, 255,   0),
            (205, 215,   0, 255,   0),
            (225, 235,   0, 255,   0),
            (245, 255, 255,   0,   0),
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


    def _pull_next_frame(self, dt=0):
        """Pull-based display loop entry point. Called on main thread.

        Schedules the next frame grab on the display worker. The worker
        calls _schedule_next() when done, which re-invokes this method
        after enforcing the minimum frame interval. This naturally adapts
        to the system's actual throughput — no timer overrun possible.
        """
        if not self._display_running or self.paused.is_set():
            return

        self._cycle_start_time = time.monotonic()

        ctx = _app_ctx.ctx
        if ctx is None:
            # Not ready yet — retry shortly
            Clock.schedule_once(self._pull_next_frame, 0.1)
            return

        # Capture widget state on the main thread (Kivy widgets are not thread-safe)
        active_layer = None
        active_layer_config = None
        open_layer = None
        try:
            from modules.config_ui_getters import get_active_layer_config
            active_layer, active_layer_config = get_active_layer_config()
        except Exception:
            pass

        if ctx.engineering_mode:
            for layer in common_utils.get_layers():
                accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
                if not accordion_item_obj.collapse:
                    open_layer = layer
                    break

        from modules.sequential_io_executor import IOTask
        dispatch_time = time.monotonic()
        gen = self._display_generation
        ctx.scope_display_thread_executor.put(IOTask(
            self.update_scopedisplay_thread,
            args=(active_layer, active_layer_config, open_layer, dispatch_time, gen),
        ))

    def update_scopedisplay(self, dt=0):
        """Trigger a one-shot display update (used as callback by protocol executors).

        In the pull-based loop, this simply kicks a frame grab if the loop
        isn't already running. Safe to call from Clock.schedule_once or as
        a direct callback.
        """
        self._pull_next_frame(dt)

    def set_engineering_ui(self, mean, stddev, af_score, open_layer):
        ctx = _app_ctx.ctx
        open_layer_obj = ctx.image_settings.layer_lookup(layer=open_layer)
        new_mean_text = f"Mean: {mean}"
        if open_layer_obj.ids['image_stats_mean_id'].text != new_mean_text:
            open_layer_obj.ids['image_stats_mean_id'].text = new_mean_text
        new_stddev_text = f"StdDev: {stddev}"
        if open_layer_obj.ids['image_stats_stddev_id'].text != new_stddev_text:
            open_layer_obj.ids['image_stats_stddev_id'].text = new_stddev_text
        new_af_text = f"AF Score: {af_score}"
        if open_layer_obj.ids['image_af_score_id'].text != new_af_text:
            open_layer_obj.ids['image_af_score_id'].text = new_af_text

    def set_camera_disconnected_display(self):
        self.source = "./data/icons/camera_to_USB.png"
        self.camera_disconnected_display_set = True
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

    def update_scopedisplay_thread(self, active_layer, active_layer_config, open_layer, dispatch_time=0, generation=0):
        ctx = _app_ctx.ctx

        # Drop stale callbacks from a previous start()/stop() cycle
        if generation != self._display_generation:
            return

        t_worker_start = time.monotonic()
        t_queue_wait = t_worker_start - dispatch_time if dispatch_time else 0

        # Snapshot counter value before scheduling increment on main thread
        display_counter = self._display_update_counter + 1
        Clock.schedule_once(self._increment_display_counter, 0)

        if not ctx.scope.camera_is_connected():
            if not self.camera_disconnected_display_set:
                Clock.schedule_once(lambda dt: self.set_camera_disconnected_display(), 0)
            # No frame — retry after a short delay
            Clock.schedule_once(self._pull_next_frame, 0.2)
            return

        if self.camera_disconnected_display_set:
            Clock.schedule_once(lambda dt: self.source_clear(), 0)

        # Update scale bar color based on active channel (black for transmitted, white for fluorescence)
        if active_layer is not None:
            ctx.scope._scale_bar['color'] = active_layer

        # Likely not an IO call as image will be stored in buffer
        t_grab_start = time.monotonic()
        image, frame_ts = ctx.scope.get_image_from_buffer(force_to_8bit=True)
        if (image is False) or (image is None) or (image.size == 0):
            # No new frame available — retry after minimum interval
            Clock.schedule_once(self._pull_next_frame, self._min_frame_interval)
            return

        # Skip duplicate frames (same camera timestamp = same data)
        if frame_ts is not None and frame_ts == self._last_frame_ts:
            Clock.schedule_once(self._pull_next_frame, self._min_frame_interval)
            return
        self._last_frame_ts = frame_ts
        t_grab_end = time.monotonic()

        # Record queue wait for perf logging (debug only)
        if logger.isEnabledFor(logging.DEBUG):
            self._perf_blit_schedule_times.append(t_queue_wait)

        # FPS tracking
        self._fps_frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_value = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_last_time = now

        if display_counter % 10 == 0:
            Clock.schedule_once(self._reset_display_counter, 0)

            if active_layer_config is not None and active_layer_config['auto_gain']:
                from modules.sequential_io_executor import IOTask
                ctx.camera_executor.put(IOTask(action=self.get_true_gain_exp, args=(active_layer,)))


        t_eng_stats = 0
        if ctx.engineering_mode:
            # Engineering stats: 2x per second (time-based, not frame-based)
            now_eng = time.monotonic()
            if now_eng - self._eng_stats_last_time >= 0.5 and not self.use_bullseye:
                self._eng_stats_last_time = now_eng
                t_eng_start = time.monotonic()
                mean = round(np.mean(a=image), 2)
                stddev = round(np.std(a=image), 2)
                af_score = autofocus_functions.focus_function(image=image, skip_score_logging=True)
                t_eng_stats = time.monotonic() - t_eng_start

                if open_layer is not None:
                    Clock.schedule_once(lambda dt: self.set_engineering_ui(mean, stddev, af_score, open_layer), 0)

        if self.use_bullseye:
            now_be = time.monotonic()
            if now_be - self._bullseye_last_time >= self._bullseye_min_interval:
                self._bullseye_last_time = now_be
                image_bullseye = self.transform_to_bullseye_prealloc(image=image)
                bullseye_bytes = image_bullseye.tobytes()
                bullseye_shape = image_bullseye.shape
                g = generation
                Clock.schedule_once(lambda dt, b=bullseye_bytes, s=bullseye_shape, gen=g: self.create_and_set_bullseye_texture(b, s, gen), 0)
            else:
                # Bullseye frame-rate cap skipped this frame — keep the loop alive
                self._schedule_next()

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
            Clock.schedule_once(lambda dt, b=image_bytes, s=image_shape, ts=t_blit_scheduled, gen=g: self.create_and_set_texture(b, s, ts, gen), 0)

            # Performance instrumentation — only when DEBUG logging enabled
            if logger.isEnabledFor(logging.DEBUG):
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
                        avg_queue = sum(self._perf_blit_schedule_times) / max(1, len(self._perf_blit_schedule_times)) * 1000
                        kivy_fps = Clock.get_fps()
                        kivy_rfps = Clock.get_rfps()
                        display_fps = self._display_fps_value
                        avg_blit_delay = sum(self._perf_blit_delays) / max(1, len(self._perf_blit_delays)) * 1 if self._perf_blit_delays else 0
                        max_blit_delay = max(self._perf_blit_delays) if self._perf_blit_delays else 0
                        logger.debug(
                            f'[PERF] worker={n/self._perf_log_interval:.1f} display={display_fps:.1f} '
                            f'kivy={kivy_fps:.0f}/{kivy_rfps:.0f} FPS | '
                            f'queue={avg_queue:.1f}ms grab={avg_grab:.1f}ms(max {max_grab:.1f}) '
                            f'proc={avg_proc:.1f}ms(max {max_proc:.1f}) '
                            f'blit_delay={avg_blit_delay:.1f}ms(max {max_blit_delay:.0f}) eng={t_eng_stats*1000:.1f}ms'
                        )
                    self._perf_grab_times.clear()
                    self._perf_process_times.clear()
                    self._perf_blit_schedule_times.clear()
                    self._perf_blit_delays.clear()
            

    def _schedule_next(self):
        """Schedule the next frame grab, enforcing minimum frame interval.

        Called on the main thread after blit completes. Measures elapsed time
        from the START of the current cycle (_pull_next_frame) to account for
        the full pipeline: dispatch → worker grab → worker process → blit.
        """
        if not self._display_running or self.paused.is_set():
            return
        now = time.monotonic()
        elapsed = now - self._cycle_start_time
        wait = max(0, self._min_frame_interval - elapsed)
        Clock.schedule_once(self._pull_next_frame, wait)

    def create_and_set_bullseye_texture(self, image_bytes, shape, generation=0):
        if generation != self._display_generation:
            return  # Stale callback from previous start/stop cycle
        size = (shape[1], shape[0])
        self._bullseye_texture = Texture.create(size=size, colorfmt='rgb')
        self._bullseye_texture.blit_buffer(image_bytes, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = self._bullseye_texture
        self.canvas.ask_update()
        self._count_display_fps()
        self._schedule_next()

    def create_and_set_texture(self, image_bytes, shape, scheduled_time=0, generation=0):
        if generation != self._display_generation:
            return  # Stale callback from previous start/stop cycle
        if scheduled_time and logger.isEnabledFor(logging.DEBUG):
            blit_delay = (time.monotonic() - scheduled_time) * 1000
            self._perf_blit_delays.append(blit_delay)
            if blit_delay > 100:
                logger.debug(f'[PERF] Blit callback delayed {blit_delay:.0f}ms (main thread congested)')
        size = (shape[1], shape[0])
        if not hasattr(self, '_mono_texture') or self._mono_texture is None or self._mono_texture.size != size:
            self._mono_texture = Texture.create(size=size, colorfmt='luminance')
        self._mono_texture.blit_buffer(image_bytes, colorfmt='luminance', bufferfmt='ubyte')
        self.texture = self._mono_texture
        self.canvas.ask_update()
        self._count_display_fps()
        self._schedule_next()



    def _count_display_fps(self):
        """Track actual rendered frame rate (called on main thread after blit)."""
        self._display_fps_count += 1
        now = time.monotonic()
        elapsed = now - self._display_fps_last_time
        if elapsed >= 1.0:
            self._display_fps_value = self._display_fps_count / elapsed
            self._display_fps_count = 0
            self._display_fps_last_time = now

    def get_true_gain_exp(self, layer):
        ctx = _app_ctx.ctx
        actual_gain = ctx.scope.camera_gain
        actual_exp = ctx.scope.camera_exposure_ms
        Clock.schedule_once(lambda dt: self.update_auto_gain_ui(layer, actual_gain, actual_exp), 0)

    def update_auto_gain_ui(self, layer, actual_gain, actual_exp):
        ctx = _app_ctx.ctx
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        # Only update if values changed to prevent unnecessary ScrollView layout recalculation
        if abs(layer_obj.ids['gain_slider'].value - actual_gain) > 0.01:
            layer_obj.ids['gain_slider'].value = actual_gain
        if abs(layer_obj.ids['exp_slider'].value - actual_exp) > 0.01:
            layer_obj.ids['exp_slider'].value = actual_exp
