# Copyright Etaluma, Inc.
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
    record = BooleanProperty(None)
    record = False
    play = BooleanProperty(None)
    play = True

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

        # FPS tracking
        self._fps_frame_count = 0
        self._fps_last_time = time.monotonic()
        self._fps_value = 0.0

        self._contrast_stretcher = ContrastStretcher(
            window_len=3,
            bottom_pct=0.3,
            top_pct=0.3,
        )

        self.use_full_pixel_depth = False

        # Counters (were module-level globals in lumaviewpro.py)
        self._debug_counter = 0
        self._display_update_counter = 0

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

        logger.info('[LVP Main  ] Clock.schedule_interval(self.update, 1.0 / self.fps)')
        self.paused.clear()

        Clock.unschedule(self.update_scopedisplay)
        Clock.schedule_interval(self.update_scopedisplay, 1.0 / self.fps)

    def stop(self):
        self.paused.set()
        logger.info('[LVP Main  ] ScopeDisplay.stop()')
        logger.info('[LVP Main  ] Clock.unschedule(self.update)')
        Clock.unschedule(self.update_scopedisplay)


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

                from modules.config_getters import get_current_objective_info, get_binning_from_ui
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


    def update_scopedisplay(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            return

        # Backpressure: avoid flooding the scope display executor if it is still draining
        try:
            if hasattr(ctx.scope_display_thread_executor, 'queue_size') and ctx.scope_display_thread_executor.queue_size() > 3:
                return
        except Exception:
            pass

        # Capture widget state on the main thread (Kivy widgets are not thread-safe)
        active_layer = None
        active_layer_config = None
        open_layer = None
        try:
            from modules.config_getters import get_active_layer_config
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
        ctx.scope_display_thread_executor.put(IOTask(
            self.update_scopedisplay_thread,
            args=(active_layer, active_layer_config, open_layer),
        ))

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

    def update_scopedisplay_thread(self, active_layer, active_layer_config, open_layer):
        ctx = _app_ctx.ctx

        self._display_update_counter += 1

        if not ctx.scope.camera_is_connected():
            if self.camera_disconnected_display_set:
                return

            Clock.schedule_once(lambda dt: self.set_camera_disconnected_display(), 0)
            return

        if self.camera_disconnected_display_set:
            Clock.schedule_once(lambda dt: self.source_clear(), 0)

        # Update scale bar color based on active channel (black for transmitted, white for fluorescence)
        if active_layer is not None:
            ctx.scope._scale_bar['color'] = active_layer

        # Likely not an IO call as image will be stored in buffer
        image = ctx.scope.get_image_from_buffer(force_to_8bit=True)
        #image = ctx.scope.image_buffer
        if (image is False) or (image.size == 0) :
            return

        # FPS tracking
        self._fps_frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_value = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_last_time = now

        if self._display_update_counter % 10 == 0:
            self._display_update_counter = 0

            if active_layer_config is not None and active_layer_config['auto_gain']:
                from modules.sequential_io_executor import IOTask
                ctx.camera_executor.put(IOTask(action=self.get_true_gain_exp, args=(active_layer)))


        if ctx.engineering_mode:
            self._debug_counter += 1
            if self._debug_counter == 30:
                self._debug_counter = 0

            if self._debug_counter % 10 == 0:
                mean = round(np.mean(a=image), 2)
                stddev = round(np.std(a=image), 2)
                af_score = autofocus_functions.focus_function(image=image, skip_score_logging=True)

                if open_layer is not None:
                    Clock.schedule_once(lambda dt: self.set_engineering_ui(mean, stddev, af_score, open_layer), 0)

        if ctx.engineering_mode and self.use_bullseye:
            image_bullseye = self.transform_to_bullseye_prealloc(image=image)
            bullseye_bytes = image_bullseye.tobytes()
            bullseye_shape = image_bullseye.shape
            Clock.schedule_once(lambda dt, b=bullseye_bytes, s=bullseye_shape: self.create_and_set_bullseye_texture(b, s), 0)

        if not self.use_bullseye:
            if self.use_live_image_histogram_equalization:
                image = self._contrast_stretcher.update(image)

            # Convert to bytes on worker thread, blit on main thread
            image_bytes = image.tobytes()
            image_shape = image.shape
            Clock.schedule_once(lambda dt, b=image_bytes, s=image_shape: self.create_and_set_texture(b, s), 0)

        if self.record:
            ctx.lumaview.live_capture()

    def create_and_set_bullseye_texture(self, image_bytes, shape):
        size = (shape[1], shape[0])
        if not hasattr(self, '_bullseye_texture') or self._bullseye_texture is None or self._bullseye_texture.size != size:
            self._bullseye_texture = Texture.create(size=size, colorfmt='rgb')
        self._bullseye_texture.blit_buffer(image_bytes, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = self._bullseye_texture

    def create_and_set_texture(self, image_bytes, shape):
        size = (shape[1], shape[0])
        if not hasattr(self, '_mono_texture') or self._mono_texture is None or self._mono_texture.size != size:
            self._mono_texture = Texture.create(size=size, colorfmt='luminance')
        self._mono_texture.blit_buffer(image_bytes, colorfmt='luminance', bufferfmt='ubyte')
        self.texture = self._mono_texture



    def get_true_gain_exp(self, layer):
        ctx = _app_ctx.ctx
        actual_gain = ctx.scope.camera.get_gain()
        actual_exp = ctx.scope.camera.get_exposure_t()
        Clock.schedule_once(lambda dt: self.update_auto_gain_ui(layer, actual_gain, actual_exp), 0)

    def update_auto_gain_ui(self, layer, actual_gain, actual_exp):
        ctx = _app_ctx.ctx
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        # Only update if values changed to prevent unnecessary ScrollView layout recalculation
        if abs(layer_obj.ids['gain_slider'].value - actual_gain) > 0.01:
            layer_obj.ids['gain_slider'].value = actual_gain
        if abs(layer_obj.ids['exp_slider'].value - actual_exp) > 0.01:
            layer_obj.ids['exp_slider'].value = actual_exp
