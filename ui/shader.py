# Copyright Etaluma, Inc.
import logging
import time

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.scatter import Scatter

import modules.app_context as _app_ctx
import modules.config_ui_getters as config_ui_getters

logger = logging.getLogger('LVP.ui.shader')


# -----------------------------------------------------------------------------
# Shader code
# Based on code from the kivy example Live Shader Editor found at:
# kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html
# -----------------------------------------------------------------------------
fs_header = """
#ifdef GL_ES
precision highp float;
#endif

/* Outputs from the vertex shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* uniform texture samplers */
uniform sampler2D texture0;

/* fragment attributes
attribute float red_gain;
attribute float green_gain;
attribute float blue_gain; */

/* custom one */
uniform vec2 resolution;
uniform float time;
uniform vec4 black_point;
uniform vec4 white_point;
"""

vs_header = """
#ifdef GL_ES
precision highp float;
#endif

/* Outputs to the fragment shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* vertex attributes */
attribute vec2     vPosition;
attribute vec2     vTexCoords0;

/* uniform variables */
uniform mat4       modelview_mat;
uniform mat4       projection_mat;
uniform vec4       color;
"""


# ============================================================================
# ShaderViewer -- GPU Shader-Based Image Display with Pan/Zoom
# ============================================================================


class ShaderViewer(Scatter):
    black = ObjectProperty(0.0)
    white = ObjectProperty(1.0)

    fs = StringProperty("""
void main (void) {
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
    //gl_FragColor = pow(glFragColor.rgb, 1/gamma)
}
""")
    vs = StringProperty("""
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
""")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] ShaderViewer.__init__()')
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        self.white = 1.0
        self.black = 0.0

        # Status bar update interval. Drives FPS readout AND cursor
        # XY/Plate readouts in the window title (#638 follow-up: 1 Hz
        # was too sluggish for stage-position feedback during motion).
        # 10 Hz keeps cursor XY responsive without saturating
        # Window.set_title() on SDL2/Windows.
        self._status_bar_trigger = Clock.create_trigger(self._update_status_bar, 0.1, interval=True)
        self._status_bar_trigger()
        self._mouse_pixel_x = -1
        self._mouse_pixel_y = -1
        self._mouse_over_image = False
        Window.bind(mouse_pos=self._on_mouse_pos)

        # Scroll-to-focus: accumulate scroll ticks and debounce into single move
        self._scroll_z_pending = 0.0  # Accumulated Z delta (um)
        self._scroll_z_trigger = Clock.create_trigger(self._flush_scroll_z, 0.05)
        self._scroll_last_time = 0.0  # monotonic time of last scroll event
        self._scroll_inertia_window = 0.15  # seconds -- scrolls faster than this get multiplied

    def on_touch_down(self, touch, *args):
        logger.debug('[LVP Main  ] ShaderViewer.on_touch_down()')
        ctx = _app_ctx.ctx

        ZOOM_BLOCKERS = [ctx.image_settings, ctx.motion_settings]
        x, y = touch.pos

        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            for w in ZOOM_BLOCKERS:
                lx, ly = w.to_widget(x, y)
                if w.collide_point(lx, ly):
                    return

            # Query the window's live modifier state (the single owner) rather
            # than a private mirror that can desync when a ctrl/shift transition
            # happens while this widget is not receiving key events (focus lost
            # or regained with the key held, a popup consuming the event). The
            # window rebuilds this from the OS modifier state on every key
            # event, so there is nothing to go stale.
            if 'ctrl' in Window.modifiers:
                # Focus control -- accumulate scroll ticks, debounce into single move
                if ctx.session.controls_locked:
                    return

                try:
                    _, objective = ctx.session.get_current_objective_info()
                except Exception:
                    logger.debug('[LVP Main  ] Scroll-to-focus: objective info unavailable')
                    return

                if 'shift' in Window.modifiers:
                    step_um = objective['z_coarse']
                else:
                    step_um = objective['z_fine']

                # Inertial scaling: faster scrolling = larger steps
                now = time.monotonic()
                dt = now - self._scroll_last_time
                self._scroll_last_time = now

                if dt < self._scroll_inertia_window and dt > 0:
                    # Scale up when scrolling fast (up to 2x -- was 5x; at low mag
                    # z_fine is already 25-50 um and 5x drove 125-250 um per tick
                    # past the user's intent, esp at 4x/10x. 2x keeps the
                    # "fast scroll = bigger step" feel without overshoot.
                    speed_factor = min(2.0, self._scroll_inertia_window / dt)
                else:
                    speed_factor = 1.0

                # Replace, don't accumulate. Only the LAST tick's intent commits
                # when the user stops -- fast scrolling still produces a bigger
                # move per tick (via speed_factor) but no leftover motion after
                # the user stops, and sign flips become immediate.
                delta = step_um * speed_factor
                if touch.button == 'scrolldown':
                    self._scroll_z_pending = delta
                elif touch.button == 'scrollup':
                    self._scroll_z_pending = -delta

                # Reset the debounce trigger -- fires 50ms after last scroll event
                self._scroll_z_trigger()

            else:
                # Digital zoom control
                if touch.button == 'scrolldown' and self.scale < 100:
                    self.scale = self.scale * 1.1
                elif touch.button == 'scrollup' and self.scale > 1:
                    self.scale = max(1, self.scale * 0.8)
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            # Let side panels handle touches that land on them
            for w in ZOOM_BLOCKERS:
                lx, ly = w.to_widget(x, y)
                if w.collide_point(lx, ly):
                    return w.on_touch_down(touch)
            super().on_touch_down(touch)

    def _flush_scroll_z(self, dt):
        """Debounced scroll-to-focus: send one accumulated Z move."""
        from ui.ui_helpers import move_relative

        delta = self._scroll_z_pending
        self._scroll_z_pending = 0.0

        if delta == 0.0:
            return

        move_relative('Z', delta, overshoot_enabled=False)

    def _on_mouse_pos(self, window, pos):
        """Convert window mouse position to image pixel coordinates."""
        scope_display = self.ids.get('scope_display_id')
        if scope_display is None or scope_display.texture is None:
            self._mouse_over_image = False
            return

        # Convert window coords to ShaderViewer (Scatter) local coords
        local_x, local_y = self.to_local(*pos)

        # Get the ScopeDisplay's rendered image bounds within the widget
        norm_w, norm_h = scope_display.norm_image_size
        img_x_min = scope_display.center_x - norm_w / 2
        img_y_min = scope_display.center_y - norm_h / 2
        img_x_max = scope_display.center_x + norm_w / 2
        img_y_max = scope_display.center_y + norm_h / 2

        if img_x_min <= local_x <= img_x_max and img_y_min <= local_y <= img_y_max:
            # Full-resolution sensor frame, not the downscaled preview texture,
            # so the reported pixel coordinate is in sensor pixels.
            frame_w, frame_h = scope_display.full_resolution_frame_size()
            self._mouse_pixel_x = int((local_x - img_x_min) * frame_w / norm_w)
            # Kivy Y is bottom-up, image Y is top-down
            self._mouse_pixel_y = frame_h - 1 - int((local_y - img_y_min) * frame_h / norm_h)
            self._mouse_pixel_x = max(0, min(self._mouse_pixel_x, frame_w - 1))
            self._mouse_pixel_y = max(0, min(self._mouse_pixel_y, frame_h - 1))
            self._mouse_over_image = True
        else:
            self._mouse_over_image = False

    def _update_status_bar(self, dt):
        """Periodic status bar update (~5 Hz). SOLE owner of Window.set_title().

        Composes: 'LumaViewPro {ver} -- Capture: X | Display: Y FPS [ | Camera: Z MB/s ]
        [ | Pixel: (px, py) | Plate: (sx, sy) mm ]
        [ -- {event_text} ]'. Other call sites push their event text into
        ui_helpers.set_title_event_text() instead of writing the title directly,
        which prevents FPS clobbering and product-name spelling oscillation.
        """
        try:
            ctx = _app_ctx.ctx
            if ctx is None:
                return

            from kivy.core.window import Window
            from ui.ui_helpers import get_title_event_text

            scope_display = self.ids.get('scope_display_id')
            if scope_display:
                capture_fps = scope_display._capture_fps_value
                display_fps = scope_display._display_fps_value
                title = f'LumaViewPro {ctx.version} -- Capture: {capture_fps:.0f} | Display: {display_fps:.0f} FPS'
                if ctx.engineering_mode:
                    mbps = scope_display._camera_mbps
                    title += f' | Camera: {mbps:.1f} MB/s'

                # Cursor XY readouts -- pixel + plate coords when mouse
                # hovers the live view. Restored after d423d3c's
                # single-owner pattern dropped them. (#638)
                if self._mouse_over_image:
                    title += f'   |   Pixel: ({self._mouse_pixel_x}, {self._mouse_pixel_y})'
                    try:
                        from modules.config_ui_getters import (
                            get_binning_from_ui,
                            get_selected_labware,
                        )

                        _, objective = _app_ctx.ctx.session.get_current_objective_info()
                        pixel_size_um = config_ui_getters.get_pixel_size(
                            focal_length=objective['focal_length'],
                            binning_size=get_binning_from_ui(),
                        )
                        # The plate (um) readout converts a cursor offset into a
                        # stage distance; it needs both a connected motor and a
                        # known pixel size. Without either, the pixel readout
                        # above stands alone -- never an invented distance.
                        if ctx.lumaview.scope.motor_connected and pixel_size_um is not None:
                            # _mouse_pixel_* are sensor-pixel coords (full frame);
                            # center on the full-resolution frame, not the
                            # downscaled preview texture.
                            frame_w, frame_h = scope_display.full_resolution_frame_size()
                            dx_um = (self._mouse_pixel_x - frame_w / 2) * pixel_size_um
                            dy_um = (self._mouse_pixel_y - frame_h / 2) * pixel_size_um
                            pos = ctx.lumaview.scope.motion.get_current_position(axis=None)
                            _, labware = get_selected_labware()
                            px, py = ctx.coordinate_transformer.stage_to_plate(
                                labware=labware,
                                stage_offset=ctx.settings['stage_offset'],
                                sx=pos['X'] + dx_um,
                                sy=pos['Y'] - dy_um,
                            )
                            title += f'   |   Plate: ({px:.2f}, {py:.2f}) mm'
                    except Exception:
                        pass

                event_text = get_title_event_text()
                if event_text:
                    title += f'   --   {event_text}'
                Window.set_title(title)
        except Exception as e:
            logger.debug(f'[LVP Main  ] Status bar update failed: {e}')

    def current_false_color(self) -> str:
        return self._false_color

    def update_shader(self, false_color='BF'):
        # logger.info('[LVP Main  ] ShaderViewer.update_shader()')

        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = (self.black,) * 4
        c['gamma'] = 2.2

        if false_color == 'Red':
            c['white_point'] = (self.white, 0.0, 0.0, 1.0)
        elif false_color == 'Green':
            c['white_point'] = (0.0, self.white, 0.0, 1.0)
        elif false_color in ('Blue', 'Lumi'):
            c['white_point'] = (0.0, 0.0, self.white, 1.0)
        else:
            c['white_point'] = (self.white,) * 4

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value


Factory.register('ShaderViewer', cls=ShaderViewer)
