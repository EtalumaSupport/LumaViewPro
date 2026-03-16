# Copyright Etaluma, Inc.
import logging
import time

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scatter import Scatter

import modules.app_context as _app_ctx
import modules.common_utils as common_utils

logger = logging.getLogger('LVP.ui.shader')


# -----------------------------------------------------------------------------
# Shader code
# Based on code from the kivy example Live Shader Editor found at:
# kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html
# -----------------------------------------------------------------------------
fs_header = '''
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
'''

vs_header = '''
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
'''


# ============================================================================
# ShaderViewer — GPU Shader-Based Image Display with Pan/Zoom
# ============================================================================

class ShaderViewer(Scatter):
    black = ObjectProperty(0.)
    white = ObjectProperty(1.)

    fs = StringProperty('''
void main (void) {
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
    //gl_FragColor = pow(glFragColor.rgb, 1/gamma)
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
''')


    def __init__(self, **kwargs):
        super(ShaderViewer, self).__init__(**kwargs)
        logger.debug('[LVP Main  ] ShaderViewer.__init__()')
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        self.white = 1.
        self.black = 0.

        Window.bind(on_key_up=self._key_up)
        Window.bind(on_key_down=self._key_down)

        self._track_keys = ['ctrl', 'shift']
        self._active_key_presses = set()

        # Status bar: mouse position tracking (throttled to ~5 Hz)
        self._status_bar_trigger = Clock.create_trigger(self._update_status_bar, 0.2, interval=True)
        self._status_bar_trigger()
        self._mouse_pixel_x = -1
        self._mouse_pixel_y = -1
        self._mouse_over_image = False
        Window.bind(mouse_pos=self._on_mouse_pos)

        # Scroll-to-focus: accumulate scroll ticks and debounce into single move
        self._scroll_z_pending = 0.0          # Accumulated Z delta (µm)
        self._scroll_z_trigger = Clock.create_trigger(self._flush_scroll_z, 0.05)
        self._scroll_last_time = 0.0          # monotonic time of last scroll event
        self._scroll_inertia_window = 0.15    # seconds — scrolls faster than this get multiplied


    def _key_up(self, *args):
        if len(args) < 5: # No modifiers present
            self._active_key_presses.clear()
            return

        modifiers = args[4]
        for key in self._track_keys:
            if (key not in modifiers) and (key in self._active_key_presses):
                self._active_key_presses.remove(key)


    def _key_down(self, *args):
        modifiers = args[4]
        for key in self._track_keys:
            if (key in modifiers) and (key not in self._active_key_presses):
                self._active_key_presses.add(key)


    def on_touch_down(self, touch, *args):
        logger.info('[LVP Main  ] ShaderViewer.on_touch_down()')
        from modules.config_ui_getters import get_current_objective_info
        ctx = _app_ctx.ctx

        ZOOM_BLOCKERS = [ctx.image_settings, ctx.motion_settings]
        x, y = touch.pos

        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:

            for w in ZOOM_BLOCKERS:
                lx, ly = w.to_widget(x, y)
                if w.collide_point(lx, ly):
                    return

            if 'ctrl' in self._active_key_presses:
                # Focus control — accumulate scroll ticks, debounce into single move
                if ctx.protocol_running.is_set():
                    return

                try:
                    _, objective = get_current_objective_info()
                except Exception:
                    return

                if 'shift' in self._active_key_presses:
                    step_um = objective['z_coarse']
                else:
                    step_um = objective['z_fine']

                # Inertial scaling: faster scrolling = larger steps
                now = time.monotonic()
                dt = now - self._scroll_last_time
                self._scroll_last_time = now

                if dt < self._scroll_inertia_window and dt > 0:
                    # Scale up when scrolling fast (up to 5x)
                    speed_factor = min(5.0, self._scroll_inertia_window / dt)
                else:
                    speed_factor = 1.0

                delta = step_um * speed_factor
                if touch.button == 'scrolldown':
                    self._scroll_z_pending += delta
                elif touch.button == 'scrollup':
                    self._scroll_z_pending -= delta

                # Reset the debounce trigger — fires 50ms after last scroll event
                self._scroll_z_trigger()

            else:
                # Digital zoom control
                if touch.button == 'scrolldown':
                    if self.scale < 100:
                        self.scale = self.scale * 1.1
                elif touch.button == 'scrollup':
                    if self.scale > 1:
                        self.scale = max(1, self.scale * 0.8)
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            # Let side panels handle touches that land on them
            for w in ZOOM_BLOCKERS:
                lx, ly = w.to_widget(x, y)
                if w.collide_point(lx, ly):
                    return w.on_touch_down(touch)
            super(ShaderViewer, self).on_touch_down(touch)


    def _flush_scroll_z(self, dt):
        """Debounced scroll-to-focus: send one accumulated Z move."""
        from modules.ui_helpers import move_relative_position
        from modules.sequential_io_executor import IOTask

        delta = self._scroll_z_pending
        self._scroll_z_pending = 0.0

        if delta == 0.0:
            return

        _app_ctx.ctx.io_executor.put(IOTask(
            action=move_relative_position,
            args=('Z', delta),
            kwargs={"overshoot_enabled": False},
        ))


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
            tex_w, tex_h = scope_display.texture_size
            self._mouse_pixel_x = int((local_x - img_x_min) * tex_w / norm_w)
            # Kivy Y is bottom-up, image Y is top-down
            self._mouse_pixel_y = tex_h - 1 - int((local_y - img_y_min) * tex_h / norm_h)
            self._mouse_pixel_x = max(0, min(self._mouse_pixel_x, tex_w - 1))
            self._mouse_pixel_y = max(0, min(self._mouse_pixel_y, tex_h - 1))
            self._mouse_over_image = True
        else:
            self._mouse_over_image = False


    def _update_status_bar(self, dt):
        """Periodic status bar update (~5 Hz).

        FPS is shown in the window title bar to avoid overlapping the side
        panel.  Pixel and plate coordinates are shown in the on-image overlay
        only when the mouse is hovering over the live view.
        """
        try:
            from kivy.core.window import Window
            from modules.config_ui_getters import get_current_objective_info, get_binning_from_ui, get_selected_labware
            ctx = _app_ctx.ctx
            status_label = ctx.lumaview.ids.get('status_bar_id')

            scope_display = self.ids.get('scope_display_id')
            fps = scope_display._fps_value if scope_display else 0

            # Build title bar: FPS + XY location (on hover)
            current_title = Window.title
            base_title = f"Lumaview Pro {ctx.version}"
            temp_statuses = ('Homing', 'Recording', 'Writing')
            has_temp_status = any(s in current_title for s in temp_statuses)

            if not has_temp_status:
                title_parts = [f"FPS: {fps:.1f}"]

                if self._mouse_over_image:
                    title_parts.append(f'Pixel: ({self._mouse_pixel_x}, {self._mouse_pixel_y})')

                    # Convert pixel offset from image center to plate coordinates
                    try:
                        _, objective = get_current_objective_info()
                        pixel_size_um = common_utils.get_pixel_size(
                            focal_length=objective['focal_length'],
                            binning_size=get_binning_from_ui(),
                        )
                        tex_w, tex_h = scope_display.texture_size
                        dx_px = self._mouse_pixel_x - tex_w / 2
                        dy_px = self._mouse_pixel_y - tex_h / 2
                        dx_um = dx_px * pixel_size_um
                        dy_um = dy_px * pixel_size_um

                        if ctx.lumaview.scope.motion.driver:
                            pos = ctx.lumaview.scope.get_current_position(axis=None)
                            cursor_sx = pos['X'] + dx_um
                            cursor_sy = pos['Y'] - dy_um
                            _, labware = get_selected_labware()
                            px, py = ctx.coordinate_transformer.stage_to_plate(
                                labware=labware,
                                stage_offset=ctx.settings['stage_offset'],
                                sx=cursor_sx,
                                sy=cursor_sy,
                            )
                            title_parts.append(f'Plate: ({px:.2f}, {py:.2f}) mm')
                    except Exception:
                        pass

                Window.set_title(f"{base_title}   |   {'   |   '.join(title_parts)}")

            # Clear the on-image overlay (coordinates now in title bar)
            if status_label is not None:
                status_label.text = ''
        except Exception:
            pass


    def current_false_color(self) -> str:
        return self._false_color


    def update_shader(self, false_color='BF'):
        # logger.info('[LVP Main  ] ShaderViewer.update_shader()')

        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = (self.black, )*4
        c['gamma'] = 2.2

        if false_color == 'Red':
            c['white_point'] = (self.white, 0., 0., 1.)
        elif false_color == 'Green':
            c['white_point'] = (0., self.white, 0., 1.)
        elif false_color in ('Blue', 'Lumi'):
            c['white_point'] = (0., 0., self.white, 1.)
        else:
            c['white_point'] = (self.white, )*4

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value


Factory.register('ShaderViewer', cls=ShaderViewer)


class ShaderEditor(BoxLayout):
    fs = StringProperty('''
void main (void){
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
''')

    viewer = ObjectProperty(None)
    hide_editor = ObjectProperty(None)
    hide_editor = True


    def __init__(self, **kwargs):
        super(ShaderEditor, self).__init__(**kwargs)
        logger.info('[LVP Main  ] ShaderEditor.__init__()')
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        logger.info('[LVP Main  ] ShaderEditor.compile_shaders()')
        if not self.viewer:
            logger.warning('[LVP Main  ] ShaderEditor.compile_shaders() Fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        self.viewer.fs = fs
        self.viewer.vs = vs

    # Hide (and unhide) Shader settings
    def toggle_editor(self):
        logger.info('[LVP Main  ] ShaderEditor.toggle_editor()')
        if not self.hide_editor:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0
