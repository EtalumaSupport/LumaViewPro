# Copyright Etaluma, Inc.
import logging
from functools import partial

import numpy as np

from kivy.clock import Clock
from kivy.graphics import Color, Line, Rectangle, Ellipse, Fbo
from kivy.uix.widget import Widget

import modules.app_context as _app_ctx
from modules.config_ui_getters import get_selected_labware
from modules.sequential_io_executor import IOTask
from modules.step_navigation import go_to_step
from modules.ui_helpers import find_nearest_step, move_absolute_position

logger = logging.getLogger('LVP.ui.stage')


class Stage(Widget):

    def _triggered_full_redraw(self, *args):
        """Triggered version of full_redraw to debounce rapid events."""
        self.full_redraw()

    def full_redraw(self, *args):
        # Invalidate FBO caches on size/position change
        self._labware_fbos.clear()
        self._step_locations_fbo = None
        self._cached_step_locations_hash = None
        self.draw_labware(full_redraw=True)


    def remove_parent(self):
        ctx = _app_ctx.ctx
        if self.parent is not None:
            self.parent.remove_widget(ctx.stage)


    def get_id(self):
        return id(self)


    def __init__(self, **kwargs):
        super(Stage, self).__init__(**kwargs)
        logger.debug('[LVP Main  ] Stage.__init__()')
        self.ROI_min = [0,0]
        self.ROI_max = [0,0]
        self._motion_enabled = True
        self.ROIs = []

        # Track labware state for smart redraws
        self._cached_labware_name = None
        self._labware_fbos = {}
        self._step_locations_fbo = None
        self._cached_step_locations_hash = None

        # Stage Coordinates (defaults; overridden by motorconfig at draw time)
        self.STAGE_W = 120
        self.STAGE_H = 80

        # Use a trigger to debounce redraw calls and prevent memory leaks from excessive events
        self._redraw_trigger = Clock.create_trigger(self._triggered_full_redraw, 0.1)
        self.full_redraw()
        self.bind(
            pos=self._redraw_trigger,
            size=self._redraw_trigger
        )
        self._protocol_step_locations_df = None
        self._protocol_step_redraw = False
        self._protocol_step_locations_show = False

        self._prev_x_target = None
        self._prev_y_target = None
        self._prev_x_current = None
        self._prev_y_current = None

        # Persistent canvas objects for crosshairs and selected well
        # These are created once and properties are updated to avoid memory accumulation
        self._selected_well_color = None
        self._selected_well_line = None
        self._crosshair_color = None
        self._crosshair_h_line = None
        self._crosshair_v_line = None
        self._create_persistent_canvas_objects()

    def _stage_limits_um(self):
        """Return (x_max_um, y_max_um) from motorconfig, with fallback defaults."""
        ctx = _app_ctx.ctx
        if ctx is not None and hasattr(ctx, 'scope') and ctx.scope is not None:
            return (ctx.scope.travel_limit_um('X'), ctx.scope.travel_limit_um('Y'))
        from modules.common_utils import DEFAULT_STAGE_TRAVEL_UM
        return (DEFAULT_STAGE_TRAVEL_UM["x"], DEFAULT_STAGE_TRAVEL_UM["y"])


    def _create_persistent_canvas_objects(self):
        """Create persistent canvas objects for crosshairs and selected well.
        These objects are updated in place rather than being removed and recreated.
        Using canvas.after ensures they're always drawn on top of everything else."""
        with self.canvas.after:
            # Selected well (green ellipse)
            self._selected_well_color = Color(0., 1., 0., 1.)
            self._selected_well_line = Line(ellipse=(0, 0, 0, 0), group='selected_well')

            # Crosshairs (red lines) - drawn last so they're on top
            self._crosshair_color = Color(1., 0., 0., 1.)
            self._crosshair_h_line = Line(points=[0, 0, 0, 0], width=1, group='crosshairs')
            self._crosshair_v_line = Line(points=[0, 0, 0, 0], width=1, group='crosshairs')

    def show_protocol_steps(self, enable: bool):
        self._protocol_step_locations_show = enable
        self._protocol_step_redraw = True


    def set_protocol_steps(self, df):
        # Filter to only keep the X/Y locations
        df = df.copy()
        df = df[['X','Y']]
        self._protocol_step_locations_df = df.drop_duplicates()
        self._protocol_step_redraw = True
        # Invalidate step locations FBO cache
        self._step_locations_fbo = None
        self._cached_step_locations_hash = None


    def append_ROI(self, x_min, y_min, x_max, y_max):
        self.ROI_min = [x_min, y_min]
        self.ROI_max = [x_max, y_max]
        self.ROIs.append([self.ROI_min, self.ROI_max])

    def set_motion_capability(self, enabled: bool):
        self._motion_enabled = enabled

    def on_touch_down(self, touch):
        logger.debug('[LVP Main  ] Stage.on_touch_down()')

        if not self._motion_enabled:
            return

        if self.collide_point(*touch.pos) and (touch.button == 'left' or touch.button == 'right'):

            # Get mouse position in pixels
            (mouse_x, mouse_y) = touch.pos

            # Convert to relative mouse position in pixels
            mouse_x = mouse_x-self.x
            mouse_y = mouse_y-self.y

            # Create current labware instance
            _, labware = get_selected_labware()

            # Get labware dimensions
            dim_max = labware.get_dimensions()

            # Scale from pixels to mm (from the bottom left)
            scale_x = dim_max['x'] / self.width
            scale_y = dim_max['y'] / self.height

            # Convert to plate position in mm (from the top left)
            plate_x = mouse_x*scale_x
            plate_y = dim_max['y'] - mouse_y*scale_y

            # Convert from plate position to stage position
            ctx = _app_ctx.ctx
            settings = ctx.settings
            coordinate_transformer = ctx.coordinate_transformer
            _, labware = get_selected_labware()
            stage_x, stage_y = coordinate_transformer.plate_to_stage(
                labware=labware,
                stage_offset=settings['stage_offset'],
                px=plate_x,
                py=plate_y
            )

            if touch.button == 'left':
                io_executor = ctx.io_executor
                io_executor.put(IOTask(action=move_absolute_position, args=('X', stage_x)))
                io_executor.put(IOTask(action=move_absolute_position, args=('Y', stage_y)))

            elif touch.button == 'right':
                try:
                    logger.info(f"[Stage   ] Finding nearest step to {plate_x}, {plate_y}")
                    step_idx = find_nearest_step(x=plate_x, y=plate_y, protocol=ctx.motion_settings.ids['protocol_settings_id']._protocol)
                    if step_idx == -1:
                        return

                    go_to_step(protocol=ctx.motion_settings.ids['protocol_settings_id']._protocol,
                               step_idx=step_idx,
                               called_from_protocol=False,
                               include_move=True)
                    logger.info(f"[Stage   ] Successfully moved to step {step_idx}")
                except Exception as e:
                    logger.error(f"[Stage   ] Error finding nearest step: {e}")
            # move_absolute_position('X', stage_x)
            # move_absolute_position('Y', stage_y)

    def draw_labware(self, *args, full_redraw: bool = False):
        if self.parent is None:
            return

        ctx = _app_ctx.ctx

        if not hasattr(ctx, 'lumaview') or ctx.lumaview is None:
            return

        if not hasattr(ctx, 'settings') or ctx.settings is None:
            return

        # Position reads are all from the push-based cache (zero serial I/O),
        # so we can run calculations directly on the main thread instead of
        # queuing onto the IO executor (which can be busy with protocol moves).
        # This ensures the crosshair updates immediately when positions change.
        self.draw_labware_io_calculations(full_redraw)

    def create_labware_fbo(self):
        """
        Create FBO for static labware rendering (wells, outlines, stage area).
        This significantly optimizes performance by pre-rendering static elements.
        """
        labware_name, labware = get_selected_labware()

        if labware_name in self._labware_fbos:
            logger.debug(f"[Stage     ] Using cached labware FBO for {labware_name}")
            return self._labware_fbos[labware_name]

        logger.debug(f"[Stage     ] Creating new labware FBO for {labware_name}, size=({int(self.width)}, {int(self.height)})")

        # Labware FBO is not cached - create new one
        fbo_width = int(self.width)
        fbo_height = int(self.height)
        fbo = Fbo(size=(fbo_width, fbo_height), with_stencilbuffer=False)

        # Bind and clear FBO
        fbo.bind()

        # Clear to transparent
        from kivy.graphics.opengl import glClearColor, glClear, GL_COLOR_BUFFER_BIT
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)

        fbo.release()

        settings = _app_ctx.ctx.settings
        coordinate_transformer = _app_ctx.ctx.coordinate_transformer

        # Now draw to the FBO
        with fbo:
            # Get dimensions for drawing - use exact FBO size
            w = float(fbo_width)
            h = float(fbo_height)
            x = 0  # FBO coordinates are relative to FBO, not screen
            y = 0

            # Get labware dimensions
            dim_max = labware.get_dimensions()

            # mm to pixels scale
            scale_x = w/dim_max['x']
            scale_y = h/dim_max['y']

            # Stage Coordinates from motorconfig
            x_max, y_max = self._stage_limits_um()
            stage_w = x_max / 1000.0
            stage_h = y_max / 1000.0

            stage_x = settings['stage_offset']['x']/1000
            stage_y = settings['stage_offset']['y']/1000

            # Draw stage area outline
            Color(.2, .2, .2, 0.5)
            Rectangle(
                pos=(x+(dim_max['x']-stage_w-stage_x)*scale_x, y+stage_y*scale_y),
                size=(stage_w*scale_x, stage_h*scale_y)
            )

            # Draw plate outline from above
            # Add 0.5 pixel offset to prevent edge clipping (lines are centered on coords)
            Color(50/255, 164/255, 206/255, 1.)  # kivy aqua
            Line(points=(x+0.5, y+0.5, x+0.5, y+h-15), width=1)          # Left
            Line(points=(x+w-0.5, y+0.5, x+w-0.5, y+h-0.5), width=1)         # Right
            Line(points=(x+0.5, y+0.5, x+w-0.5, y+0.5), width=1)             # Bottom
            Line(points=(x+15, y+h-0.5, x+w-0.5, y+h-0.5), width=1)      # Top
            Line(points=(x+0.5, y+h-15, x+15, y+h-0.5), width=1)     # Diagonal

            # Draw ROI rectangle if set
            if self.ROI_max[0] > self.ROI_min[0]:
                roi_min_x, roi_min_y = coordinate_transformer.stage_to_pixel(
                    labware=labware,
                    stage_offset=settings['stage_offset'],
                    sx=self.ROI_min[0],
                    sy=self.ROI_min[1],
                    scale_x=scale_x,
                    scale_y=scale_y
                )

                roi_max_x, roi_max_y = coordinate_transformer.stage_to_pixel(
                    labware=labware,
                    stage_offset=settings['stage_offset'],
                    sx=self.ROI_max[0],
                    sy=self.ROI_max[1],
                    scale_x=scale_x,
                    scale_y=scale_y
                )

                Color(50/255, 164/255, 206/255, 1.)
                Line(rectangle=(x+roi_min_x, y+roi_min_y, roi_max_x - roi_min_x, roi_max_y - roi_min_y), width=1)

            # Draw all wells
            cols = labware.config['columns']
            rows = labware.config['rows']

            well_spacing_x = labware.config['spacing']['x']
            well_spacing_y = labware.config['spacing']['y']

            well_diameter = labware.config['diameter']
            if well_diameter == -1:
                well_radius_pixel_x = well_spacing_x
                well_radius_pixel_y = well_spacing_y
            else:
                well_radius = well_diameter / 2
                well_radius_pixel_x = well_radius * scale_x
                well_radius_pixel_y = well_radius * scale_y

            Color(0.4, 0.4, 0.4, 0.5)
            for i in range(cols):
                for j in range(rows):
                    well_plate_x, well_plate_y = labware.get_well_position(i, j)
                    well_pixel_x, well_pixel_y = coordinate_transformer.plate_to_pixel(
                        labware=labware,
                        px=well_plate_x,
                        py=well_plate_y,
                        scale_x=scale_x,
                        scale_y=scale_y
                    )
                    # Use float for precise positioning to avoid rounding errors
                    x_center = x + well_pixel_x
                    y_center = y + well_pixel_y

                    # Draw ellipse
                    Ellipse(
                        pos=(x_center - well_radius_pixel_x, y_center - well_radius_pixel_y),
                        size=(well_radius_pixel_x * 2, well_radius_pixel_y * 2)
                    )

        # Force FBO to render
        fbo.draw()

        # Cache the FBO for this labware
        self._labware_fbos[labware_name] = fbo
        return fbo

    def create_step_locations_fbo(self):
        """
        Create FBO for protocol step locations rendering.
        This optimizes performance by pre-rendering step markers.
        """
        labware_name, labware = get_selected_labware()
        coordinate_transformer = _app_ctx.ctx.coordinate_transformer

        # Check if we need to regenerate the FBO
        if self._protocol_step_locations_df is None or not self._protocol_step_locations_show:
            return None

        # Create a hash of the step locations for cache validation
        step_hash = hash(tuple(self._protocol_step_locations_df.to_records(index=False).tolist()))

        # Return cached FBO if it's still valid
        if (self._step_locations_fbo is not None and
            self._cached_step_locations_hash == step_hash and
            self._cached_labware_name == labware_name):
            return self._step_locations_fbo

        # Create new FBO for step locations
        fbo_width = int(self.width)
        fbo_height = int(self.height)
        fbo = Fbo(size=(fbo_width, fbo_height), with_stencilbuffer=False)

        # Bind and clear FBO
        fbo.bind()

        # Clear to transparent
        from kivy.graphics.opengl import glClearColor, glClear, GL_COLOR_BUFFER_BIT
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)

        fbo.release()

        # Now draw to the FBO
        with fbo:
            # Get dimensions for drawing - use exact FBO size
            w = float(fbo_width)
            h = float(fbo_height)
            x = 0  # FBO coordinates are relative to FBO, not screen
            y = 0

            # Get labware dimensions
            dim_max = labware.get_dimensions()

            # mm to pixels scale
            scale_x = w/dim_max['x']
            scale_y = h/dim_max['y']

            # Draw protocol step markers
            half_size = 2
            Color(1., 1., 0., 1.)  # Yellow color for step markers

            for _, step in self._protocol_step_locations_df.iterrows():
                pixel_x, pixel_y = coordinate_transformer.plate_to_pixel(
                    labware=labware,
                    px=step['X'],
                    py=step['Y'],
                    scale_x=scale_x,
                    scale_y=scale_y
                )

                x_center = x + pixel_x
                y_center = y + pixel_y

                # Draw crosshair for step location
                Line(points=(x_center-half_size, y_center, x_center+half_size, y_center), width=1)  # horizontal
                Line(points=(x_center, y_center-half_size, x_center, y_center+half_size), width=1)  # vertical

        # Force FBO to render
        fbo.draw()

        # Cache the FBO
        self._step_locations_fbo = fbo
        self._cached_step_locations_hash = step_hash

        return fbo


    def draw_labware_io_calculations(self, full_redraw: bool = False):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        scope = ctx.scope

        # Try to get current and target positions - may fail if not homed yet
        position_available = False
        x_target = None
        y_target = None
        x_current = None
        y_current = None

        try:
            if scope.has_xyhomed():
                # Position cache auto-refreshes on first read if stale (>80ms)
                x_target = scope.get_target_position('X')
                y_target = scope.get_target_position('Y')
                x_max, y_max = self._stage_limits_um()
                x_current = np.clip(scope.get_current_position('X'), 0, x_max)
                y_current = np.clip(scope.get_current_position('Y'), 0, y_max)
                position_available = True
        except Exception:
            # If we can't get positions (not homed yet), we'll still draw the labware
            logger.debug('[Stage     ] Position not available yet, drawing labware only')
            position_available = False

        if not full_redraw and not self._protocol_step_redraw and position_available:
            if x_target == self._prev_x_target and y_target == self._prev_y_target and x_current == self._prev_x_current and y_current == self._prev_y_current:
                return

        # Create current labware instance
        labware_name, labware = get_selected_labware()

        # Check if labware changed - trigger full redraw
        if labware_name != self._cached_labware_name:
            full_redraw = True
            self._cached_labware_name = labware_name

        # Clear canvas based on redraw type
        if full_redraw:
            # Use Clock.schedule_once with single combined callback to reduce lambda creation
            def clear_and_recreate(_):
                self.canvas.clear()
                self.canvas.after.clear()
                self._create_persistent_canvas_objects()
            Clock.schedule_once(clear_and_recreate, 0)

        if self._protocol_step_redraw:
            Clock.schedule_once(lambda dt: self.canvas.remove_group('steps_fbo'), 0)

        w = self.width
        h = self.height
        x = self.x
        y = self.y

        # Get labware dimensions
        dim_max = labware.get_dimensions()

        # mm to pixels scale
        scale_x = w/dim_max['x']
        scale_y = h/dim_max['y']


        stage_x = settings['stage_offset']['x']/1000
        stage_y = settings['stage_offset']['y']/1000

        cols = labware.config['columns']
        rows = labware.config['rows']

        well_spacing_x = labware.config['spacing']['x']
        well_spacing_y = labware.config['spacing']['y']

        well_spacing_pixel_x = well_spacing_x
        well_spacing_pixel_y = well_spacing_y

        well_diameter = labware.config['diameter']
        if well_diameter == -1:
            well_radius_pixel_x = well_spacing_pixel_x
            well_radius_pixel_y = well_spacing_pixel_y
        else:
            well_radius = well_diameter / 2
            well_radius_pixel_x = well_radius * scale_x
            well_radius_pixel_y = well_radius * scale_y

        # Draw static labware elements using FBO (only on full redraw)
        if full_redraw:
            # Schedule FBO creation and drawing on main thread
            # Use partial to avoid lambda closure memory accumulation
            Clock.schedule_once(partial(self._draw_labware_fbo_scheduled, x, y, w, h), 0)

        # Draw protocol steps using FBO (if needed)
        if full_redraw or self._protocol_step_redraw:
            self._protocol_step_redraw = False

            # Remove old step locations if they exist
            Clock.schedule_once(lambda dt: self.canvas.remove_group('steps_fbo'), 0)

            # Capture variables by value for the lambda
            pos_x, pos_y, size_w, size_h = x, y, w, h

            # Schedule FBO creation and drawing on main thread
            # Use partial to avoid lambda closure memory accumulation
            Clock.schedule_once(partial(self._draw_steps_fbo_scheduled, x, y, w, h), 0)

        # Only draw crosshairs and selected well if position is available (after homing)
        if position_available:
            # Draw selected well (updates when target changes)
            target_plate_x, target_plate_y = coordinate_transformer.stage_to_plate(
                    labware=labware,
                    stage_offset=settings['stage_offset'],
                    sx=x_target,
                    sy=y_target
                )

            target_i, target_j = labware.get_well_index(target_plate_x, target_plate_y)
            target_well_plate_x, target_well_plate_y = labware.get_well_position(target_i, target_j)
            target_well_pixel_x, target_well_pixel_y = coordinate_transformer.plate_to_pixel(
                labware=labware,
                px=target_well_plate_x,
                py=target_well_plate_y,
                scale_x=scale_x,
                scale_y=scale_y
            )
            target_well_center_x = int(x+target_well_pixel_x) # on screen center
            target_well_center_y = int(y+target_well_pixel_y) # on screen center

            # Update selected well ellipse properties (instead of recreating)
            ellipse_params = (
                target_well_center_x - well_radius_pixel_x,
                target_well_center_y - well_radius_pixel_y,
                well_radius_pixel_x * 2,
                well_radius_pixel_y * 2
            )
            Clock.schedule_once(lambda dt, ep=ellipse_params: setattr(self._selected_well_line, 'ellipse', ep), 0)

            # Draw crosshairs (updates every frame - but only 2 lines!)
            pixel_x, pixel_y = coordinate_transformer.stage_to_pixel(
                    labware=labware,
                    stage_offset=settings['stage_offset'],
                    sx=x_current,
                    sy=y_current,
                    scale_x=scale_x,
                    scale_y=scale_y
                )

            x_center = x+pixel_x
            y_center = y+pixel_y

            # Update crosshairs properties (instead of recreating)
            h_line_points = [x_center-10, y_center, x_center+10, y_center]
            v_line_points = [x_center, y_center-10, x_center, y_center+10]
            Clock.schedule_once(lambda dt, pts=h_line_points: setattr(self._crosshair_h_line, 'points', pts), 0)
            Clock.schedule_once(lambda dt, pts=v_line_points: setattr(self._crosshair_v_line, 'points', pts), 0)

            self._prev_x_target = x_target
            self._prev_y_target = y_target
            self._prev_x_current = x_current
            self._prev_y_current = y_current
        else:
            # Hide crosshairs and selected well by setting them to zero size/empty points
            Clock.schedule_once(lambda dt: setattr(self._selected_well_line, 'ellipse', (0, 0, 0, 0)), 0)
            Clock.schedule_once(lambda dt: setattr(self._crosshair_h_line, 'points', [0, 0, 0, 0]), 0)
            Clock.schedule_once(lambda dt: setattr(self._crosshair_v_line, 'points', [0, 0, 0, 0]), 0)


    def schedule_to_draw(self, draw_function, *args, **kwargs):
        """
        Schedule a drawing operation to be executed on the main UI thread.

        Args:
            draw_function: A callable that performs the drawing operation
            *args, **kwargs: Arguments to pass to the draw_function

        Example usage:
            # From a background thread:
            self.schedule_to_draw(self.draw_line, points=[0, 0, 100, 100], color=(1, 0, 0, 1))
            self.schedule_to_draw(self.draw_circle, pos=(50, 50), radius=20)
        """
        def execute_draw(_):
            try:
                draw_function(*args, **kwargs)
            except Exception as e:
                print(f"Error in scheduled draw operation: {e}")

        Clock.schedule_once(execute_draw, 0)

    def draw_line(self, points=None, color=(1, 1, 1, 1), width=1, group=None, circle=None, ellipse=None, rectangle=None):
        """Draw a line on the canvas - safe to call from schedule_to_draw_on_canvas"""
        with self.canvas:
            Color(*color)

            if points:
                Line(points=points, width=width, group=group)
            elif circle:
                # circle = (center_x, center_y, radius)
                Line(circle=circle, group=group)
            elif ellipse:
                # ellipse = (bottom_left_x, bottom_left_y, width, height)
                Line(ellipse=ellipse, group=group)
            elif rectangle:
                # rectangle = (bottom_left_x, bottom_left_y, width, height)
                Line(rectangle=rectangle, group=group)

    def draw_rectangle(self, pos, size, color=(1, 1, 1, 1), group=None):
        """Draw a rectangle on the canvas - safe to call from schedule_to_draw_on_canvas"""
        with self.canvas:
            Color(*color)
            Rectangle(pos=pos, size=size, group=group)

    def draw_ellipse(self, pos, radius, color=(1, 1, 1, 1), group=None):
        """Draw an ellipse on the canvas - safe to call from schedule_to_draw_on_canvas"""
        with self.canvas:
            Color(*color)
            Ellipse(pos=pos, size=radius, group=group)

    def _draw_labware_fbo_scheduled(self, x, y, w, h, *args):
        """Scheduled callback for drawing labware FBO."""
        try:
            labware_fbo = self.create_labware_fbo()
            if labware_fbo and labware_fbo.texture:
                self.draw_fbo_texture(texture=labware_fbo.texture, pos=(x, y), size=(w, h), group='labware_fbo')
        except Exception as e:
            logger.exception(f"[Stage     ] Error drawing labware FBO: {e}")

    def _draw_steps_fbo_scheduled(self, x, y, w, h, *args):
        """Scheduled callback for drawing steps FBO."""
        try:
            steps_fbo = self.create_step_locations_fbo()
            if steps_fbo and steps_fbo.texture:
                self.draw_fbo_texture(texture=steps_fbo.texture, pos=(x, y), size=(w, h), group='steps_fbo')
        except Exception as e:
            logger.exception(f"[Stage     ] Error drawing steps FBO: {e}")

    def draw_fbo_texture(self, texture, pos, size, group=None):
        """Draw an FBO texture on the canvas - safe to call from schedule_to_draw_on_canvas"""
        with self.canvas:
            Color(1, 1, 1, 1)  # White color with full opacity to render texture as-is
            # Draw the FBO texture directly - Kivy handles FBO texture coordinates automatically
            Rectangle(texture=texture, pos=pos, size=size, group=group)


    def get_target_xy(self):
        scope = _app_ctx.ctx.scope
        try:
            target_stage_x = scope.get_target_position('X')
            target_stage_y = scope.get_target_position('Y')
        except Exception:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
            return None

        return (target_stage_x, target_stage_y)

    def get_target_callback(self, scale_x, scale_y, well_radius_pixel_x, x, y, result=None, exception=None):
        settings = _app_ctx.ctx.settings
        coordinate_transformer = _app_ctx.ctx.coordinate_transformer
        io_executor = _app_ctx.ctx.io_executor

        if not result is None:
            target_stage_x = result[0]
            target_stage_y = result[1]

            _, labware = get_selected_labware()
            target_plate_x, target_plate_y = coordinate_transformer.stage_to_plate(
                labware=labware,
                stage_offset=settings['stage_offset'],
                sx=target_stage_x,
                sy=target_stage_y
            )

            target_i, target_j = labware.get_well_index(target_plate_x, target_plate_y)
            target_well_plate_x, target_well_plate_y = labware.get_well_position(target_i, target_j)
            target_well_pixel_x, target_well_pixel_y = coordinate_transformer.plate_to_pixel(
                labware=labware,
                px=target_well_plate_x,
                py=target_well_plate_y,
                scale_x=scale_x,
                scale_y=scale_y
            )
            target_well_center_x = int(x+target_well_pixel_x) # on screen center
            target_well_center_y = int(y+target_well_pixel_y) # on screen center

            # Green selection circle
            with self.canvas:
                Color(0., 1., 0., 1., group='selected_well')
                Line(circle=(target_well_center_x, target_well_center_y, well_radius_pixel_x), group='selected_well')

            #  Red Crosshairs
            # ------------------
            if self._motion_enabled:
                io_executor.put(IOTask(
                    action=self.motion_enabled_io,
                    callback=self.motion_enabled_callback,
                    cb_args=(
                        scale_x,
                        scale_y,
                        x,
                        y,
                        labware
                    ),
                    pass_result=True
                ))


    def motion_enabled_io(self):
        scope = _app_ctx.ctx.scope
        try:
            x_current = scope.get_current_position('X')
            x_max, y_max = self._stage_limits_um()
            x_current = np.clip(x_current, 0, x_max)
            y_current = scope.get_current_position('Y')
            y_current = np.clip(y_current, 0, y_max)
        except Exception:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
            return None

        return (x_current, y_current)

    def motion_enabled_callback(self, scale_x, scale_y, x, y, labware, result=None, exception=None):
        settings = _app_ctx.ctx.settings
        coordinate_transformer = _app_ctx.ctx.coordinate_transformer

        if result is not None:
            x_current = result[0]
            y_current = result[1]

            # Convert stage coordinates to relative pixel coordinates
            pixel_x, pixel_y = coordinate_transformer.stage_to_pixel(
                labware=labware,
                stage_offset=settings['stage_offset'],
                sx=x_current,
                sy=y_current,
                scale_x=scale_x,
                scale_y=scale_y
            )

            x_center = x+pixel_x
            y_center = y+pixel_y
            with self.canvas:
                Color(1., 0., 0., 1., group='crosshairs')
                Line(points=(x_center-10, y_center, x_center+10, y_center), width = 1, group='crosshairs') # horizontal line
                Line(points=(x_center, y_center-10, x_center, y_center+10), width = 1, group='crosshairs') # vertical line
            x = 1
            y = 1
