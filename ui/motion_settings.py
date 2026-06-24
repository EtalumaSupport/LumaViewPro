# Copyright Etaluma, Inc.
import logging

import numpy as np
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
from modules import gui_logger
from modules.config_ui_getters import get_current_objective_info, get_selected_labware
from modules.debounce import debounce
from modules.sequential_io_executor import IOTask
from ui.ui_helpers import move_absolute_position, move_home, move_relative_position
from ui.image_settings import AccordionItemXyStageControl

logger = logging.getLogger('LVP.ui.motion_settings')


# ============================================================================
# MotionSettings -- Left Sidebar Panel (Motion, Protocol, Post-Processing)
# ============================================================================


class MotionSettings(BoxLayout):
    settings_width = dp(300)
    tab_width = dp(30)

    # Canonical top-to-bottom display order for the LEFT-side accordion.
    # Mirrors the right-side _LAYER_DISPLAY_ORDER pattern in
    # ui/image_settings.py. Used by _resort_accordion() so live
    # scope-model transitions (LS850 <-> LS820 <-> LS620, etc.) keep the
    # accordion items in canonical order regardless of which were
    # hidden / re-shown along the way (UI-1 left-side follow-up,
    # 2026-05-03).
    _LAYER_DISPLAY_ORDER = (
        'microscope',
        'objective',
        'xystage',
        'protocol',
        'postproc',
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] MotionSettings.__init__()')
        self._accordion_item_xystagecontrol = AccordionItemXyStageControl()
        self._accordion_item_xystagecontrol_visible = False
        # Objective Control accordion is defined inline in lumaviewpro.kv
        # (wrapping VerticalControl); its widget ref lives in self.ids and
        # is resolved lazily on first show/hide. Visible-by-default here
        # matches the kv starting state; set_objective_control_visibility
        # hides it for scopes that declare Focus=false (LS560/LS620 -- no
        # motorised Z axis).
        self._accordion_item_objective_control = None
        self._accordion_item_objective_control_visible = True
        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)

    def _init_ui(self, dt=0):
        if _app_ctx.ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > 50:
                logger.error(
                    '[LVP Main  ] MotionSettings._init_ui: ctx still None after 50 retries, giving up'
                )
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        self.enable_ui_features_for_engineering_mode()

    def enable_ui_features_for_engineering_mode(self):
        ENGINEERING_MODE = _app_ctx.ctx.engineering_mode
        if ENGINEERING_MODE:
            # for layer in common_utils.get_layers():
            ps = _app_ctx.ctx.motion_settings.ids['protocol_settings_id']
            ps.ids['protocol_disable_image_saving_box_id'].opacity = 1
            ps.ids['protocol_disable_image_saving_box_id'].height = '30dp'
            ps.ids['protocol_disable_image_saving_id'].height = '30dp'
            ps.ids['protocol_disable_image_saving_label_id'].height = '30dp'

            _app_ctx.ctx.motion_settings.ids['microscope_settings_id'].ids[
                'enable_bullseye_box_id'
            ].height = '30dp'
            _app_ctx.ctx.motion_settings.ids['microscope_settings_id'].ids[
                'enable_bullseye_box_id'
            ].opacity = 1

    def accordion_collapse(self):
        logger.info('[LVP Main  ] MotionSettings.accordion_collapse()')

        ctx = _app_ctx.ctx
        stage = ctx.stage

        # Handles removing/adding the stage display depending on whether or not the accordion item is visible
        protocol_accordion_item = self.ids['motionsettings_protocol_accordion_id']
        protocol_stage_widget_parent = self.ids['protocol_settings_id'].ids[
            'protocol_stage_holder_id'
        ]
        xystage_widget_parent = self._accordion_item_xystagecontrol.ids['xy_stagecontrol_id'].ids[
            'xy_stage_holder_id'
        ]

        # Determine which accordion is open
        protocol_open = protocol_accordion_item.collapse is False
        xystage_open = self._accordion_item_xystagecontrol.collapse is False

        # If switching between accordions, move the stage instantly
        if protocol_open or xystage_open:
            # Store current parent
            current_parent = stage.parent
            target_parent = protocol_stage_widget_parent if protocol_open else xystage_widget_parent

            # Only move if parent is changing
            if current_parent != target_parent:
                # Remove from current parent
                if current_parent is not None:
                    stage.remove_parent()

                # Add to new parent with consistent settings
                stage.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
                stage.size_hint = (1, 1)
                target_parent.add_widget(stage)

                # Use lightweight redraw that preserves FBO cache
                # This avoids regenerating the entire stage visualization
                stage.draw_labware(full_redraw=False)
        else:
            # Both closed - remove stage
            stage.remove_parent()

    def set_xystage_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_xystage_control()
        else:
            self._hide_xystage_control()

    def _show_xystage_control(self):
        if not self._accordion_item_xystagecontrol_visible:
            self._accordion_item_xystagecontrol_visible = True
            self.ids['motionsettings_accordion_id'].add_widget(
                self._accordion_item_xystagecontrol, 2
            )
            self._resort_accordion()

    def _hide_xystage_control(self):
        if self._accordion_item_xystagecontrol_visible:
            self._accordion_item_xystagecontrol_visible = False
            self.ids['motionsettings_accordion_id'].remove_widget(
                self._accordion_item_xystagecontrol
            )

    def set_objective_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_objective_control()
        else:
            self._hide_objective_control()

    def _resolve_objective_accordion(self):
        """Return the Objective Control accordion widget, resolving from
        self.ids on first use. `set_ui_features_for_scope()` runs during
        load_settings; lazy resolution mirrors the PC accordion pattern
        (LVP a0e2eb3) so the initial hide works before any eager ref
        would be captured.
        """
        if self._accordion_item_objective_control is None:
            self._accordion_item_objective_control = self.ids.get('objective_control_accordion_id')
        return self._accordion_item_objective_control

    def _show_objective_control(self):
        widget = self._resolve_objective_accordion()
        if widget is not None and not self._accordion_item_objective_control_visible:
            self._accordion_item_objective_control_visible = True
            self.ids['motionsettings_accordion_id'].add_widget(widget)
            self._resort_accordion()

    def _hide_objective_control(self):
        widget = self._resolve_objective_accordion()
        if widget is not None and self._accordion_item_objective_control_visible:
            widget.collapse = True
            self._accordion_item_objective_control_visible = False
            self.ids['motionsettings_accordion_id'].remove_widget(widget)

    def _resort_accordion(self):
        """Rebuild the left-side accordion children list in canonical order.

        Mirrors ui/image_settings.ImageSettings._resort_accordion. Live
        scope-model transitions (LS850 <-> LS620 <-> LS820) re-add
        previously hidden accordion items via add_widget -- and after
        multiple switches the children list ends up out of canonical
        order (e.g. Objective Control re-shown ends up at the bottom
        instead of below Microscope Settings). Called from every
        ``_show_*`` path after the add_widget call. Walks
        ``_LAYER_DISPLAY_ORDER`` forward -- Kivy renders children[0]
        last (= bottom), so the first canonical layer added ends up
        at children[-1] = TOP of the visual accordion.
        """
        accordion = self.ids.get('motionsettings_accordion_id') if hasattr(self, 'ids') else None
        if accordion is None:
            return

        widget_for_layer = {
            'microscope': self.ids.get('motionsettings_microscope_accordion_id'),
            'objective': self._resolve_objective_accordion(),
            'xystage': self._accordion_item_xystagecontrol,
            'protocol': self.ids.get('motionsettings_protocol_accordion_id'),
            'postproc': self.ids.get('motionsettings_postprocessing_accordion_id'),
        }
        visible_for_layer = {
            'microscope': True,  # always visible (kv-defined)
            'objective': self._accordion_item_objective_control_visible,
            'xystage': self._accordion_item_xystagecontrol_visible,
            'protocol': True,  # always visible (kv-defined)
            'postproc': True,  # always visible (kv-defined)
        }

        # Snapshot current children. Anything that's NOT in the
        # canonical-order map is an untracked accordion item (e.g. the
        # ``etaluma_engineering`` plugin tab, registered at runtime
        # AFTER kv build). Untracked items belong at the BOTTOM of the
        # display per Eric 2026-05-03 -- they're auxiliary surfaces, not
        # primary navigation.
        #
        # Kivy gotcha (caught 2026-05-03 via runtime diagnostic): the
        # ``self.ids.get('foo_id')`` lookup returns a Kivy ``WeakProxy``
        # -- ``id(weakproxy) != id(real_widget)``. So a tracked-set keyed
        # on Python ``id()`` matched ONLY widgets that were stored
        # directly as instance attributes (xystage), and the four
        # kv-id-resolved widgets (microscope / objective / protocol /
        # postproc) were misclassified as untracked. They got re-added
        # in their pre-resort order at children index 0, which moved
        # XY Stage Control to the top instead of leaving it in slot 2.
        # Fix: compare via ``widget.uid`` -- Kivy's stable per-widget
        # integer that proxies correctly through WeakProxy.
        tracked_uids = {w.uid for w in widget_for_layer.values() if w is not None}
        # Capture untracked widgets in their pre-resort order -- they
        # render in REVERSE children order, so children[0] is the
        # bottom-most in the display today.
        untracked_in_display_order = [
            w for w in list(reversed(accordion.children)) if w.uid not in tracked_uids
        ]
        present_tracked = [w for w in list(accordion.children) if w.uid in tracked_uids]
        for widget in present_tracked:
            accordion.remove_widget(widget)
        for widget in untracked_in_display_order:
            try:
                accordion.remove_widget(widget)
            except Exception:
                pass

        # Re-add tracked widgets first in canonical order. Each
        # ``add_widget`` (no index) PREPENDS to children -- Kivy
        # renders children[0] LAST (= bottom). So forward iteration
        # over canonical order lands the FIRST canonical layer
        # (microscope) at children[-1] = visual top, and the LAST
        # canonical layer (postproc) at children[0] = visual bottom.
        for layer in self._LAYER_DISPLAY_ORDER:
            if not visible_for_layer.get(layer, False):
                continue
            widget = widget_for_layer.get(layer)
            if widget is None:
                continue
            if widget.parent is not None:
                try:
                    widget.parent.remove_widget(widget)
                except Exception:
                    pass
            accordion.add_widget(widget)

        # Append untracked widgets at the bottom of the display.
        # ``add_widget(w, 0)`` inserts at children index 0 -> renders
        # LAST = bottom. We process untracked items in their original
        # display order, so the first untracked one ends up just below
        # 'postproc' and any subsequent untracked items below that.
        for widget in untracked_in_display_order:
            if widget.parent is not None:
                try:
                    widget.parent.remove_widget(widget)
                except Exception:
                    pass
            accordion.add_widget(widget, 0)

    def set_turret_control_visibility(self, visible: bool) -> None:
        vert_control = self.ids['verticalcontrol_id']
        for turret_id in ('turret_selection_label', 'turret_btn_box'):
            vert_control.ids[turret_id].visible = visible

        vert_control.ids['set_turret_objective_btn'].disabled = not visible
        vert_control.ids['set_turret_objective_btn'].opacity = 1 if visible else 0
        vert_control.ids['reset_turret_objective_btn'].disabled = not visible
        vert_control.ids['reset_turret_objective_btn'].opacity = 1 if visible else 0

    def set_tiling_control_visibility(self, visible: bool) -> None:
        vert_control = self.ids['protocol_settings_id']

        if visible:
            vert_control.ids['tiling_size_spinner'].disabled = False
            vert_control.ids['tiling_size_spinner'].opacity = 1
            vert_control.ids['tiling_size_apply_id'].disabled = False
            vert_control.ids['tiling_size_apply_id'].opacity = 1
            vert_control.ids['tiling_box_label_id'].opacity = 1
        else:
            vert_control.ids['tiling_size_spinner'].text = '1x1'
            vert_control.ids['tiling_size_spinner'].disabled = True
            vert_control.ids['tiling_size_spinner'].opacity = 0
            vert_control.ids['tiling_size_apply_id'].disabled = True
            vert_control.ids['tiling_size_apply_id'].opacity = 0
            vert_control.ids['tiling_box_label_id'].opacity = 0

    # Hide (and unhide) motion settings
    def toggle_settings(self):
        logger.info('[LVP Main  ] MotionSettings.toggle_settings()')
        self.ids['verticalcontrol_id'].update_gui()
        self.ids['protocol_settings_id'].select_labware()

        # move position of motion control
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width + self.tab_width, 0
        else:
            self.pos = 0, 0

        # if scope_display.play == True:
        #     scope_display.start()

    def update_xy_stage_control_gui(self, *args, full_redraw: bool = False):
        self._accordion_item_xystagecontrol.update_gui(full_redraw=full_redraw)

    def check_settings(self, *args):
        logger.info('[LVP Main  ] MotionSettings.check_settings()')
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width + self.tab_width, 0
        else:
            self.pos = 0, 0


# ============================================================================
# XYStageControl -- XY Stage Movement and Bookmarks
# ============================================================================


class XYStageControl(BoxLayout):
    def update_gui(self, dt=0, full_redraw: bool = False):
        ctx = _app_ctx.ctx
        if ctx.sequenced_capture_runner.run_in_progress():
            # During protocol: update crosshair directly from position cache
            # (zero serial I/O). Don't go through IO executor -- its callback
            # runs on a worker thread which can't touch Kivy widgets.
            result = self.get_xy_targets()
            self.get_targets_ui_callback(result=result)
            return
        # Normal (non-protocol): query via IO executor as before
        ctx.io_executor.put(
            IOTask(
                action=self.get_xy_targets, callback=self.get_targets_ui_callback, pass_result=True
            )
        )

    def get_xy_targets(self):
        ctx = _app_ctx.ctx
        scope = ctx.lumaview.scope
        # Cold-start without motor leaves axis_travel_limits_um empty;
        # gate on has_xy_stage so the KeyError doesn't get swallowed by
        # the broad except below into a misleading "Error talking to
        # Motor board" log line.
        if not scope.capabilities.has_xy_stage:
            return None
        try:
            x_target = scope.motion.get_target_position('X')
            x_target = np.clip(x_target, 0, scope.capabilities.axis_travel_limits_um['X'])
            y_target = scope.motion.get_target_position('Y')
            y_target = np.clip(y_target, 0, scope.capabilities.axis_travel_limits_um['Y'])
        except Exception:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
            return None

        return (x_target, y_target)

    def get_targets_ui_callback(self, result=None, exception=None):
        ctx = _app_ctx.ctx
        if result is not None:
            x_target = result[0]
            y_target = result[1]

            # Convert from plate position to stage position
            _, labware = get_selected_labware()
            # Periodic Clock-tick callback -- short-circuit when no labware
            # is selected. Without this guard, the coord transform would
            # raise NoLabwareSelectedError on every tick (~1 Hz), filling
            # the log with identical tracebacks (#634 cluster fallout --
            # log showed 24x in one startup). Steady-state empty-selection
            # is the default first-launch state; it's not a user-action
            # error and shouldn't notify.
            if labware is None:
                return
            settings = ctx.settings
            coordinate_transformer = ctx.coordinate_transformer
            stage_x, stage_y = coordinate_transformer.stage_to_plate(
                labware=labware, stage_offset=settings['stage_offset'], sx=x_target, sy=y_target
            )

            if not self.ids['x_pos_id'].focus:
                # Cache text to prevent redundant ScrollView updates
                new_x_text = format(max(0, stage_x), '.2f')
                if self.ids['x_pos_id'].text != new_x_text:
                    self.ids['x_pos_id'].text = new_x_text  # Update x position text box

            if not self.ids['y_pos_id'].focus:
                new_y_text = format(max(0, stage_y), '.2f')
                if self.ids['y_pos_id'].text != new_y_text:
                    self.ids['y_pos_id'].text = new_y_text  # Update y position text box

    def _xy_jog(self, axis: str, direction: int, coarse: bool):
        """Shared XY-axis jog handler.

        Args:
            axis: 'X' or 'Y'.
            direction: +1 or -1.
            coarse: True for coarse step, False for fine step.
        """
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        dir_names = {('X', 1): 'RIGHT', ('X', -1): 'LEFT', ('Y', 1): 'FWD', ('Y', -1): 'BACK'}
        label = f'XY_{"COARSE" if coarse else "FINE"}_{dir_names[(axis, direction)]}'
        gui_logger.button(label)
        logger.info(f'[LVP Main  ] XYStageControl._xy_jog({label})')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] {label}: no objective info: {e}')
            return
        step = objective['xy_coarse' if coarse else 'xy_fine']
        move_relative_position(axis, direction * step)

    @debounce(0.2)
    def fine_left(self):
        self._xy_jog('X', -1, coarse=False)

    @debounce(0.2)
    def fine_right(self):
        self._xy_jog('X', +1, coarse=False)

    @debounce(0.2)
    def coarse_left(self):
        self._xy_jog('X', -1, coarse=True)

    @debounce(0.2)
    def coarse_right(self):
        self._xy_jog('X', +1, coarse=True)

    @debounce(0.2)
    def fine_back(self):
        self._xy_jog('Y', -1, coarse=False)

    @debounce(0.2)
    def fine_fwd(self):
        self._xy_jog('Y', +1, coarse=False)

    @debounce(0.2)
    def coarse_back(self):
        self._xy_jog('Y', -1, coarse=True)

    @debounce(0.2)
    def coarse_fwd(self):
        self._xy_jog('Y', +1, coarse=True)

    def set_xposition(self, x_pos):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.set_xposition()')
        try:
            x_pos = float(x_pos)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid X position input: {x_pos!r}')
            return
        gui_logger.button('SET_X_POSITION', f'plate_mm={x_pos:.3f}')

        # x_pos is the the plate position in mm
        # Find the coordinates for the stage
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        stage_x, _ = coordinate_transformer.plate_to_stage(
            labware=labware, stage_offset=settings['stage_offset'], px=x_pos, py=0
        )

        logger.info(f'[LVP Main  ] X pos {x_pos} Stage X {stage_x}')

        # Move to x-position
        move_absolute_position('X', stage_x)

    def set_yposition(self, y_pos):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.set_yposition()')

        try:
            y_pos = float(y_pos)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid Y position input: {y_pos!r}')
            return
        gui_logger.button('SET_Y_POSITION', f'plate_mm={y_pos:.3f}')

        # y_pos is the the plate position in mm
        # Find the coordinates for the stage
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        _, stage_y = coordinate_transformer.plate_to_stage(
            labware=labware, stage_offset=settings['stage_offset'], px=0, py=y_pos
        )

        # Move to y-position
        move_absolute_position('Y', stage_y)

    def set_xbookmark(self):
        gui_logger.button('SET_X_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.set_xbookmark()')
        ctx.io_executor.put(IOTask(action=self.ex_set_xbookmark))

    def ex_set_xbookmark(self):
        ctx = _app_ctx.ctx

        # Get current stage x-position in um
        x_pos = ctx.lumaview.scope.motion.get_current_position('X')

        # Save plate x-position to settings
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        plate_x, _ = coordinate_transformer.stage_to_plate(
            labware=labware, stage_offset=settings['stage_offset'], sx=x_pos, sy=0
        )

        settings['bookmark']['x'] = plate_x

    def set_ybookmark(self):
        gui_logger.button('SET_Y_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.set_ybookmark()')

        ctx.io_executor.put(IOTask(action=self.ex_set_ybookmark))

    def ex_set_ybookmark(self):
        ctx = _app_ctx.ctx
        y_pos = ctx.lumaview.scope.motion.get_current_position('Y')  # Get current y pos in um

        # Save plate y-position to settings
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        _, plate_y = coordinate_transformer.stage_to_plate(
            labware=labware, stage_offset=settings['stage_offset'], sx=0, sy=y_pos
        )

        settings['bookmark']['y'] = plate_y

    def goto_xbookmark(self):
        gui_logger.button('GOTO_X_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.goto_xbookmark()')

        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer

        # Get bookmark plate x-position in mm
        x_pos = settings['bookmark']['x']

        # Move to x-position
        _, labware = get_selected_labware()
        stage_x, _ = coordinate_transformer.plate_to_stage(
            labware=labware, stage_offset=settings['stage_offset'], px=x_pos, py=0
        )
        move_absolute_position('X', stage_x)

    def goto_ybookmark(self):
        gui_logger.button('GOTO_Y_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.goto_ybookmark()')

        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer

        # Get bookmark plate y-position in mm
        y_pos = settings['bookmark']['y']

        # Move to y-position
        _, labware = get_selected_labware()
        _, stage_y = coordinate_transformer.plate_to_stage(
            labware=labware, stage_offset=settings['stage_offset'], px=0, py=y_pos
        )
        move_absolute_position('Y', stage_y)  # set current y position in um

    # def calibrate(self):
    #     logger.info('[LVP Main  ] XYStageControl.calibrate()')
    #     global lumaview
    #     x_pos = lumaview.scope.get_current_position('X')  # Get current x position in um
    #     y_pos = lumaview.scope.get_current_position('Y')  # Get current x position in um

    #     _, labware = get_selected_labware()
    #     x_plate_offset = labware.plate['offset']['x']*1000
    #     y_plate_offset = labware.plate['offset']['y']*1000

    #     settings['stage_offset']['x'] = x_plate_offset-x_pos
    #     settings['stage_offset']['y'] = y_plate_offset-y_pos
    #     self.update_gui()

    @debounce(1.0)
    def home(self):
        try:
            gui_logger.button('HOME_XY')
            ctx = _app_ctx.ctx
            if ctx.protocol_running.is_set():
                return
            logger.info('[LVP Main  ] XYStageControl.home()')

            if ctx.lumaview.scope.motor_connected:  # motor controller is actively connected
                move_home(axis='ALL')

                # Firmware seems to move the turret back to position 1 when performing XY homing
                # Use this command to make sure the UI is in-sync
                ctx.motion_settings.ids['verticalcontrol_id'].turret_select(selected_position=1)

            else:
                logger.warning('[LVP Main  ] Motion controller not available.')
        except Exception as e:
            logger.error(f'[UI] home failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))
