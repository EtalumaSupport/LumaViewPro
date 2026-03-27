# Copyright Etaluma, Inc.
import logging

import numpy as np
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.config_ui_getters import get_current_objective_info, get_selected_labware
from modules.debounce import debounce
from modules.sequential_io_executor import IOTask
from modules.ui_helpers import move_absolute_position, move_home, move_relative_position
from ui.image_settings import AccordionItemXyStageControl

logger = logging.getLogger('LVP.ui.motion_settings')


# ============================================================================
# MotionSettings — Left Sidebar Panel (Motion, Protocol, Post-Processing)
# ============================================================================

class MotionSettings(BoxLayout):
    settings_width = dp(300)
    tab_width = dp(30)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] MotionSettings.__init__()')
        self._accordion_item_xystagecontrol = AccordionItemXyStageControl()
        self._accordion_item_xystagecontrol_visible = False
        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)


    def _init_ui(self, dt=0):
        if _app_ctx.ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > 50:
                logger.error('[LVP Main  ] MotionSettings._init_ui: ctx still None after 50 retries, giving up')
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

            _app_ctx.ctx.motion_settings.ids['microscope_settings_id'].ids['enable_bullseye_box_id'].height = '30dp'
            _app_ctx.ctx.motion_settings.ids['microscope_settings_id'].ids['enable_bullseye_box_id'].opacity = 1

    def accordion_collapse(self):
        logger.info('[LVP Main  ] MotionSettings.accordion_collapse()')

        ctx = _app_ctx.ctx
        stage = ctx.stage

        # Handles removing/adding the stage display depending on whether or not the accordion item is visible
        protocol_accordion_item = self.ids['motionsettings_protocol_accordion_id']
        protocol_stage_widget_parent = self.ids['protocol_settings_id'].ids['protocol_stage_holder_id']
        xystage_widget_parent = self._accordion_item_xystagecontrol.ids['xy_stagecontrol_id'].ids['xy_stage_holder_id']

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
            self.ids['motionsettings_accordion_id'].add_widget(self._accordion_item_xystagecontrol, 2)


    def _hide_xystage_control(self):
        if self._accordion_item_xystagecontrol_visible:
            self._accordion_item_xystagecontrol_visible = False
            self.ids['motionsettings_accordion_id'].remove_widget(self._accordion_item_xystagecontrol)


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
        scope_display = _app_ctx.ctx.scope_display
        #scope_display.stop()
        self.ids['verticalcontrol_id'].update_gui()
        self.ids['protocol_settings_id'].select_labware()

        # move position of motion control
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width + self.tab_width, 0
        else:
            self.pos = 0, 0

        # if scope_display.play == True:
        #     scope_display.start()


    def update_xy_stage_control_gui(self, *args, full_redraw: bool=False):
        self._accordion_item_xystagecontrol.update_gui(full_redraw=full_redraw)


    def check_settings(self, *args):
        logger.info('[LVP Main  ] MotionSettings.check_settings()')
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width + self.tab_width, 0
        else:
            self.pos = 0, 0


# ============================================================================
# XYStageControl — XY Stage Movement and Bookmarks
# ============================================================================

class XYStageControl(BoxLayout):

    def update_gui(self, dt=0, full_redraw: bool = False):
        ctx = _app_ctx.ctx
        if ctx.sequenced_capture_executor.run_in_progress():
            # During protocol: update crosshair directly from position cache
            # (zero serial I/O). Don't go through IO executor — its callback
            # runs on a worker thread which can't touch Kivy widgets.
            result = self.get_xy_targets()
            self.get_targets_ui_callback(result=result)
            return
        # Normal (non-protocol): query via IO executor as before
        ctx.io_executor.put(IOTask(
            action=self.get_xy_targets,
            callback=self.get_targets_ui_callback,
            pass_result=True
        ))

    def get_xy_targets(self):
        ctx = _app_ctx.ctx
        try:
            scope = ctx.lumaview.scope
            x_target = scope.get_target_position('X')
            x_target = np.clip(x_target, 0, scope.travel_limit_um('X'))
            y_target = scope.get_target_position('Y')
            y_target = np.clip(y_target, 0, scope.travel_limit_um('Y'))
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
            settings = ctx.settings
            coordinate_transformer = ctx.coordinate_transformer
            stage_x, stage_y = coordinate_transformer.stage_to_plate(
                labware=labware,
                stage_offset=settings['stage_offset'],
                sx=x_target,
                sy=y_target
            )

            if not self.ids['x_pos_id'].focus:
                # Cache text to prevent redundant ScrollView updates
                new_x_text = format(max(0, stage_x), '.2f')
                if self.ids['x_pos_id'].text != new_x_text:
                    self.ids['x_pos_id'].text = new_x_text # Update x position text box


            if not self.ids['y_pos_id'].focus:
                new_y_text = format(max(0, stage_y), '.2f')
                if self.ids['y_pos_id'].text != new_y_text:
                    self.ids['y_pos_id'].text = new_y_text # Update y position text box

    @debounce(0.2)
    def fine_left(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.fine_left()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] fine_left: no objective info: {e}')
            return
        fine = objective['xy_fine']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('X', -fine)))

    @debounce(0.2)
    def fine_right(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.fine_right()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] fine_right: no objective info: {e}')
            return
        fine = objective['xy_fine']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('X', fine)))

    @debounce(0.2)
    def coarse_left(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.coarse_left()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] coarse_left: no objective info: {e}')
            return
        coarse = objective['xy_coarse']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('X', -coarse)))

    @debounce(0.2)
    def coarse_right(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.coarse_right()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] coarse_right: no objective info: {e}')
            return
        coarse = objective['xy_coarse']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('X', coarse)))

    @debounce(0.2)
    def fine_back(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.fine_back()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] fine_back: no objective info: {e}')
            return
        fine = objective['xy_fine']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('Y', -fine)))

    @debounce(0.2)
    def fine_fwd(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.fine_fwd()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] fine_fwd: no objective info: {e}')
            return
        fine = objective['xy_fine']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('Y', fine)))

    @debounce(0.2)
    def coarse_back(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.coarse_back()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] coarse_back: no objective info: {e}')
            return
        coarse = objective['xy_coarse']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('Y', -coarse)))

    @debounce(0.2)
    def coarse_fwd(self):
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            return
        logger.info('[LVP Main  ] XYStageControl.coarse_fwd()')
        try:
            _, objective = get_current_objective_info()
        except Exception as e:
            logger.warning(f'[Motion] coarse_fwd: no objective info: {e}')
            return
        coarse = objective['xy_coarse']
        ctx.io_executor.put(IOTask(action=move_relative_position, args=('Y', coarse)))

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

        # x_pos is the the plate position in mm
        # Find the coordinates for the stage
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        stage_x, _ = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=x_pos,
            py=0
        )

        logger.info(f'[LVP Main  ] X pos {x_pos} Stage X {stage_x}')

        # Move to x-position
        ctx.io_executor.put(IOTask(action=move_absolute_position, args=('X', stage_x)))


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

        # y_pos is the the plate position in mm
        # Find the coordinates for the stage
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        _, stage_y = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=0,
            py=y_pos
        )

        # Move to y-position
        ctx.io_executor.put(IOTask(action=move_absolute_position, args=('Y', stage_y)))


    def set_xbookmark(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.set_xbookmark()')
        ctx.io_executor.put(IOTask(action=self.ex_set_xbookmark))

    def ex_set_xbookmark(self):
        ctx = _app_ctx.ctx

        # Get current stage x-position in um
        x_pos = ctx.lumaview.scope.get_current_position('X')

        # Save plate x-position to settings
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        plate_x, _ = coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=settings['stage_offset'],
            sx=x_pos,
            sy=0
        )

        settings['bookmark']['x'] = plate_x

    def set_ybookmark(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.set_ybookmark()')

        ctx.io_executor.put(IOTask(action=self.ex_set_ybookmark))

    def ex_set_ybookmark(self):
        ctx = _app_ctx.ctx
        y_pos = ctx.lumaview.scope.get_current_position('Y')  # Get current y pos in um

        # Save plate y-position to settings
        _, labware = get_selected_labware()
        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer
        _, plate_y = coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=settings['stage_offset'],
            sx=0,
            sy=y_pos
        )

        settings['bookmark']['y'] = plate_y

    def goto_xbookmark(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.goto_xbookmark()')

        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer

        # Get bookmark plate x-position in mm
        x_pos = settings['bookmark']['x']

        # Move to x-position
        _, labware = get_selected_labware()
        stage_x, _ = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=x_pos,
            py=0
        )
        ctx.io_executor.put(IOTask(move_absolute_position, args=('X', stage_x)))

    def goto_ybookmark(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.goto_ybookmark()')

        settings = ctx.settings
        coordinate_transformer = ctx.coordinate_transformer

        # Get bookmark plate y-position in mm
        y_pos = settings['bookmark']['y']

        # Move to y-position
        _, labware = get_selected_labware()
        _, stage_y = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=0,
            py=y_pos
        )
        ctx.io_executor.put(IOTask(move_absolute_position, args=('Y', stage_y))) # set current y position in um

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
            logger.info('[LVP Main  ] XYStageControl.home()')

            if ctx.lumaview.scope.motor_connected: # motor controller is actively connected
                ctx.io_executor.put(IOTask(move_home, kwargs={'axis':'XY'}))

                # Firmware seems to move the turret back to position 1 when performing XY homing
                # Use this command to make sure the UI is in-sync
                ctx.motion_settings.ids['verticalcontrol_id'].turret_select(selected_position=1)

            else:
                logger.warning('[LVP Main  ] Motion controller not available.')
        except Exception as e:
            logger.error(f'[UI] home failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Error", message=str(e))
