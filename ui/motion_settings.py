# Copyright Etaluma, Inc.
import logging

import numpy as np
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
from modules import gui_logger
from modules.config_ui_getters import get_selected_labware
from modules.debounce import debounce
from ui.image_settings import AccordionItemXyStageControl
from ui.ui_helpers import (
    move_absolute,
    move_home,
    move_relative,
    resort_accordion,
    run_reported,
    typed_number,
)

logger = logging.getLogger('LVP.ui.motion_settings')


# ============================================================================
# MotionSettings -- Left Sidebar Panel (Motion, Protocol, Post-Processing)
# ============================================================================


class MotionSettings(BoxLayout):
    settings_width = dp(300)
    tab_width = dp(30)

    # Canonical top-to-bottom display order for the LEFT-side accordion
    # (the right-side accordion derives its order from the release layer
    # catalogue instead; these items are panels, not layers, so the
    # order is authored here). Used by _resort_accordion() so live
    # scope-model transitions (LS850 <-> LS820 <-> LS620, etc.) keep the
    # accordion items in canonical order regardless of which were
    # hidden / re-shown along the way.
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
        ENGINEERING_MODE = _app_ctx.ctx.session.engineering_mode
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

    def _resort_accordion(self):
        """Put the left accordion back in canonical order after a model switch.

        Called from every ``_show_*`` path after its add_widget: a live
        scope-model switch (LS850 <-> LS620 <-> LS820) re-adds a hidden item
        out of order. Any item not named here, such as the engineering
        plugin's tab, goes to the bottom (Eric, 2026-05-03: auxiliary
        surfaces, not primary navigation).
        """
        accordion = self.ids.get('motionsettings_accordion_id') if hasattr(self, 'ids') else None
        if accordion is None:
            return
        widget_for_layer = {
            'microscope': self.ids.get('motionsettings_microscope_accordion_id'),
            'objective': self.ids.get('objective_control_accordion_id'),
            'xystage': self._accordion_item_xystagecontrol,
            'protocol': self.ids.get('motionsettings_protocol_accordion_id'),
            'postproc': self.ids.get('motionsettings_postprocessing_accordion_id'),
        }
        shown_for_layer = {'xystage': self._accordion_item_xystagecontrol_visible}
        resort_accordion(
            accordion,
            [
                (widget_for_layer[layer], shown_for_layer.get(layer, True))
                for layer in self._LAYER_DISPLAY_ORDER
            ],
        )

    def set_turret_control_visibility(self, visible: bool) -> None:
        vert_control = self.ids['verticalcontrol_id']
        for turret_id in ('turret_selection_label', 'turret_btn_box'):
            vert_control.ids[turret_id].visible = visible

        vert_control.ids['reset_turret_objective_btn'].disabled = not visible
        vert_control.ids['reset_turret_objective_btn'].opacity = 1 if visible else 0

    def set_focus_control_visibility(self, visible: bool) -> None:
        vert_control = self.ids['verticalcontrol_id']
        for focus_id in ('adjust_focus_label', 'focus_block', 'autofocus_id', 'zstack_id'):
            vert_control.ids[focus_id].visible = visible

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
    def toggle_settings(self) -> None:
        # A ToggleButton's state is 'normal' or 'down' and both are truthy, so
        # the comparison is what carries the user's intent. Named for the panel
        # rather than the button because ImageSettings has a same-named handler
        # logging its own panel, and the two must stay tellable apart.
        gui_logger.toggle(
            'MOTION_SETTINGS_PANEL', self.ids['toggle_motionsettings'].state == 'down'
        )
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
        # The targets are a cache and config read with no serial I/O, so they
        # are read here, on the GUI thread, whether or not a run is going.
        self.get_targets_ui_callback(result=self.get_xy_targets())

    def get_xy_targets(self):
        ctx = _app_ctx.ctx
        scope = ctx.lumaview.scope
        # A cold start without a motor has no X/Y travel, and no limits to
        # read.
        if not scope.capabilities.has_xy_stage:
            return None
        x_target = scope.motion.get_target_position('X')
        y_target = scope.motion.get_target_position('Y')
        # After disconnect() the null board is installed and there is no
        # target to show; a position listener can still tick here then.
        if x_target is None or y_target is None:
            return None
        x_target = np.clip(x_target, 0, scope.motion.get_axis_limits('X')['max'])
        y_target = np.clip(y_target, 0, scope.motion.get_axis_limits('Y')['max'])
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
            plate_x, plate_y = coordinate_transformer.stage_to_plate(
                labware=labware, stage_offset=settings['stage_offset'], sx=x_target, sy=y_target
            )

            if not self.ids['x_pos_id'].focus:
                # Cache text to prevent redundant ScrollView updates
                new_x_text = format(plate_x, '.2f')
                if self.ids['x_pos_id'].text != new_x_text:
                    self.ids['x_pos_id'].text = new_x_text  # Update x position text box

            if not self.ids['y_pos_id'].focus:
                new_y_text = format(plate_y, '.2f')
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
        if ctx.session.controls_locked:
            return
        dir_names = {('X', 1): 'RIGHT', ('X', -1): 'LEFT', ('Y', 1): 'FWD', ('Y', -1): 'BACK'}
        label = f'XY_{"COARSE" if coarse else "FINE"}_{dir_names[(axis, direction)]}'
        gui_logger.button(label)
        logger.info(f'[LVP Main  ] XYStageControl._xy_jog({label})')
        run_reported(
            lambda: move_relative(axis, direction * ctx.scope.motion.jog_step(axis, coarse)),
            redraw=None,
            label=label,
        )

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
        if ctx.session.controls_locked:
            return
        logger.info('[LVP Main  ] XYStageControl.set_xposition()')
        typed = x_pos
        x_pos = typed_number(typed, float, self.update_gui)
        if x_pos is None:
            # An entry the box refuses is still the user pressing this control.
            # Returning silently left the bundle with no line at all, so a
            # stage that did not move looked like a stage nobody asked to move.
            gui_logger.button('SET_X_POSITION', f'refused: {typed!r}')
            gui_logger.text_input('SET_X_POSITION_APPLIED', self.ids['x_pos_id'].text)
            return
        gui_logger.button('SET_X_POSITION', f'plate_mm={x_pos:.3f}')

        # The typed number goes to the API in the frame it was typed in;
        # the API owns both the conversion and the bound, so a refusal can
        # name the number the user entered instead of its stage equivalent.
        move_absolute('X', x_pos, frame='plate')

    def set_yposition(self, y_pos):
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return
        logger.info('[LVP Main  ] XYStageControl.set_yposition()')
        typed = y_pos
        y_pos = typed_number(typed, float, self.update_gui)
        if y_pos is None:
            # An entry the box refuses is still the user pressing this control.
            # Returning silently left the bundle with no line at all, so a
            # stage that did not move looked like a stage nobody asked to move.
            gui_logger.button('SET_Y_POSITION', f'refused: {typed!r}')
            gui_logger.text_input('SET_Y_POSITION_APPLIED', self.ids['y_pos_id'].text)
            return
        gui_logger.button('SET_Y_POSITION', f'plate_mm={y_pos:.3f}')

        move_absolute('Y', y_pos, frame='plate')

    def set_xbookmark(self):
        gui_logger.button('SET_X_BOOKMARK')
        logger.info('[LVP Main  ] XYStageControl.set_xbookmark()')
        run_reported(self.ex_set_xbookmark, None, 'SET_X_BOOKMARK')

    def ex_set_xbookmark(self):
        _app_ctx.ctx.session.save_bookmark(('X',))

    def set_ybookmark(self):
        gui_logger.button('SET_Y_BOOKMARK')
        logger.info('[LVP Main  ] XYStageControl.set_ybookmark()')
        run_reported(self.ex_set_ybookmark, None, 'SET_Y_BOOKMARK')

    def ex_set_ybookmark(self):
        _app_ctx.ctx.session.save_bookmark(('Y',))

    def goto_xbookmark(self):
        gui_logger.button('GOTO_X_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.goto_xbookmark()')

        settings = ctx.settings

        # Get bookmark plate x-position in mm
        x_pos = settings['bookmark']['x']

        move_absolute('X', x_pos, frame='plate')

    def goto_ybookmark(self):
        gui_logger.button('GOTO_Y_BOOKMARK')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] XYStageControl.goto_ybookmark()')

        settings = ctx.settings

        # Get bookmark plate y-position in mm
        y_pos = settings['bookmark']['y']

        move_absolute('Y', y_pos, frame='plate')

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
        gui_logger.button('HOME_XY')
        ctx = _app_ctx.ctx
        if ctx.session.controls_locked:
            return
        logger.info('[LVP Main  ] XYStageControl.home()')
        # The home's display shows every axis, the turret included: the
        # firmware's home returns the turret to position 1. With no motor
        # controller the home's own refusal is what the person is shown.
        run_reported(lambda: move_home(axis='ALL'), None, 'HOME_XY')
