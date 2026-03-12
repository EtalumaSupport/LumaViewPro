# Copyright Etaluma, Inc.
"""
Protocol step navigation logic extracted from lumaviewpro.py.

These functions handle navigating to protocol steps (moving stage,
updating LED/camera settings, and refreshing UI controls).
They are re-exported by lumaviewpro.py so existing call sites work.
"""

import copy
import logging

from kivy.clock import Clock

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.scope_commands as scope_commands
from modules.sequential_io_executor import IOTask

logger = logging.getLogger('LVP.modules.step_navigation')


def go_to_step(
    protocol,
    step_idx: int,
    ignore_auto_gain: bool = False,
    include_move: bool = True,
    called_from_protocol: bool = True
):
    from modules.config_ui_getters import get_selected_labware
    from modules.ui_helpers import move_absolute_position
    try:
        from ui.notification_popup import show_notification_popup
    except ImportError:
        show_notification_popup = None

    ctx = _app_ctx.ctx
    settings = ctx.settings
    coordinate_transformer = ctx.coordinate_transformer
    io_executor = ctx.io_executor

    num_steps = protocol.num_steps()
    protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
    if num_steps <= 0:
        protocol_settings.curr_step = -1
        Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    if (step_idx < 0) or (step_idx >= num_steps):
        protocol_settings.curr_step = -1
        Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    step = protocol.step(idx=step_idx)
    protocol_settings.curr_step = step_idx

    Clock.schedule_once(lambda dt: protocol_settings.generate_step_name_input(), 0)
    Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)


    # Convert plate coordinates to stage coordinates
    if include_move:
        _, labware = get_selected_labware()
        sx, sy = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=step["X"],
            py=step["Y"]
        )

        turret_pos = None
        if ctx.scope.has_turret():
            step_objective_id = step["Objective"]
            turret_pos = ctx.scope.get_turret_position_for_objective_id(
                objective_id=step_objective_id,
            )

            if turret_pos is None:

                logger.error(f"Cannot move turret for step {step_idx}. No position found with objective {step_objective_id}")

                error_msg = f"Cannot move turret to step {step_idx}. No objective position found matching step's objective: {step_objective_id}. Please check objective settings."
                if show_notification_popup is not None:
                    Clock.schedule_once(lambda dt: show_notification_popup(title="Protocol Objective Not Set", message=error_msg), 0)

        # Move into position
        if ctx.scope.motion.driver:
            if not called_from_protocol:
                if turret_pos is not None:
                    io_executor.put(IOTask(action=move_absolute_position,kwargs={"axis":'T',"pos": turret_pos,"protocol": False}))
                    Clock.schedule_once(lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(turret_pos), 0)
                io_executor.put(IOTask(action=move_absolute_position,kwargs={"axis":'X',"pos": sx,"protocol": False}))
                io_executor.put(IOTask(
                        action=move_absolute_position,
                        kwargs={
                            "axis":'Y',
                            "pos": sy,
                            "protocol": False
                        }
                    ))
                io_executor.put(IOTask(
                        action=move_absolute_position,
                        kwargs={
                            "axis":'Z',
                            "pos": step['Z'],
                            "protocol": False
                        }
                    ))
            else:
                if turret_pos is not None:
                    move_absolute_position(axis='T', pos=turret_pos, protocol=True)
                    Clock.schedule_once(lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(turret_pos), 0)
                move_absolute_position('X', sx, protocol=True)
                move_absolute_position('Y', sy, protocol=True)
                move_absolute_position('Z', step["Z"], protocol=True, wait_until_complete=True)
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

        # Update settings to correspond with step — batch write under lock for thread safety
        color = step['Color']
        with ctx.settings_lock:
            settings[color].update({
                'autofocus': step['Auto_Focus'],
                'false_color': step['False_Color'],
                'ill': step["Illumination"],
                'gain': step["Gain"],
                'auto_gain': step["Auto_Gain"],
                'exp': step["Exposure"],
                'sum': step["Sum"],
                'acquire': step['Acquire'],
                'focus': step['Z'],  # Keep per-layer focus in sync with step (#535)
            })

        layer_obj = ctx.image_settings.layer_lookup(layer=color)

        def temp():
            layer_obj.ids['enable_led_btn'].state = 'down'
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)

        if not called_from_protocol and settings['protocol_led_on']:
            scope_commands.led_on(ctx.scope, io_executor, color, step['Illumination'])
            Clock.schedule_once(lambda dt: temp(), 0)
        else:
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)



        Clock.schedule_once(lambda dt: go_to_step_update_ui(step), 0)



def go_to_step_update_ui(step):
    ctx = _app_ctx.ctx
    settings = ctx.settings
    protocol_running_global = ctx.protocol_running

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # open ImageSettings
    ctx.image_settings.ids['toggle_imagesettings'].state = 'down'
    ctx.image_settings.toggle_settings()

    # set accordion item to corresponding channel (skip during protocol to prevent memory leaks)
    if not protocol_running_global.is_set():
        accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=color)
        accordion_item_obj.collapse = False



    # set autofocus checkbox
    logger.info(f'[LVP Main  ] autofocus: {step["Auto_Focus"]}')

    layer_obj.ids['autofocus'].active = step['Auto_Focus']

    # set false_color checkbox
    logger.info(f'[LVP Main  ] false_color: {step["False_Color"]}')

    layer_obj.ids['false_color'].active = step['False_Color']

    # set illumination settings, text, and slider
    logger.info(f'[LVP Main  ] ill: {step["Illumination"]}')

    layer_obj.ids['ill_text'].text = str(step["Illumination"])
    layer_obj.ids['ill_slider'].value = float(step["Illumination"])

    # set gain settings, text, and slider
    logger.info(f'[LVP Main  ] gain: {step["Gain"]}')

    layer_obj.ids['gain_text'].text = str(step["Gain"])
    layer_obj.ids['gain_slider'].value = float(step["Gain"])

    # set auto_gain checkbox
    logger.info(f'[LVP Main  ] auto_gain: {step["Auto_Gain"]}')

    layer_obj.ids['auto_gain'].active = step["Auto_Gain"]

    # set exposure settings, text, and slider
    logger.info(f'[LVP Main  ] exp: {step["Exposure"]}')

    layer_obj.ids['exp_text'].text = str(step["Exposure"])
    layer_obj.ids['exp_slider'].value = float(step["Exposure"])

    # set sum count settings, text, and slider
    logger.info(f'[LVP Main  ] sum: {step["Sum"]}')

    layer_obj.ids['sum_text'].text = str(step["Sum"])
    layer_obj.ids['sum_slider'].value = int(step["Sum"])

    # set video config (e.g., duration) controls
    if 'Video Config' in step and isinstance(step['Video Config'], dict):
        vc = step['Video Config']
        settings[color]['video_config'] = copy.deepcopy(vc)
        if 'duration' in vc:
            layer_obj.ids['video_duration_text'].text = str(vc['duration'])
            layer_obj.ids['video_duration_slider'].value = float(vc['duration'])

    # Set stim configuration for each channel
    if 'Stim_Config' in step:
        if isinstance(step['Stim_Config'], dict):
            # Update each channel's stim config
            for layer in step['Stim_Config']:
                stim_config = step['Stim_Config'][layer]
                settings[layer]['stim_config'] = copy.deepcopy(stim_config)

                stim_layer_obj = ctx.image_settings.layer_lookup(layer=layer)

                if stim_config['enabled']:
                    stim_layer_obj.ids['stim_enable_btn'].active = True
                    stim_layer_obj.ids['stim_disable_btn'].active = False
                else:
                    stim_layer_obj.ids['stim_disable_btn'].active = True
                    stim_layer_obj.ids['stim_enable_btn'].active = False

                stim_layer_obj.update_stim_controls_visibility()

                stim_layer_obj.ids['stim_freq_text'].text = str(stim_config['frequency'])
                stim_layer_obj.ids['stim_freq_slider'].value = float(stim_config['frequency'])
                stim_layer_obj.ids['stim_pulse_width_text'].text = str(stim_config['pulse_width'])
                stim_layer_obj.ids['stim_pulse_width_slider'].value = float(stim_config['pulse_width'])
                stim_layer_obj.ids['stim_pulse_count_text'].text = str(stim_config['pulse_count'])
                stim_layer_obj.ids['stim_pulse_count_slider'].value = int(stim_config['pulse_count'])

    # acquire type

    for acquire_sel in ('acquire_video', 'acquire_image', 'acquire_none'):
        layer_obj.ids[acquire_sel].active = False

    if step['Acquire'] == 'video':
        layer_obj.ids['acquire_video'].active = True
    elif step['Acquire'] == 'image':
        layer_obj.ids['acquire_image'].active = True
    else:
        layer_obj.ids['acquire_none'].active = True

    # Set LED button state to match: 'down' if LED is on, 'normal' if off.
    # The actual LED hardware command is sent in go_to_step() when
    # protocol_led_on is True and not called_from_protocol.
    if settings['protocol_led_on'] and not protocol_running_global.is_set():
        layer_obj.ids['enable_led_btn'].state = 'down'
