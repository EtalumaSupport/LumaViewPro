# Copyright Etaluma, Inc.
"""
Protocol step navigation logic extracted from lumaviewpro.py.

These functions handle navigating to protocol steps (moving stage,
updating LED/camera settings, and refreshing UI controls).
They are re-exported by lumaviewpro.py so existing call sites work.
"""

import logging

from modules.kivy_utils import schedule_ui as _schedule_ui

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
    from modules.notification_center import notifications

    ctx = _app_ctx.ctx
    settings = ctx.settings
    coordinate_transformer = ctx.coordinate_transformer
    io_executor = ctx.io_executor

    num_steps = protocol.num_steps()
    protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
    if num_steps <= 0:
        protocol_settings.curr_step = -1
        _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    if (step_idx < 0) or (step_idx >= num_steps):
        protocol_settings.curr_step = -1
        _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    step = protocol.step(idx=step_idx)
    protocol_settings.curr_step = step_idx

    _schedule_ui(lambda dt: protocol_settings.generate_step_name_input(), 0)
    _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)


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
                notifications.error("Protocol", "Protocol Objective Not Set", error_msg)

        # Move into position
        if ctx.scope.motor_connected:
            if not called_from_protocol:
                if turret_pos is not None:
                    io_executor.put(IOTask(action=move_absolute_position,kwargs={"axis":'T',"pos": turret_pos,"protocol": False}))
                    _schedule_ui(lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(turret_pos), 0)
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
                    _schedule_ui(lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(turret_pos), 0)
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

        # #610 diagnostic: trace what go_to_step does with camera settings
        _curr_gain = ctx.scope.get_gain() if ctx.scope.camera_active else '?'
        _curr_exp = ctx.scope.get_exposure_time() if ctx.scope.camera_active else '?'
        logger.info(
            f"[GO_TO_STEP DIAG] step_idx={step_idx} color={color} "
            f"step_gain={step['Gain']} step_exp={step['Exposure']} "
            f"step_auto_gain={step['Auto_Gain']!r} "
            f"camera_gain={_curr_gain} camera_exp={_curr_exp} "
            f"called_from_protocol={called_from_protocol} "
            f"protocol_running={ctx.protocol_running.is_set()}"
        )

        def temp():
            layer_obj.ids['enable_led_btn'].state = 'down'
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)

        if not called_from_protocol and settings['protocol_led_on']:
            # Turn off previous channel before switching — one LED at a time,
            # same pattern as composite capture and protocol capture (#605).
            scope_commands.leds_off(ctx.scope, io_executor)
            scope_commands.led_on(ctx.scope, io_executor, color, step['Illumination'])
            _schedule_ui(lambda dt: temp(), 0)
        else:
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)


        # Force stage crosshair + position text update after step navigation.
        # The move_position callback in _default_move normally handles this,
        # but when go_to_step is used (all UI-triggered protocols), _default_move
        # is bypassed. Schedule on main thread since go_to_step may be called
        # from the protocol executor thread.
        _schedule_ui(lambda dt: ctx.motion_settings.update_xy_stage_control_gui(), 0)
        # Also force a stage widget redraw so the crosshair/well indicator moves
        _schedule_ui(lambda dt: ctx.stage.draw_labware(), 0)

        _schedule_ui(lambda dt: go_to_step_update_ui(step), 0)


def go_to_step_update_ui(step):
    """Update UI widgets to reflect a protocol step.

    Delegates per-layer widget updates to LayerControl.set_step_state(),
    which encapsulates widget knowledge. This function handles only the
    cross-layer concerns: opening the settings panel, expanding the
    accordion, and setting the LED button during protocol preview.
    """
    ctx = _app_ctx.ctx
    settings = ctx.settings
    protocol_running_global = ctx.protocol_running

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # Open ImageSettings panel
    ctx.image_settings.ids['toggle_imagesettings'].state = 'down'
    ctx.image_settings.toggle_settings()

    # Expand accordion to step's channel (skip during protocol to prevent memory leaks)
    if not protocol_running_global.is_set():
        accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=color)
        accordion_item_obj.collapse = False

    # Delegate all per-layer widget updates to LayerControl
    layer_obj.set_step_state(step)

    # Stim config spans multiple layers — update non-current layers too
    sc = step.get('Stim_Config')
    if isinstance(sc, dict):
        for layer in sc:
            if layer != color:
                other_obj = ctx.image_settings.layer_lookup(layer=layer)
                # Build a minimal step dict for the other layer's stim only
                other_obj.set_step_state({'Stim_Config': {layer: sc[layer]}})

    # Set LED button state to show which channel is active for this step.
    # During protocol: show the step's channel as 'down' so user sees which
    # LED is being used, even though the actual on/off happens in the executor.
    # Outside protocol: only if protocol_led_on is enabled (preview mode).
    if protocol_running_global.is_set() or settings.get('protocol_led_on', False):
        from ui.layer_control import LayerControl
        LayerControl._suppressing_led_log = True
        try:
            layer_obj.ids['enable_led_btn'].state = 'down'
        finally:
            LayerControl._suppressing_led_log = False
