# Copyright Etaluma, Inc.
"""
Small free functions extracted from lumaviewpro.py.

These helpers manipulate UI state (LED buttons, window titles, histogram,
protocol step indicators, motion controls) and are called from both
lumaviewpro.py and other modules.  They are re-exported by lumaviewpro.py
so that existing ``import lumaviewpro; lumaviewpro.func()`` call sites
continue to work.
"""

import logging

from kivy.clock import Clock
from kivy.core.window import Window

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.scope_commands as scope_commands
import modules.config_helpers as config_helpers
from modules.sequential_io_executor import IOTask

logger = logging.getLogger('LVP.modules.ui_helpers')


# ============================================================================
# Saved-folder helper
# ============================================================================

def set_last_save_folder(dir):
    if dir is None:
        return

    import lumaviewpro
    lumaviewpro.last_save_folder = dir


# ============================================================================
# Protocol nav helpers
# ============================================================================

def focus_log(positions, values):
    import lumaviewpro
    lumaviewpro.focus_round = config_helpers.focus_log(positions, values, lumaviewpro.focus_round, lumaviewpro.source_path)

def update_autofocus_selection_after_protocol():
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        layer_obj.init_autofocus()

def find_nearest_step(x, y, protocol):
    return config_helpers.find_nearest_step(x, y, protocol)


# ============================================================================
# LED / Illumination Helpers
# ============================================================================

def _handle_ui_for_leds_off():
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers_with_led():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        layer_obj.ids['enable_led_btn'].state = 'normal'


def _handle_ui_for_led(layer: str, enabled: bool, **kwargs):
    ctx = _app_ctx.ctx
    if enabled:
        state = "down"
    else:
        state = "normal"

    layer_obj = ctx.image_settings.layer_lookup(layer=layer)
    layer_obj.ids['enable_led_btn'].state = state


def scope_leds_off(no_callback: bool = False):
    import lumaviewpro
    if lumaviewpro.protocol_running_global.is_set():
        return

    callback = None if no_callback else _handle_ui_for_leds_off
    scope_commands.leds_off(lumaviewpro.lumaview.scope, lumaviewpro.io_executor, callback=callback)


# ============================================================================
# Protocol Step Navigation Helpers
# ============================================================================

def _update_step_number_callback(step_num: int):
    ctx = _app_ctx.ctx
    protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
    protocol_settings.curr_step = step_num-1
    Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)


# ============================================================================
# Motion Helpers
# ============================================================================

def _handle_ui_update_for_axis(axis: str, vertical_control: bool = False):
    ctx = _app_ctx.ctx
    axis = axis.upper()
    if axis == 'Z':
        ctx.motion_settings.ids['verticalcontrol_id'].update_gui(vertical_control=vertical_control)
    elif axis in ('X', 'Y', 'XY'):
        ctx.motion_settings.update_xy_stage_control_gui()

def _handle_autofocus_ui(pos: float):
    ctx = _app_ctx.ctx
    ctx.motion_settings.ids['verticalcontrol_id'].update_autofocus_gui(pos=pos)


# Wrapper function when moving to update UI position
def move_absolute_position(
    axis: str,
    pos: float,
    wait_until_complete: bool = False,
    overshoot_enabled: bool = True,
    protocol: bool = False,
    vertical_control: bool = False
):
    import lumaviewpro
    ctx = _app_ctx.ctx
    io_executor = lumaviewpro.io_executor

    if protocol:
        put_func = io_executor.protocol_put
    else:
        put_func = io_executor.put

    if axis == 'T':
        # Turret moves go through the GUI widget which manages homing and objective settings
        if not protocol:
            io_executor.put(IOTask(
                action=ctx.motion_settings.ids['verticalcontrol_id'].turret_select,
                kwargs={'selected_position': pos},
                callback=_handle_ui_update_for_axis,
                cb_kwargs={'axis': axis, 'vertical_control': vertical_control},
            ))
        else:
            ctx.motion_settings.ids['verticalcontrol_id'].turret_select(selected_position=pos, protocol=True)
    else:
        if not protocol:
            scope_commands.move_absolute(
                lumaviewpro.lumaview.scope, io_executor, axis, pos,
                wait_until_complete=wait_until_complete,
                overshoot_enabled=overshoot_enabled,
                callback=_handle_ui_update_for_axis,
                cb_kwargs={'axis': axis},
            )
        else:
            lumaviewpro.lumaview.scope.move_absolute_position(
                axis=axis, pos=pos,
                wait_until_complete=wait_until_complete,
                overshoot_enabled=overshoot_enabled,
            )

        Clock.schedule_once(lambda dt: _handle_ui_update_for_axis(axis=axis), 0)


def move_relative_position(
    axis: str,
    um: float,
    wait_until_complete: bool = False,
    overshoot_enabled: bool = True
):
    import lumaviewpro
    scope_commands.move_relative(
        lumaviewpro.lumaview.scope, lumaviewpro.io_executor, axis, um,
        wait_until_complete=wait_until_complete,
        overshoot_enabled=overshoot_enabled,
        callback=_handle_ui_update_for_axis,
        cb_kwargs={'axis': axis},
    )


def move_home(axis: str):
    import lumaviewpro
    axis = axis.upper()
    Clock.schedule_once(lambda dt: Window.set_title(f"Lumaview Pro {lumaviewpro.version}   |   Homing, please wait..."), 0)
    scope_commands.move_home(lumaviewpro.lumaview.scope, lumaviewpro.io_executor, axis, callback=move_home_cb, cb_args=(axis))


# ============================================================================
# Window Title Helpers
# ============================================================================

# Should only be called from main thread
def set_recording_title(progress=None):
    import lumaviewpro
    if progress is None:
        Window.set_title(f"Lumaview Pro {lumaviewpro.version}   |   Recording Video...")
    else:
        Window.set_title(f"Lumaview Pro {lumaviewpro.version}   |   Recording Video... {int(progress)}%")

# Should only be called from main thread
def set_writing_title(progress=None):
    import lumaviewpro
    if progress is None:
        Window.set_title(f"Lumaview Pro {lumaviewpro.version}   |   Writing Video...")
    else:
        Window.set_title(f"Lumaview Pro {lumaviewpro.version}   |   Writing Video... {int(progress)}%")

def reset_title():
    import lumaviewpro
    Window.set_title(f"Lumaview Pro {lumaviewpro.version}")


def move_home_cb(axis):
    import lumaviewpro
    _handle_ui_update_for_axis(axis=axis)
    Window.set_title(f"Lumaview Pro {lumaviewpro.version}")


# ============================================================================
# Histogram / Contrast Helpers
# ============================================================================

def live_histo_off():
    import lumaviewpro
    ctx = _app_ctx.ctx
    if lumaviewpro.live_histo_setting and ctx.scope_display.use_live_image_histogram_equalization:
        ctx.scope_display.use_live_image_histogram_equalization = False
        logger.info('[LVP Main  ] Live Histogram Equalization] False')

def live_histo_reverse():
    import lumaviewpro
    ctx = _app_ctx.ctx
    if lumaviewpro.live_histo_setting and not ctx.scope_display.use_live_image_histogram_equalization:
        ctx.scope_display.use_live_image_histogram_equalization = True
        logger.info('[LVP Main  ] Live Histogram Equalization] True')


# ============================================================================
# UI State Helpers
# ============================================================================

def reset_acquire_ui():
    import lumaviewpro
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        if lumaviewpro.settings[layer]['acquire'] == "image":
            layer_obj.ids['acquire_image'].active = True
        elif lumaviewpro.settings[layer]['acquire'] == "video":
            layer_obj.ids['acquire_video'].active = True
        else:
            layer_obj.ids['acquire_none'].active = True

def reset_stim_ui():
    import lumaviewpro
    ctx = _app_ctx.ctx
    for layer in common_utils.get_layers():
        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
        if "stim_config" in lumaviewpro.settings[layer]:
            if lumaviewpro.settings[layer]['stim_config'] is not None:
                lumaviewpro.settings[layer]['stim_config']['enabled'] = False
                layer_obj.ids['stim_disable_btn'].active = True
                layer_obj.update_stim_controls_visibility()
