# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Protocol step navigation logic extracted from lumaviewpro.py.

These functions handle navigating to protocol steps (moving stage,
updating LED/camera settings, and refreshing UI controls). They are
GUI-coupled (Kivy widgets, Clock) and live in ui/; protocol execution
reaches them only through the injected go_to_step callback, so the
protocol layer never imports this module directly.
"""

import logging

from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx

import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.step_navigation')


def go_to_step(
    protocol,
    step_idx: int,
    ignore_auto_gain: bool = False,
    include_move: bool = True,
    called_from_protocol: bool = True,
):
    from modules.config_ui_getters import get_selected_labware

    # Deferred import: ui/ui_helpers.move_absolute wraps the
    # API call with UI update callbacks. step_navigation still reaches
    # upward here -- tracked as part of LAYER-H/LV-13 follow-up.
    from ui.ui_helpers import move_absolute
    from modules.notification_center import notifications

    ctx = _app_ctx.ctx
    settings = ctx.settings
    coordinate_transformer = ctx.coordinate_transformer

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
    # A same-step re-selection (re-clicking / re-typing the current number)
    # must leave a user-lit channel alone; only a REAL step change drives
    # the LED preview transition below.
    step_changed = protocol_settings.curr_step != step_idx
    protocol_settings.curr_step = step_idx

    _schedule_ui(lambda dt: protocol_settings.generate_step_name_input(), 0)
    _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)

    # Convert plate coordinates to stage coordinates
    if include_move:
        _, labware = get_selected_labware()
        sx, sy = coordinate_transformer.plate_to_stage(
            labware=labware, stage_offset=settings['stage_offset'], px=step['X'], py=step['Y']
        )

        turret_pos = None
        if ctx.scope.capabilities.has_turret:
            step_objective_id = step['Objective']
            turret_pos = ctx.scope.motion.get_turret_position_for_objective_id(
                objective_id=step_objective_id,
                persisted_position=settings.get('turret_position'),
            )

            if turret_pos is None:
                logger.error(
                    f'Cannot move turret for step {step_idx}. No position found with objective {step_objective_id}'
                )

                error_msg = f"Cannot move turret to step {step_idx}. No objective position found matching step's objective: {step_objective_id}. Please check objective settings."
                notifications.error('Protocol', 'Protocol Objective Not Set', error_msg)

        # Move into position
        if ctx.scope.motor_connected:
            if not called_from_protocol:
                if turret_pos is not None:
                    move_absolute(axis='T', position=turret_pos, protocol=False)
                    _schedule_ui(
                        lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(
                            turret_pos
                        ),
                        0,
                    )
                move_absolute(axis='X', position=sx, protocol=False)
                move_absolute(axis='Y', position=sy, protocol=False)
                move_absolute(axis='Z', position=step['Z'], protocol=False)
            else:
                if turret_pos is not None:
                    # restore_z=False -- the Z move below overwrites Z with
                    # step['Z'] immediately, so _safe_turret_move's default
                    # Z-restore-after-T-move would be wasted motion (#524).
                    move_absolute(axis='T', position=turret_pos, protocol=True, restore_z=False)
                    _schedule_ui(
                        lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(
                            turret_pos
                        ),
                        0,
                    )
                move_absolute('X', sx, protocol=True)
                move_absolute('Y', sy, protocol=True)
                move_absolute('Z', step['Z'], protocol=True, wait_until_complete=True)
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

        # Update settings to correspond with step -- batch write under lock for thread safety
        color = step['Color']
        with ctx.settings_lock:
            settings[color].update(
                {
                    'autofocus': step['Auto_Focus'],
                    'false_color': step['False_Color'],
                    'illumination_ma': step['Illumination'],
                    'gain_db': step['Gain'],
                    'auto_gain': step['Auto_Gain'],
                    'exposure_ms': step['Exposure'],
                    'sum': step['Sum'],
                    'acquire': step['Acquire'],
                    'focus': step['Z'],  # Keep per-layer focus in sync with step (#535)
                }
            )

        layer_obj = ctx.image_settings.layer_lookup(layer=color)

        # #610 diagnostic: trace what go_to_step does with camera settings
        _curr_gain = ctx.scope.imaging.get_gain_db() if ctx.scope.imaging.active_cached else '?'
        _curr_exp = ctx.scope.imaging.get_exposure_ms() if ctx.scope.imaging.active_cached else '?'
        logger.debug(
            f'[GO_TO_STEP DIAG] step_idx={step_idx} color={color} '
            f'step_gain={step["Gain"]} step_exp={step["Exposure"]} '
            f'step_auto_gain={step["Auto_Gain"]!r} '
            f'camera_gain={_curr_gain} camera_exp={_curr_exp} '
            f'called_from_protocol={called_from_protocol} '
            f'protocol_running={ctx.session.is_protocol_running}'
        )

        if not called_from_protocol:
            _apply_manual_nav_outcome(
                ctx=ctx,
                settings=settings,
                layer_obj=layer_obj,
                step=step,
                color=color,
                ignore_auto_gain=ignore_auto_gain,
                step_changed=step_changed,
            )
        else:
            # Protocol-cycle invocation runs on the executor thread and must
            # not submit MANUAL_STEP transitions or widget-writing applies --
            # the run's own LED authority calls own the hardware.
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)

        # Force stage crosshair + position text update after step navigation.
        # The move_position callback in _default_move normally handles this,
        # but when go_to_step is used (all UI-triggered protocols), _default_move
        # is bypassed. Schedule on main thread since go_to_step may be called
        # from the protocol executor thread.
        _schedule_ui(lambda dt: ctx.motion_settings.update_xy_stage_control_gui(), 0)
        # Also force a stage widget redraw so the crosshair/well indicator moves
        _schedule_ui(lambda dt: ctx.stage.draw_labware(), 0)

        # Capture called_from_protocol in the closure so the UI-thread
        # callback knows whether this is a protocol-cycle invocation
        # (skip accordion-open) or a manual-navigation one (do it).
        # Reading the run-lockout state inside the UI callback races
        # at protocol-end: the last step's scheduled callback can fire
        # AFTER cleanup releases the lockout, see it clear, and open
        # the last-step's accordion (Red on a 4-channel protocol).
        # Closure-capture is race-free.
        _schedule_ui(
            lambda dt: go_to_step_update_ui(step, called_from_protocol=called_from_protocol),
            0,
        )


def _apply_manual_nav_outcome(
    *, ctx, settings, layer_obj, step, color, ignore_auto_gain, step_changed
):
    """Manual navigation owns its entire outcome: one LED authority
    transition plus one direct settings apply -- the accordion reconcile is
    not load-bearing for navigation in either preview state.

    LED: the authority's MANUAL_STEP target is the step's channel when the
    preview is on and all-dark when it is off, and its diff against cached
    state clears a previously-lit different-colour channel without blinking
    a same-colour one. Fires only on a REAL step change: a same-step
    re-selection leaves a user-lit (or user-darkened) channel exactly as it
    is. Outside a run nothing holds the LED lease -- live-UI control is
    unleased -- so this routes through the lease-free apply_transition.

    Camera + histogram: applied directly in BOTH preview states
    (protocol=False runs the camera block and the histogram-layer sync;
    with the preview off there is no early-return leaving them to the
    reconcile). update_led=False: the transition above is the one LED
    command -- apply_settings must not re-derive LED intent from the
    enable button, which REFLECTS driver state via the listener bridge;
    reading it as a command channel re-lights a channel the user toggled
    off. protocol=False also keeps the autofocus-owns-the-camera
    suppression: manual nav does not coordinate with a live AF the way the
    protocol runner does, so pushing camera settings mid-AF would corrupt
    the sweep.
    """
    if step_changed:
        led_ctx = LedTransitionCtx(
            channel=ctx.scope.illumination.color2ch(color),
            illumination_ma=step['Illumination'],
            preview_on=settings['protocol_led_on'],
        )
        ctx.scope.illumination.apply_transition_async(LedTransition.MANUAL_STEP, led_ctx)
    _schedule_ui(
        lambda dt: layer_obj.apply_settings(
            ignore_auto_gain=ignore_auto_gain, protocol=False, update_led=False
        ),
        0,
    )


def go_to_step_update_ui(step, called_from_protocol: bool = False):
    """Update UI widgets to reflect a protocol step.

    Delegates per-layer widget updates to LayerControl.set_step_state(),
    which encapsulates widget knowledge. This function handles only the
    cross-layer concerns: opening the settings panel, expanding the
    accordion, and setting the LED button during protocol preview.

    ``called_from_protocol``: when True, skip the accordion expand
    (the user's chosen open accordion is preserved during + after a
    protocol run). When False (manual step navigation), expand to
    the step's channel as the user expects.
    """
    ctx = _app_ctx.ctx

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # Open the ImageSettings panel so the step's settings are visible.
    # Act only when it is not already open: this runs once per step during
    # a protocol run, and the panel toggle is an expand/collapse handler,
    # not an idempotent refresh -- re-invoking it on an already-open panel
    # repeats the reposition + histogram rescheduling every step and logs a
    # toggle line when nothing actually toggled.
    imagesettings_toggle = ctx.image_settings.ids['toggle_imagesettings']
    if imagesettings_toggle.state != 'down':
        imagesettings_toggle.state = 'down'
        ctx.image_settings.toggle_settings()

    # Expand accordion to step's channel ONLY for manual navigation.
    # Direct `collapse = False` on a single item doesn't propagate to
    # siblings in Kivy's Accordion -- only user clicks auto-collapse
    # others -- so manual nav from Green -> Red would leave Green
    # visually expanded without this call. Protocol-cycle invocations
    # skip the call entirely (the in-protocol guard inside
    # set_expanded_layer has a race at protocol-end: the last step's
    # UI-scheduled callback runs after the run lockout releases,
    # leaving the accordion stuck on the last step's color).
    if not called_from_protocol:
        ctx.image_settings.set_expanded_layer(layer=color)

    # Delegate all per-layer widget updates to LayerControl
    layer_obj.set_step_state(step)

    # Stim config spans multiple layers -- update non-current layers too
    sc = step.get('Stim_Config')
    if isinstance(sc, dict):
        for layer in sc:
            if layer != color:
                other_obj = ctx.image_settings.layer_lookup(layer=layer)
                # Build a minimal step dict for the other layer's stim only
                other_obj.set_step_state({'Stim_Config': {layer: sc[layer]}})

    # Set LED button state to show which channel is active for this step.
    # During protocol only: show the step's channel as 'down' so the user sees
    # which LED is being used, even though the actual on/off happens in the
    # executor. Outside a run the listener bridge is the sole button writer,
    # reflecting driver truth -- a forced 'down' here would go stale and any
    # later apply_settings(update_led=True) would re-light the channel.
    # "During protocol" is the RUNNER's truth, not the run lockout: the
    # lockout deliberately holds through the post-run writing-files
    # window, when stepping is manual and no LED event will ever correct a
    # forced 'down' left here.
    if ctx.sequenced_capture_runner.run_in_progress():
        from ui.layer_control import LayerControl

        LayerControl._suppressing_led_log = True
        try:
            layer_obj.ids['enable_led_btn'].state = 'down'
        finally:
            LayerControl._suppressing_led_log = False
