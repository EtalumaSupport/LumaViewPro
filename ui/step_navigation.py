# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Protocol step navigation logic extracted from lumaviewpro.py.

These functions handle navigating to protocol steps (moving stage,
updating LED/camera settings, and refreshing UI controls). They are
GUI-coupled (Kivy widgets, Clock) and live in ui/; protocol execution
reaches them only through the injected go_to_step callback, so the
protocol layer never imports this module directly.
"""

import copy
import logging

import modules.common_utils as common_utils
from modules.exceptions import ProtocolRunRefusedError
from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx

import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.step_navigation')


def go_to_step(
    protocol,
    step_idx: int,
    ignore_auto_gain: bool = False,
    include_move: bool = True,
):
    """Show a protocol step, and for a person's navigation go to it.

    ``include_move=False`` is the run's display of the step it is executing:
    the run moves the scope itself, so this only moves the step pointer.
    ``include_move=True`` is a person going to a step: one gesture on the IO
    lane asks whether the scope knows where its axes are, moves them and
    lights the step's preview; the pointer, the step panel and the layer
    settings follow only once that task has run without a refusal.
    """
    # Deferred import: ui_helpers imports the display modules, and
    # step_navigation still reaches upward here, which the display-only
    # direction has yet to undo.
    from ui.ui_helpers import submit_gesture

    ctx = _app_ctx.ctx
    settings = ctx.settings

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
    step_list_revision = protocol.step_list_revision

    # Above the pointer write, and that position is the whole point. The
    # pointer is what `modify_step_ex` and `insert_step_ex` address a step
    # BY, and they fill that step from the LIVE stage -- so a pointer left
    # on a step the scope never reached means the next edit silently writes
    # the previous step's coordinates into it. Refusing after the write
    # would leave exactly that.
    #
    # The rule is the API's, asked here for the one step this call is about
    # to navigate to. The refusal does not propagate: it has already been
    # logged and shown to the user, this call has changed nothing yet, and
    # go_to_step is also the run's navigation callback -- raising would end
    # a run over a question the engine already answered at prepare().
    try:
        ctx.scope.protocols.refuse_unaddressable_objectives([step['Objective']])
    except ProtocolRunRefusedError:
        return

    if not include_move:
        protocol_settings.curr_step = step_idx
        _schedule_ui(lambda dt: protocol_settings.generate_step_name_input(), 0)
        _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    # A same-step re-selection (re-clicking / re-typing the current number)
    # must leave a user-lit channel alone; only a REAL step change drives
    # the LED preview transition below.
    step_changed = protocol_settings.curr_step != step_idx

    # A step stores plate mm; the API converts and bounds it.
    plate_x = step['X']
    plate_y = step['Y']

    turret_pos = None
    if ctx.scope.capabilities.has_turret:
        step_objective_id = step['Objective']
        # The same lookup the run makes, so navigating to a step and
        # running it choose the same slot.
        turret_pos = ctx.scope.motion.get_turret_position_for_objective_id(
            objective_id=step_objective_id
        )

        if turret_pos is None:
            # Unreachable: the rule above admitted this objective by
            # reading the same turret configuration this lookup reads,
            # so a slot carrying it exists. If the two ever disagree,
            # stop -- what this replaces logged, raised a dialog, and
            # then moved X, Y and Z anyway, capturing through whatever
            # glass was in the path and naming the file for the glass
            # the step asked for.
            raise RuntimeError(
                f'No turret slot carries {step_objective_id!r} for step {step_idx}, '
                'yet the admissibility rule accepted it from the same turret '
                'configuration. The rule and the slot lookup have disagreed.'
            )

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # Trace what go_to_step does with camera settings. The camera values
    # are the cached ones: a debug line never reads the camera on the
    # GUI thread.
    _curr_gain = ctx.scope.imaging.gain_db_cached if ctx.scope.imaging.active_cached else '?'
    _curr_exp = ctx.scope.imaging.exposure_ms_cached if ctx.scope.imaging.active_cached else '?'
    logger.debug(
        f'[GO_TO_STEP DIAG] step_idx={step_idx} color={color} '
        f'step_gain={step["Gain"]} step_exp={step["Exposure"]} '
        f'step_auto_gain={step["Auto_Gain"]!r} '
        f'camera_gain={_curr_gain} camera_exp={_curr_exp} '
        f'protocol_running={ctx.session.is_protocol_running}'
    )

    led_ctx = (
        _step_led_ctx(ctx=ctx, settings=settings, step=step, color=color) if step_changed else None
    )

    motion = ctx.scope.motion
    illumination = ctx.scope.illumination
    # A scope with no motor board (a manual model) goes to the step without
    # moving: the pointer, the settings and the preview still follow.
    has_motor = ctx.scope.motor_connected

    def moves():
        if has_motor:
            if turret_pos is not None:
                # The step's own Z move follows, so the turret need not put
                # Z back first.
                motion.move_turret(turret_pos, restore_z=False)
            motion.move_absolute('X', plate_x, frame='plate')
            motion.move_absolute('Y', plate_y, frame='plate')
            motion.move_absolute('Z', step['Z'])
        if led_ctx is not None:
            # After the step's moves, in the same task: a toggle the person
            # makes while the stage travels lands after the step's preview.
            illumination.apply_transition(LedTransition.MANUAL_STEP, led_ctx)

    def on_moved():
        # An edit to the step list (Delete, Add, a new or loaded protocol)
        # can land while the stage moves, and it places the pointer itself.
        # Against the changed list step_idx names another step or none, so
        # writing it back would put the pointer on a step the panel does not
        # show, or past the end of the list.
        if (
            protocol_settings._protocol is not protocol
            or protocol.step_list_revision != step_list_revision
        ):
            return
        protocol_settings.curr_step = step_idx
        protocol_settings.generate_step_name_input()
        protocol_settings.update_step_ui()
        _load_step_into_layer(
            ctx=ctx,
            settings=settings,
            layer_obj=layer_obj,
            step=step,
            color=color,
            ignore_auto_gain=ignore_auto_gain,
        )
        go_to_step_update_ui(step)

    # Every axis the navigation drives is asked about, the turret included:
    # a failed turret home leaves T unknown while the stage axes still know theirs.
    submit_gesture(
        'GO_TO_STEP',
        axes=ctx.scope.capabilities.axes if has_motor else (),
        then='go to the step',
        moves=moves,
        on_moved=on_moved,
    )


def _step_led_ctx(*, ctx, settings, step, color) -> LedTransitionCtx:
    """The step's LED preview: its channel when the preview is on, all dark when off.

    The authority's MANUAL_STEP transition diffs this against cached state,
    so it clears a previously-lit different-colour channel without blinking
    a same-colour one. Outside a run nothing holds the LED lease -- live-UI
    control is unleased -- so the transition goes through the lease-free
    apply_transition.
    """
    channel = ctx.scope.illumination.color2ch(color)
    if channel is None and ctx.scope.led_connected and color in common_utils.get_layers_with_led():
        # A preview click on a layer this unit's identity lacks would
        # otherwise just not light, with nothing anywhere naming why.
        logger.warning(
            f"[Step Nav  ] This scope has no '{color}' LED channel; preview will not light."
        )
    return LedTransitionCtx(
        channel=channel,
        illumination_ma=step['Illumination'],
        preview_on=settings['protocol_led_on'],
    )


def _load_step_into_layer(*, ctx, settings, layer_obj, step, color, ignore_auto_gain):
    """Load the step into the layer's live settings, then apply them.

    Manual navigation loads the step into the layer's live settings: the
    panel, the settings and the camera then agree on the step the user
    clicked, and the apply reads the settings. A protocol run never writes
    here -- the run reads its steps from the protocol, and the user's
    live-view configuration is theirs to keep across it. The step's config
    dicts are copied: the protocol owns them.

    Camera + histogram: applied directly in BOTH preview states
    (protocol=False runs the camera block and the histogram-layer sync).
    update_led=False: the step's LED transition is the one LED command --
    apply_settings must not re-derive LED intent from the enable button,
    which REFLECTS driver state via the listener bridge; reading it as a
    command channel re-lights a channel the user toggled off. protocol=False
    also keeps the autofocus-owns-the-camera suppression: manual nav does
    not coordinate with a live AF the way the protocol runner does, so
    pushing camera settings mid-AF would corrupt the sweep.
    """
    layer_values = {
        'autofocus': step['Auto_Focus'],
        'false_color': step['False_Color'],
        'illumination_ma': step['Illumination'],
        'gain_db': step['Gain'],
        'auto_gain': step['Auto_Gain'],
        'exposure_ms': step['Exposure'],
        'sum': step['Sum'],
        'acquire': step['Acquire'],
        # Manual navigation follows the step's Z, so the layer's
        # Goto Focus lands where the step was set up.
        'focus': step['Z'],
    }
    video_config = step.get('Video Config')
    if isinstance(video_config, dict):
        layer_values['video_config'] = copy.deepcopy(video_config)
    stim_configs = step.get('Stim_Config')
    with ctx.settings_lock:
        settings[color].update(layer_values)
        # A step's stim config spans every layer it names, so each
        # named layer's settings take theirs.
        if isinstance(stim_configs, dict):
            for stim_layer, stim_config in stim_configs.items():
                settings[stim_layer]['stim_config'] = copy.deepcopy(stim_config)
    layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=False, update_led=False)


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
