# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Protocol step navigation logic extracted from lumaviewpro.py.

These functions show a protocol step on the panel and the layer widgets
once the Session has gone to it. They are GUI-coupled (Kivy widgets,
Clock) and live in ui/; protocol execution reaches them only through the
injected go_to_step callback, so the protocol layer never imports this
module directly.
"""

import logging

from modules.kivy_utils import schedule_ui as _schedule_ui

import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.step_navigation')


def go_to_step(
    protocol,
    step_idx: int,
    include_move: bool = True,
):
    """Show a protocol step, and for a person's navigation go to it.

    ``include_move=False`` is the run's display of the step it is executing:
    the run moves the scope itself, so this only moves the step pointer.
    ``include_move=True`` is a person going to a step: the Session goes to
    it (``ScopeSession.start_go_to_step``: the moves started, the step's
    settings into its layer, the LED preview, one task on the IO lane); the
    pointer, the step panel, the camera and the layer's widgets follow only
    once that task has run without a refusal. A click is a gesture: it does
    not wait for the stage to arrive.
    """
    # Deferred import: ui_helpers imports the display modules, and
    # step_navigation still reaches upward here, which the display-only
    # direction has yet to undo.
    from ui.ui_helpers import submit_move

    ctx = _app_ctx.ctx

    protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
    if protocol.num_steps() <= 0:
        # No step to show: a Delete emptied the list, or a file loaded none.
        # Any other index, a typed 0 included, is the Session's to refuse.
        protocol_settings.curr_step = -1
        _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    if not include_move:
        protocol_settings.curr_step = step_idx
        _schedule_ui(lambda dt: protocol_settings.generate_step_name_input(), 0)
        _schedule_ui(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    step_list_revision = protocol.step_list_revision
    session = ctx.session

    def superseded() -> bool:
        # An edit to the step list (Delete, Add, a new or loaded protocol)
        # can land while this click waits on the lane or while the stage
        # moves, and it places the pointer itself. Against the changed list
        # step_idx names another step or none: going there would move the
        # scope to a step the person did not pick, and writing the pointer
        # back would put it on a step the panel does not show, or past the
        # end of the list.
        return (
            protocol_settings._protocol is not protocol
            or protocol.step_list_revision != step_list_revision
        )

    def call():
        if superseded():
            return
        session.start_go_to_step(protocol, step_idx)

    def on_moved():
        if superseded():
            return
        step = protocol.step(idx=step_idx)
        protocol_settings.curr_step = step_idx
        protocol_settings.generate_step_name_input()
        protocol_settings.update_step_ui()
        # The layer's settings now hold the step; the camera and the
        # histogram take them. update_led=False: the Session's LED preview
        # was the one LED command, and the apply must not re-derive LED
        # intent from the enable button, which REFLECTS driver state through
        # the listener bridge; read as a command it re-lights a channel the
        # user toggled off. The apply keeps its autofocus-owns-the-camera
        # suppression: manual navigation does not coordinate with a live AF.
        ctx.image_settings.layer_lookup(layer=step['Color']).apply_settings(update_led=False)
        go_to_step_update_ui(step)

    # The axes redrawn once the task has ended: none on a scope with no
    # motor board, which goes to the step without moving.
    submit_move(
        'GO_TO_STEP',
        axes=ctx.scope.capabilities.axes if ctx.scope.motor_connected else (),
        call=call,
        on_moved=on_moved,
    )


def go_to_step_update_ui(step):
    """Update UI widgets to reflect a protocol step.

    Delegates per-layer widget updates to LayerControl.set_step_state(),
    which encapsulates widget knowledge. This function handles only the
    cross-layer concerns: opening the settings panel, expanding the
    accordion, and setting the LED button during protocol preview.
    """
    ctx = _app_ctx.ctx

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # Open the ImageSettings panel so the step's settings are visible.
    # Act only when it is not already open: this runs on every step
    # navigation, and the panel toggle is an expand/collapse handler,
    # not an idempotent refresh -- re-invoking it on an already-open panel
    # repeats the reposition + histogram rescheduling every step and logs a
    # toggle line when nothing actually toggled.
    imagesettings_toggle = ctx.image_settings.ids['toggle_imagesettings']
    if imagesettings_toggle.state != 'down':
        imagesettings_toggle.state = 'down'
        ctx.image_settings.toggle_settings()

    # Expand the accordion to the step's channel. Direct
    # `collapse = False` on a single item doesn't propagate to siblings
    # in Kivy's Accordion -- only user clicks auto-collapse others -- so
    # manual nav from Green -> Red would leave Green visually expanded
    # without this call.
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
    # "During protocol" is a run in progress, not the run lockout: the
    # lockout deliberately holds through the post-run writing-files
    # window, when stepping is manual and no LED event will ever correct a
    # forced 'down' left here.
    if ctx.session.run_in_progress:
        layer_obj.ids['enable_led_btn'].state = 'down'
