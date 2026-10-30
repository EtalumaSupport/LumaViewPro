# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Protocol step navigation logic extracted from lumaviewpro.py.

These functions show a protocol step on the panel and the layer widgets
once the Session has gone to it. They are GUI-coupled (Kivy widgets,
Clock) and live in ui/; protocol execution reaches them only through the
panel's step_started handler, so the protocol layer never imports this
module directly.
"""

import logging

from modules.kivy_utils import schedule_ui as _schedule_ui

import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.step_navigation')


def go_to_step(
    protocol,
    step_index: int,
    include_move: bool = True,
):
    """Show a protocol step, and for a person's navigation go to it.

    ``include_move=False`` is the run's display of the step it is executing:
    the run moves the scope itself, so this only moves the step pointer.
    ``include_move=True`` is a person going to a step: the Session goes to
    it (``ScopeSession.start_go_to_step``: the moves started, the step's
    settings into its layer, the LED preview and the step's layer on the
    camera); the pointer, the step panel and the layer's widgets follow only
    once it has returned without a refusal. A click is a gesture: it does
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
        _schedule_ui(lambda dt: protocol_settings.update_step_ui())
        return

    if not include_move:
        protocol_settings.curr_step = step_index
        _schedule_ui(lambda dt: protocol_settings.generate_step_name_input())
        _schedule_ui(lambda dt: protocol_settings.update_step_ui())
        return

    step_list_revision = protocol.step_list_revision
    session = ctx.session

    def superseded() -> bool:
        # An edit to the step list (Delete, Add, a new or loaded protocol)
        # can land while this click waits on the lane or while the stage
        # moves, and it places the pointer itself. Against the changed list
        # step_index names another step or none: going there would move the
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
        session.start_go_to_step(protocol, step_index)

    def on_moved():
        if superseded():
            return
        step = protocol.step_row(step_index=step_index)
        protocol_settings.curr_step = step_index
        protocol_settings.generate_step_name_input()
        protocol_settings.update_step_ui()
        go_to_step_update_ui(step)

    # The axes redrawn once the task has ended, whatever its outcome: every
    # axis the scope has, which a refused or partial go-to may have moved.
    # On the worker pool: the go-to waits on the IO lane and then the camera
    # lane, and a lane's worker never waits on another lane.
    submit_move(
        'GO_TO_STEP',
        axes=ctx.scope.capabilities.axes,
        call=call,
        lane=None,
        on_moved=on_moved,
    )


def go_to_step_update_ui(step):
    """Update UI widgets to reflect a protocol step.

    Delegates per-layer widget updates to LayerControl.set_step_state(),
    which encapsulates widget knowledge. This function handles only the
    cross-layer concerns: opening the settings panel. The drawer is the
    listener bridge's, which draws the shown channel, and so is the LED
    button, which draws the LED.
    """
    ctx = _app_ctx.ctx

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # Open the ImageSettings panel so the step's settings are visible.
    # Act only when it is not already open: this runs on every step
    # navigation, and placing the panel repeats the reposition + histogram
    # rescheduling. The app opens it here, so the person's toggle handler,
    # and its record, are not called.
    imagesettings_toggle = ctx.image_settings.ids['toggle_imagesettings']
    if imagesettings_toggle.state != 'down':
        imagesettings_toggle.state = 'down'
        ctx.image_settings.place_settings()

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

    # The LED button is not written here: it shows the LED, and the listener
    # bridge, the button's one writer, draws it from the LED state store --
    # during a run too, so the step's button is down while, and only while,
    # its LED is lit.
