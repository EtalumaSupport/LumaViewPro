# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A raw Kivy ``Clock`` post is the UI thread posting to itself; every other post is the dispatcher's.

Twenty-one cross-thread posts called ``Clock`` directly. They worked only
because the dispatcher was also ``Clock.schedule_once``: they joined no
order with the dispatcher's posts, held no bound, and a Qt GUI would have
had to find every one. Every post from another thread now goes through
``kivy_utils``' queue (``schedule_ui`` for an event, ``schedule_ui_latest``
for a state). What remains on ``Clock`` is the UI thread's own deferrals,
triggers and intervals, which stay Kivy's.

This walk counts every ``Clock.schedule_once`` / ``schedule_interval`` /
``create_trigger`` call in production by its enclosing function and holds
the count to the table, each row with the reason it runs on the UI thread.
A new call, or a count that grows, fails until it is classified here or
moved to the dispatcher. Not caught: a function classified here later called
from another thread.
"""

from __future__ import annotations

import ast
from collections import Counter

from tests.ast_seams import Words, production_modules, walk_defs

_RAW_POSTS = Words({'schedule_once', 'schedule_interval', 'create_trigger'})

_CONSTRUCTION = 'a widget building itself, deferred to its first frame'
_DEBOUNCE = 'a trigger or interval a widget creates on the UI thread'

# (file, enclosing function) -> (raw Clock calls, why it is the UI thread's own)
_UI_THREAD = {
    ('lumaviewpro.py', 'LumaViewProApp._ask_about_rejected_settings'): (
        1,
        'called from after_the_startup_motion and on_start, both on the Clock',
    ),
    ('lumaviewpro.py', 'LumaViewProApp._close_the_session'): (
        1,
        "the window's close, on the UI thread",
    ),
    ('lumaviewpro.py', 'LumaViewProApp._on_resize'): (2, "the window's resize event"),
    ('lumaviewpro.py', 'LumaViewProApp._prompt_objective_if_needed'): (
        1,
        'called from after_the_startup_motion and on_start, both on the Clock',
    ),
    ('lumaviewpro.py', 'LumaViewProApp.on_request_close'): (2, "Kivy's close request"),
    ('lumaviewpro.py', 'LumaViewProApp.on_start'): (3, "Kivy's start, on the UI thread"),
    ('lumaviewpro.py', 'LumaViewProApp.on_start.after_the_startup_motion'): (
        1,
        'run by startup_motion_settled, posted through schedule_ui, or inline from on_start',
    ),
    ('ui/advanced_settings.py', 'AdvancedSettings.set_acceleration_limit'): (1, _DEBOUNCE),
    ('ui/image_settings.py', 'ImageSettings.__init__'): (1, _CONSTRUCTION),
    ('ui/image_settings.py', 'ImageSettings._init_ui'): (1, _CONSTRUCTION),
    ('ui/image_settings.py', 'set_histogram_layer'): (1, _DEBOUNCE),
    ('ui/layer_control.py', 'LayerControl.__init__'): (2, _CONSTRUCTION),
    ('ui/layer_control.py', 'LayerControl._bind_depth_hint'): (1, _CONSTRUCTION),
    ('ui/layer_control.py', 'LayerControl._init_ui'): (1, _CONSTRUCTION),
    ('ui/main_display.py', 'MainDisplay.draw_record_button'): (
        1,
        "the recording's poll, started from the record button's own draws",
    ),
    ('ui/motion_settings.py', 'MotionSettings.__init__'): (1, _CONSTRUCTION),
    ('ui/motion_settings.py', 'MotionSettings._init_ui'): (1, _CONSTRUCTION),
    ('ui/post_processing.py', 'ZProjectionControls.__init__'): (1, _CONSTRUCTION),
    ('ui/post_processing.py', '_run_build._show'): (
        1,
        "the build's done-callback, posted by submit_reported",
    ),
    ('ui/protocol_settings.py', 'ProtocolSettings.__init__'): (3, _CONSTRUCTION),
    ('ui/protocol_settings.py', 'ProtocolSettings._init_ui'): (1, _CONSTRUCTION),
    ('ui/protocol_settings.py', 'ProtocolSettings._run_scan_pre_callback'): (
        1,
        'a widget handler, or scan_started delivered with on_ui_thread=True',
    ),
    ('ui/shader.py', 'ShaderViewer.__init__'): (2, _CONSTRUCTION),
    ('ui/sim_walk.py', 'SimWalk._await_bring_up'): (1, "the walk's own steps, on the Clock"),
    ('ui/sim_walk.py', 'SimWalk._resume'): (2, "the walk's own steps, on the Clock"),
    ('ui/sim_walk.py', 'SimWalk.start'): (1, "the walk's own steps, on the Clock"),
    ('ui/stage.py', 'Stage.__init__'): (1, _CONSTRUCTION),
    ('ui/stage.py', 'Stage.draw_labware_io_calculations'): (
        12,
        "its callers are the bridge's posted XY draw and widget handlers",
    ),
    ('ui/tooltip.py', 'TooltipMixin.mouse_moved'): (1, "the window's mouse event"),
    ('ui/vertical_control.py', 'VerticalControl.__init__'): (1, _CONSTRUCTION),
    ('ui/vertical_control.py', 'VerticalControl.show_turret_state'): (
        1,
        "the bridge's posted turret draw and widget handlers",
    ),
    ('ui/vertical_control.py', 'VerticalControl.update_gui'): (
        2,
        'every caller is on the UI thread: a posted redraw, a widget handler, the Clock',
    ),
}


def _raw_clock_calls() -> Counter:
    """``{(file, enclosing function): count}`` of every raw ``Clock`` post in production."""
    counts: Counter = Counter()
    for rel_path, tree in production_modules():
        enclosing = {}
        for qualname, node in walk_defs(tree.body):
            for inner in ast.walk(node):
                enclosing[id(inner)] = qualname  # an inner def's walk comes later and wins
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RAW_POSTS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'Clock'
            ):
                counts[(rel_path, enclosing.get(id(node), '<module>'))] += 1
    return counts


def test_every_raw_clock_post_is_classified_as_the_ui_threads_own():
    found = _raw_clock_calls()
    held = {site: count for site, (count, _reason) in _UI_THREAD.items()}
    unclassified = [
        f'{path} {qualname}: {count} raw Clock call(s), classified {held.get((path, qualname), 0)}'
        for (path, qualname), count in sorted(found.items())
        if count != held.get((path, qualname))
    ]
    stale = [
        f'{path} {qualname}' for path, qualname in sorted(held) if (path, qualname) not in found
    ]
    assert unclassified == [] and stale == [], (
        'a raw Clock post is not the classified count; a post from another thread goes '
        'through kivy_utils.schedule_ui (an event) or schedule_ui_latest (a state), and a '
        "UI thread's own deferral is classified here with its reason:\n"
        + '\n'.join(unclassified + [f'{site}: classified, no longer found' for site in stale])
    )
