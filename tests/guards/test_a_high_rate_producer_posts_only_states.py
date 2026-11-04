# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What posts to the UI thread at a rate it may not keep up with posts only states.

The display thread posted each frame's validity dot as a fresh Clock
callback, so a UI thread held up for four days held every one: 8.7 GB. An
event (``schedule_ui``) is one entry per post; a state
(``schedule_ui_latest``) keeps one entry per key however long the UI thread
waits. So every producer that posts at the rate its source changes -- the
display's frames, a run's saved frames, post-processing's derived images,
the listener bridge's positions, LEDs, shown channel and settings, the
progress proxies, the support zip's progress, the window's run state --
posts states only. This walk follows each producer through the functions of
its own module it calls (``self.<method>`` and module-level names) and
refuses any event post or raw ``Clock`` post on the way. A lambda or nested
``def`` is what the producer hands to the UI thread, so the walk does not
enter it: it runs there, and may post events of its own.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, parse_module, walk_defs

# (file, qualname) of each producer.
_PRODUCERS = (
    ('ui/scope_display.py', 'ScopeDisplay._render_one_frame'),
    ('ui/scope_display.py', 'ScopeDisplay.hold_protocol_saved_image'),
    ('ui/ui_helpers.py', 'show_captured_frame'),
    ('ui/post_processing.py', 'QuickEnhanceControls._queue_derived_image'),
    ('ui/listener_bridge.py', 'UIListenerBridge._on_position_change'),
    ('ui/listener_bridge.py', 'UIListenerBridge._on_turret_change'),
    ('ui/listener_bridge.py', 'UIListenerBridge._on_led_state_changed'),
    ('ui/listener_bridge.py', 'UIListenerBridge._on_shown_channel_changed'),
    ('ui/listener_bridge.py', 'UIListenerBridge._on_settings_changed'),
    ('ui/progress_popup.py', '_PopupProxy.__setattr__'),
    ('ui/progress_popup.py', '_HostWidgetProxy.__setattr__'),
    ('ui/microscope_settings.py', 'MicroscopeSettings._make_zip._progress'),
    ('lumaviewpro.py', 'LumaViewProApp._on_run_state'),
)
# An event post, under its own name and the GUI helpers' alias, and Kivy's raw posts.
_EVENT_POSTS = Words(
    {'schedule_ui', '_schedule_ui', 'schedule_once', 'schedule_interval', 'create_trigger'}
)


def _defs(rel_path: str) -> dict[str, ast.AST]:
    return dict(walk_defs(parse_module(rel_path).body))


def _calls(node: ast.AST):
    """Every call in ``node``'s body; a lambda or nested def is posted, not followed."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _calls(child)


def _followed(call: ast.Call, qualname: str, defs: dict[str, ast.AST]) -> str | None:
    """The def in the same module ``call`` runs, if the walk can name it."""
    func = call.func
    if isinstance(func, ast.Name) and func.id in defs:
        return func.id
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == 'self'
        and '.' in qualname
    ):
        method = f'{qualname.split(".", 1)[0]}.{func.attr}'
        if method in defs:
            return method
    return None


def test_a_high_rate_producer_posts_only_states():
    found = []
    for rel_path, root in _PRODUCERS:
        defs = _defs(rel_path)
        assert root in defs, f'{rel_path} {root} is gone; re-pin the producer'
        todo, seen = [root], set()
        while todo:
            qualname = todo.pop()
            if qualname in seen:
                continue
            seen.add(qualname)
            for call in _calls(defs[qualname]):
                func = call.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
                if name in _EVENT_POSTS:
                    found.append(
                        f'{rel_path}:{call.lineno} {qualname} (from {root}) posts by {name}'
                    )
                followed = _followed(call, qualname, defs)
                if followed is not None:
                    todo.append(followed)
    assert found == [], (
        'a producer that posts at its source rate posts an event; post the state it shows '
        'with schedule_ui_latest((owner, name), ...):\n' + '\n'.join(found)
    )
