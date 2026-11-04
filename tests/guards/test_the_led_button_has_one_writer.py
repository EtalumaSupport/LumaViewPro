# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer's LED button shows its LED, so the listener bridge is its one writer.

The bridge draws the button from the LED state store on every LED event and
reconcile. Two other writers drew it from what the app intended: switching
a layer set the other layers' buttons off -- during a run too, where no off
was sent -- and going to a step during a run set the step's button down
whether or not its LED was lit. A button written from intent says an LED is
lit or dark when it is not (Eric, 2026-10-12: "Always the LED truth").
"""

from __future__ import annotations

import ast

from tests.ast_seams import production_modules

_BRIDGE = 'ui/listener_bridge.py'


def _is_the_led_button(node) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 'enable_led_btn'
    )


def led_button_writers(modules):
    """``file:line`` of every write to an LED button's state.

    The button reached directly (``...ids['enable_led_btn'].state = ...``) or
    through a name bound to it in the same module (``btn = ...ids[...]``).
    """
    found = []
    for rel, tree in modules:
        aliases = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and _is_the_led_button(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Attribute) and target.attr == 'state'):
                    continue
                owner = target.value
                if _is_the_led_button(owner) or (
                    isinstance(owner, ast.Name) and owner.id in aliases
                ):
                    found.append(f'{rel}:{node.lineno}')
    return found


def test_only_the_listener_bridge_writes_the_led_button():
    writers = led_button_writers(production_modules())
    others = [site for site in writers if not site.startswith(f'{_BRIDGE}:')]
    assert not others, (
        f'the LED button is written outside the listener bridge: {others}. '
        'It shows the LED; draw it from the LED state store through the bridge.'
    )


def test_the_scan_sees_a_writer():
    """Guards the test above against passing because it matches nothing."""
    tree = ast.parse(
        "layer_obj.ids['enable_led_btn'].state = 'down'\n"
        "btn = layer_obj.ids['enable_led_btn']\n"
        "btn.state = 'normal'\n"
    )
    assert led_button_writers([('ui/elsewhere.py', tree)]) == [
        'ui/elsewhere.py:1',
        'ui/elsewhere.py:3',
    ]
    assert [w for w in led_button_writers(production_modules()) if w.startswith(_BRIDGE)]
