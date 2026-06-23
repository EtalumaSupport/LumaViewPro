"""#712 regression: changing a step's channel updates its (auto) name.

A step's stored Name kept the old channel token after a channel change, so
both the displayed Step Name and the saved filename were stale -- and the
filename appended the new channel beside the old one (A2_BF_Blue_...).

Fix: a step carries an explicit Auto_Named flag. While it is set (true for
a freshly built well or custom step, cleared when the user types a name),
a channel change regenerates the name so its channel token tracks the
change. The earlier fix gated this on a well step, which excluded
custom-added steps (#719); the flag covers both and never misreads a
user-typed name that happens to match the auto pattern.

The regeneration mechanic is proven behaviorally (it must replace the
channel, not append). The UI wiring in ProtocolSettings.modify_step_ex
needs a full Kivy app to exec, so it is pinned structurally -- the same
approach as the #612 / #524 / #710 step-UI guards.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
PS_SRC = REPO / 'ui' / 'protocol_settings.py'


def test_regenerated_name_replaces_channel_not_appends():
    import dataclasses

    from modules.common_utils import (
        StepNameComponents,
        build_step_name,
        parse_step_name,
    )

    old = build_step_name(StepNameComponents(well='A2', channel='BF', objective='2xOly'))
    assert old == 'A2_BF_2xOly'
    # Regenerate for a new channel the way the UI does: parse the stored name,
    # replace the channel component, rebuild. The bug appended (A2_BF_Blue_...);
    # rebuilding from components replaces, carrying ONLY the new channel.
    new = build_step_name(
        dataclasses.replace(parse_step_name(old, known_layers=['BF', 'Blue']), channel='Blue')
    )
    assert new == 'A2_Blue_2xOly'
    assert 'BF' not in new


def _func(name: str) -> ast.FunctionDef:
    tree = ast.parse(PS_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {PS_SRC}')


def test_get_default_name_accepts_color_override():
    names = [a.arg for a in _func('get_default_name_for_curr_step').args.args]
    assert 'color' in names, (
        'get_default_name_for_curr_step must accept a color override so a '
        'step can be renamed for a different channel'
    )


def test_modify_step_regenerates_auto_name_by_flag_not_well():
    src = ast.unparse(_func('modify_step_ex'))
    assert 'get_default_name_for_curr_step(color=active_layer)' in src, (
        'modify_step_ex must regenerate the auto name for the new channel'
    )
    assert "curr_step['Auto_Named']" in src, (
        'regeneration must be gated on the explicit Auto_Named flag'
    )
    assert "curr_step['Well'] != ''" not in src, (
        'regeneration must not exclude custom-added steps by gating on a '
        'well step (#719); the Auto_Named flag covers both'
    )
