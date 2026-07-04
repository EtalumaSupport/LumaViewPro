"""#712 regression: changing a step's channel updates its (auto) name.

A step's stored Name kept the old channel token after a channel change, so
both the displayed Step Name and the saved filename were stale -- and the
filename appended the new channel beside the old one (A2_BF_Blue_...).

Fix (current shape): a step's Name is a DERIVED display column re-rendered
from the structured columns (Label, Well, Color, Tile, Z-Slice) on every
mutation, so a channel change updates exactly the channel token -- for well
and custom steps alike (#719), and without ever truncating a user label
that happens to embed a token-shaped segment. The UI passes only the
resolved rename (or None for "keep") as modify_step's label kwarg; it no
longer branches on Auto_Named or regenerates names itself.

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
        parse_legacy_step_name,
    )

    old = build_step_name(StepNameComponents(well='A2', channel='BF', objective='2xOly'))
    assert old == 'A2_BF_2xOly'
    # Regenerate for a new channel by components: replace the channel field,
    # rebuild. The bug appended (A2_BF_Blue_...); rebuilding from components
    # replaces, carrying ONLY the new channel. (The parse leg lives at the
    # legacy load boundary now; the runtime pipeline carries columns.)
    new = build_step_name(
        dataclasses.replace(
            parse_legacy_step_name(old, known_layers=['BF', 'Blue']), channel='Blue'
        )
    )
    assert new == 'A2_Blue_2xOly'
    assert 'BF' not in new


def _func(name: str) -> ast.FunctionDef:
    tree = ast.parse(PS_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {PS_SRC}')


def _modify_layer_config(acquire='image'):
    return {
        'autofocus': False,
        'false_color': False,
        'illumination_ma': 100.0,
        'gain_db': 10.0,
        'auto_gain': False,
        'exposure_ms': 5.0,
        'sum': 1,
        'acquire': acquire,
        'video_config': {'duration': 5, 'fps': 30},
    }


def _modify_channel(proto, *, layer, label=None):
    proto.modify_step(
        step_idx=0,
        label=label,
        layer=layer,
        layer_config=_modify_layer_config(),
        plate_position={'x': 0.0, 'y': 0.0, 'z': 5000.0},
        objective_id='4x Oly',
        stim_configs={},
    )
    return proto.step(idx=0)


def test_channel_change_replaces_token_on_well_step():
    # The exact #712 repro at the protocol layer: a channel change on an
    # auto-named well step regenerates the channel token (replace, never
    # append A2_BF_Blue).
    from tests.test_protocol_roundtrip import _build_protocol, _make_step

    proto = _build_protocol(
        [_make_step(name='A2_BF', well='A2', z_slice=-1, tile_group_id=-1, zstack_group_id=-1)]
    )
    step = _modify_channel(proto, layer='Blue')
    assert step['Name'] == 'A2_Blue', step['Name']
    assert 'BF' not in step['Name']


def test_channel_change_replaces_token_on_custom_step_and_keeps_user_label():
    # #719: a custom-added step is covered too -- regeneration is not gated
    # on a well anchor -- and a user label rides along untouched while its
    # channel token updates.
    from tests.test_protocol_roundtrip import _build_protocol, _make_step

    proto = _build_protocol(
        [
            _make_step(
                name='custom0000_BF',
                label='custom0000',
                well='',
                z_slice=-1,
                tile_group_id=-1,
                zstack_group_id=-1,
            )
        ]
    )
    step = _modify_channel(proto, layer='Blue')
    assert step['Name'] == 'custom0000_Blue', step['Name']

    proto2 = _build_protocol(
        [
            _make_step(
                name='MySpot_BF',
                label='MySpot',
                auto_named=False,
                well='',
                z_slice=-1,
                tile_group_id=-1,
                zstack_group_id=-1,
            )
        ]
    )
    step2 = _modify_channel(proto2, layer='Blue')
    assert step2['Name'] == 'MySpot_Blue', step2['Name']
    assert step2['Label'] == 'MySpot'
    assert not step2['Auto_Named']


def test_modify_step_ex_passes_resolved_rename_as_label_without_well_gate():
    # The UI's only naming job is to pass the resolved rename (None = keep)
    # as modify_step's label kwarg; the derived-Name re-render inside
    # Protocol.modify_step covers well and custom steps alike (#719), so the
    # UI must not re-introduce Well or Auto_Named branching around it.
    node = _func('modify_step_ex')
    src = ast.unparse(node)
    assert 'resolve_step_rename' in src, (
        'modify_step_ex must resolve the name field through resolve_step_rename'
    )
    label_kwarg_found = False
    for call in ast.walk(node):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'modify_step'
        ):
            kwargs = {k.arg for k in call.keywords}
            assert 'label' in kwargs, 'modify_step must receive the rename via label='
            assert 'step_name' not in kwargs and 'auto_named' not in kwargs
            label_kwarg_found = True
    assert label_kwarg_found, 'self._protocol.modify_step(...) call not found'
    assert "curr_step['Well'] != ''" not in src, (
        'name handling must not exclude custom-added steps by gating on a well step (#719)'
    )
