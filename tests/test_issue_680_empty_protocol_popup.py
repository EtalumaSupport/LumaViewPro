# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#680 regression: New Protocol must warn when no channels are enabled.

Bug
---
Clicking New with every layer's Acquire checkbox disabled silently
produced an empty Protocol (0 steps). The user got no warning; the
UI just displayed 0 steps. Root cause: Protocol.from_config skips
any layer whose acquire is not in ('image', 'video'); if every layer
is skipped, the resulting Protocol is legally constructed but empty,
and new_protocol() had no post-construction step-count guard.

Fix
---
After ctx.scope.create_protocol() in ProtocolSettings.new_protocol,
check protocol.num_steps() == 0 and pop a "No Channels Selected"
notification before queueing new_protocol_ex on the worker pool.

Test approach
-------------
1. Source-level structural lock via AST: extract new_protocol's body
   and assert the num_steps()==0 guard exists, uses
   show_notification_popup with the "No Channels Selected" title,
   and runs before the worker_pool.put that queues new_protocol_ex.
   Direct UI exec is impractical here (Kivy ids, _app_ctx, worker
   pool, IOTask wrapping); the AST lock catches a regression that
   removes or reorders the guard.

2. Behavioral check on Protocol.from_config: build a config whose
   every layer has acquire set to None/disabled and verify the
   resulting Protocol has zero steps -- proves the precondition the
   UI guard relies on.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    source = PROTOCOL_SETTINGS_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {PROTOCOL_SETTINGS_SRC}')


def _stmt_index_for(body: list, predicate) -> int:
    for i, stmt in enumerate(body):
        if predicate(stmt):
            return i
    return -1


def test_new_protocol_guards_empty_step_count():
    """new_protocol must check num_steps()==0 and pop a notification."""
    method = _method_node('ProtocolSettings', 'new_protocol')
    src = ast.unparse(method)

    # The exact comparison that gates the popup.
    assert 'protocol.num_steps() == 0' in src, (
        'ProtocolSettings.new_protocol must guard on '
        'protocol.num_steps() == 0 after create_protocol succeeds. '
        '(#680)'
    )

    # The notification title is user-visible and what the bench-retest
    # signal looks for.
    assert 'No Channels Selected' in src, (
        'ProtocolSettings.new_protocol must use the title '
        '"No Channels Selected" for the empty-protocol popup so the '
        'bench retest signal matches. (#680)'
    )

    # The popup helper must be invoked (not just imported).
    assert 'show_notification_popup' in src, (
        'ProtocolSettings.new_protocol must call show_notification_popup '
        'to surface the empty-protocol case. (#680)'
    )


def test_empty_step_guard_runs_before_worker_pool_put():
    """Guard must fire and return before queueing new_protocol_ex."""
    method = _method_node('ProtocolSettings', 'new_protocol')

    def has_num_steps_zero_compare(node):
        if not isinstance(node, ast.If):
            return False
        test = ast.unparse(node.test)
        return 'num_steps()' in test and '0' in test

    def has_worker_pool_put(node):
        unparsed = ast.unparse(node)
        return 'worker_pool.put' in unparsed and 'new_protocol_ex' in unparsed

    guard_idx = _stmt_index_for(method.body, has_num_steps_zero_compare)
    put_idx = _stmt_index_for(method.body, has_worker_pool_put)

    assert guard_idx >= 0, 'num_steps()==0 guard not found in ProtocolSettings.new_protocol. (#680)'
    assert put_idx >= 0, 'worker_pool.put for new_protocol_ex not found in new_protocol. (#680)'
    assert guard_idx < put_idx, (
        f'Empty-steps guard at statement {guard_idx} must run BEFORE '
        f'worker_pool.put at statement {put_idx}; otherwise an empty '
        f'protocol still queues new_protocol_ex and assigns '
        f'self._protocol. (#680)'
    )


def _from_config_input(acquire_by_layer: dict[str, str]) -> dict:
    """Minimal valid input_config for Protocol.from_config with one layer
    per entry, each with the given acquire mode."""
    layer_configs = {
        layer: {
            'acquire': acquire,
            'autofocus': False,
            'false_color': False,
            'illumination_ma': 50.0,
            'gain_db': 1.0,
            'auto_gain': False,
            'exposure_ms': 10.0,
            'sum': 1,
            'focus': 100.0,
            'video_config': {'duration': 30},
            'stim_config': None,
        }
        for layer, acquire in acquire_by_layer.items()
    }
    return {
        'labware_id': '96 well microplate',
        'objective_id': '4x Oly',
        'zstack_params': {'range': 0, 'step_size': 0, 'z_reference': 'center'},
        'use_zstacking': False,
        'tiling': '1x1',
        'layer_configs': layer_configs,
        'period': None,
        'duration': None,
        'frame_dimensions': {'width': 2048, 'height': 2048},
        'binning_size': 1,
        'stim_config': {},
    }


def test_protocol_from_config_filters_non_acquire_layers():
    """The upstream filter that produces 0 steps: layers whose acquire is
    neither 'image' nor 'video' contribute no steps, so a config where
    every layer is disabled yields an empty (0-step) Protocol -- the
    precondition the #680 UI guard catches. A layer set to 'image' still
    produces steps."""
    from modules.protocol import Protocol

    tiling_configs = REPO / 'data' / 'tiling.json'

    all_disabled = Protocol.from_config(
        input_config=_from_config_input({'BF': 'none', 'Blue': 'none'}),
        tiling_configs_file_loc=tiling_configs,
    )
    assert all_disabled.num_steps() == 0, (
        'every-layer-disabled must construct an EMPTY protocol (the #680 '
        f'guard precondition); got {all_disabled.num_steps()} steps'
    )

    one_enabled = Protocol.from_config(
        input_config=_from_config_input({'BF': 'image', 'Blue': 'none'}),
        tiling_configs_file_loc=tiling_configs,
    )
    assert one_enabled.num_steps() > 0, 'an image layer must still produce steps'
    step_colors = set(one_enabled.steps()['Color'].unique())
    assert step_colors == {'BF'}, (
        f'only the acquiring layer may contribute steps; got {step_colors}'
    )


def test_new_protocol_gates_no_channel_popup_behind_channel_check():
    """No-channel popup must be gated behind an actual channel check.

    Zero steps has two causes: no channel enabled for acquisition, or a
    labware with no wells (e.g. Blank, a 0x0 plate, which yields an empty
    well list so the channel filter never runs). The #680 fix attributed
    every empty protocol to "no channels", so selecting Blank with a
    channel enabled raised a false "No Channels Selected" error (#687).

    The fix consults the channel predicate inside the num_steps()==0
    block: only the genuine no-channels case pops the warning; a no-well
    labware with a channel enabled creates an empty protocol silently
    (the user fills it in with Add). This test fails on the pre-fix
    source, where the popup sits directly in the zero-step block.
    """
    method = _method_node('ProtocolSettings', 'new_protocol')

    # new_protocol has two num_steps() ifs: the >0 z-carryover block and
    # the ==0 empty-protocol guard. Target the guard specifically.
    outer = None
    for stmt in method.body:
        if isinstance(stmt, ast.If) and 'num_steps() == 0' in ast.unparse(stmt.test):
            outer = stmt
            break
    assert outer is not None, 'num_steps()==0 guard missing from new_protocol (#687)'

    outer_src = ast.unparse(outer)
    assert 'get_layer_configs' in outer_src, (
        'new_protocol must consult get_layer_configs() inside the '
        'num_steps()==0 block to tell "no channels" apart from "no '
        'wells" before showing the popup. (#687)'
    )

    # The popup must NOT sit directly in the zero-step block; it must be
    # nested inside the no-channels branch.
    directly_in_outer = any(
        'No Channels Selected' in ast.unparse(s) for s in outer.body if not isinstance(s, ast.If)
    )
    assert not directly_in_outer, (
        'The "No Channels Selected" popup must be gated behind a channel '
        'check, not fired for every empty protocol (a no-well labware '
        'with a channel enabled must not raise a channel error). (#687)'
    )

    nested_if_has_popup = any(
        isinstance(s, ast.If) and 'No Channels Selected' in ast.unparse(s) for s in outer.body
    )
    assert nested_if_has_popup, (
        'Expected the "No Channels Selected" popup nested inside the '
        'no-channels branch within the num_steps()==0 block. (#687)'
    )
