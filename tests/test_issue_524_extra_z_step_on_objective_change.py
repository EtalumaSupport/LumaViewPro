# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#524 regression: protocol objective change must not waste a Z restore.

Bug
---
Every protocol step that crosses an objective went:
  step_navigation.go_to_step:118
    -> ui_helpers.move_absolute('T', protocol=True)
    -> vertical_control.turret_select(protocol=True)
    -> motion.move_turret(position=...)
    -> motion._safe_turret_move() (context manager)
        -> Z to 0
        -> T move
        -> Z restore to pre-turret-move value
  step_navigation.go_to_step:127
    -> move_absolute('Z', step['Z']) [overwrites the just-restored Z]

The Z-restore inside _safe_turret_move was wasted motion -- the next
line overwrote Z with the new step's target. ~visible extra Z step.

Fix
---
Option A (chosen): the run passes restore_z=False to its turret move
so _safe_turret_move skips the wasted restore. The default is
restore_z=True so standalone callers (the turret buttons) keep their
contract. The run now moves the turret itself rather than through the
GUI's step navigation, so the flag is passed by the run's step runner.

Test approach
-------------
AST-based structural locks on the API's signatures; the run's use of
the flag is pinned by behaviour in test_the_run_turns_the_turret_itself.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
MOTION_SRC = REPO / 'modules' / 'lumascope_api' / 'motion.py'
VERTCTRL_SRC = REPO / 'ui' / 'vertical_control.py'
UIHELPERS_SRC = REPO / 'ui' / 'ui_helpers.py'
STEPNAV_SRC = REPO / 'ui' / 'step_navigation.py'


def _module_tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function_node(tree: ast.Module, name: str, class_name: str | None = None) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if class_name is not None:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == name:
                        return child
        else:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f'{class_name or "<module>"}.{name} not found')


def _default_for(method: ast.FunctionDef, arg_name: str):
    """Return the AST default-value node for arg_name, or None if no default."""
    args = method.args
    n_pos = len(args.args)
    n_defaults = len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg == arg_name:
            if i >= n_pos - n_defaults:
                return args.defaults[i - (n_pos - n_defaults)]
            return None
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        if arg.arg == arg_name:
            return default
    return None


def test_safe_turret_move_accepts_restore_z_default_true():
    method = _function_node(_module_tree(MOTION_SRC), '_safe_turret_move', class_name='MotionAPI')
    args = method.args
    all_names = [a.arg for a in args.args] + [a.arg for a in args.kwonlyargs]
    assert 'restore_z' in all_names, '_safe_turret_move must accept restore_z parameter. (#524)'
    default = _default_for(method, 'restore_z')
    assert default is not None, 'restore_z must have a default value. (#524)'
    assert isinstance(default, ast.Constant) and default.value is True, (
        'restore_z default must be True so standalone callers (_home_turret_impl, '
        'UI turret button) preserve their pre-fix behavior. (#524)'
    )


def test_safe_turret_move_gates_z_restore_on_flag():
    method = _function_node(_module_tree(MOTION_SRC), '_safe_turret_move', class_name='MotionAPI')
    # Find an If gating on self.restore_z OR restore_z (function parameter
    # captured in the contextmanager closure).
    found_guard = False
    for node in ast.walk(method):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if 'restore_z' not in test_src:
                continue
            body_src = '\n'.join(ast.unparse(s) for s in node.body)
            if "_move_absolute_impl('Z', position=initial_z" in body_src:
                found_guard = True
                break
    assert found_guard, (
        '_safe_turret_move must gate the Z-restore call on restore_z (skip when False). (#524)'
    )


def test_tmove_threads_restore_z():
    # The public move_turret is a dispatcher; the turret motion lives in the
    # _impl body, so the signature is checked on the public form and the
    # _safe_turret_move threading on the body.
    method = _function_node(_module_tree(MOTION_SRC), 'move_turret', class_name='MotionAPI')
    all_names = [a.arg for a in method.args.args] + [a.arg for a in method.args.kwonlyargs]
    assert 'restore_z' in all_names, 'move_turret must accept restore_z. (#524)'
    impl = _function_node(_module_tree(MOTION_SRC), '_move_turret_impl', class_name='MotionAPI')
    src = ast.unparse(impl)
    assert '_safe_turret_move(restore_z=restore_z)' in src, (
        'move_turret must pass its restore_z through to _safe_turret_move. (#524)'
    )


# The run's own turret move is where restore_z=False is passed now: a run
# moves the turret itself, on every host, and the GUI no longer has a
# protocol lane to thread it through. Pinned by behaviour in
# tests/test_the_run_turns_the_turret_itself.py (every run turret move
# carries restore_z=False).


# --------------------------------------------------------------------------
# Residual #524: a generated Add-Step name dropped the objective + channel
# suffix (stored a bare 'custom0000') even though the saved image kept the
# full name. The objective-aware name must survive.
# --------------------------------------------------------------------------


def _empty_protocol_for_add():
    import datetime

    import pandas as pd

    from modules.protocol import Protocol

    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': pd.DataFrame(columns=list(Protocol.CURRENT_COLUMNS)),
        'period': datetime.timedelta(minutes=20.0),
        'duration': datetime.timedelta(hours=48.0),
        'labware_id': 'Blank',
        'capture_root': '',
        'tiling': '1x1',
        'custom_step_count': 0,
    }
    return Protocol(tiling_configs_file_loc=REPO / 'data' / 'tiling.json', config=config)


def _add_layer_config():
    return {
        'autofocus': False,
        'false_color': False,
        'illumination_ma': 100.0,
        'gain_db': 10.0,
        'auto_gain': False,
        'exposure_ms': 5.0,
        'sum': 1,
        'acquire': 'image',
        'video_config': {'duration': 5, 'fps': 30},
        'focus': 7000.0,
    }


def _insert_generated_step(protocol):
    protocol.insert_step(
        step_name=None,
        layer='BF',
        layer_config=_add_layer_config(),
        plate_position={'x': 0.0, 'y': 0.0, 'z': 5000.0},
        objective_id='4x Oly',
        stim_configs={},
        before_step=0,
        after_step=None,
    )
    return protocol.step(idx=0)['Name']


def test_added_step_name_keeps_channel_not_objective():
    # The objective is a capture-time detail stamped onto the saved filename,
    # not part of a step's identity, so an added step's Name carries its
    # channel but never the objective token.
    name = _insert_generated_step(_empty_protocol_for_add())
    assert name != 'custom0000', 'generated step name must not be the bare index'
    assert 'BF' in name, f'added step name {name!r} must keep its channel token'
    assert '4xOly' not in name, f'added step name {name!r} must not carry the objective'
