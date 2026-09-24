# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every read of the live position outside the motion API says what it is for.

An axis that has lost its reference keeps answering the last number it
reported -- real-looking, and no longer true. Displaying that number is
harmless; SAVING it (into a step, a layer's focus, a bookmark, a captured
file) is how a false position reaches a user's data, and it did, at every
such site, because nothing made a new reader ask. The readers themselves
cannot refuse -- the turret-safety retract and the run's own reads need the
number while a position is being recovered -- so the contract is held here
instead: every function that reads a live position is classified, and a new
one fails this test until someone decides which kind it is. A function that
saves a position must ask the motion API's refusal before it reads.
"""

import ast

from tests.ast_seams import production_modules, walk_defs

_READERS = frozenset({'get_current_position', 'get_target_position', 'axis_positions'})

# Saves the position into a setting or a step. Must ask
# ui_helpers.unknown_position_refused(..., recording=True) before reading.
SAVES = 'saves'
# Shows the position; saves nothing.
DISPLAYS = 'displays'
# Runs inside, or hands the position straight to, a run whose prepare()
# refuses an unknown position and which ends the run when one is lost.
RUN = 'run'
# Feeds an API member that refuses an unknown position itself.
FEEDS_A_REFUSING_API = 'feeds a refusing API'
# Writes the position into a captured file's metadata.
CAPTURE_METADATA = 'capture metadata'

CLASSIFIED = {
    'modules/autofocus_runner.py::AutofocusRunner._calculate_params': RUN,
    'modules/autofocus_runner.py::AutofocusRunner._iterate': RUN,
    'modules/config_helpers.py::get_current_plate_position': FEEDS_A_REFUSING_API,
    'modules/lumascope_api/runtime_state.py::RuntimeState.get_well_label': CAPTURE_METADATA,
    'modules/manual_recording.py::_recording_position': CAPTURE_METADATA,
    'modules/recording_frames.py::frame_fact': CAPTURE_METADATA,
    'modules/protocol_step_runner.py::ProtocolStepRunner._grease_redist_w_pos': RUN,
    'ui/layer_control.py::LayerControl.execute_save_focus': SAVES,
    'ui/layer_control.py::LayerControl.execute_apply_focus_to_channel_steps': SAVES,
    'ui/motion_settings.py::XYStageControl.get_xy_targets': DISPLAYS,
    'ui/motion_settings.py::XYStageControl.ex_set_xbookmark': SAVES,
    'ui/motion_settings.py::XYStageControl.ex_set_ybookmark': SAVES,
    'ui/shader.py::ShaderViewer._update_status_bar': DISPLAYS,
    'ui/stage.py::Stage.draw_labware_io_calculations': DISPLAYS,
    'ui/stage.py::Stage.get_target_xy': DISPLAYS,
    'ui/stage.py::Stage.motion_enabled_io': DISPLAYS,
    'ui/vertical_control.py::VerticalControl.update_gui': DISPLAYS,
    'ui/vertical_control.py::VerticalControl.ex_set_bookmark': SAVES,
    'ui/vertical_control.py::VerticalControl.ex_set_all_bookmarks': SAVES,
}


def _own_nodes(fn):
    """The function's own body, nested defs excluded."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _readers_by_function():
    """``{'path::qualname': (first reader line, own nodes)}`` outside the motion API."""
    found = {}
    for rel, tree in production_modules():
        if rel == 'modules/lumascope_api/motion.py':
            continue
        for qualname, fn in walk_defs(tree.body):
            nodes = list(_own_nodes(fn))
            lines = [n.lineno for n in nodes if isinstance(n, ast.Attribute) and n.attr in _READERS]
            if lines:
                found[f'{rel}::{qualname}'] = (min(lines), nodes)
    return found


def test_every_position_reader_is_classified():
    found = set(_readers_by_function())

    assert found - set(CLASSIFIED) == set(), (
        'a function reads the live position and nobody has said what for. If it '
        'saves the position, ask ui_helpers.unknown_position_refused(..., '
        'recording=True) before the read; then classify it in CLASSIFIED.'
    )
    assert set(CLASSIFIED) - found == set(), (
        'classified functions that no longer read a position: remove them'
    )


def test_a_function_that_saves_a_position_asks_first():
    offenders = []
    for key, (first_read, nodes) in _readers_by_function().items():
        if CLASSIFIED.get(key) != SAVES:
            continue
        asks = [
            n.lineno
            for n in nodes
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'unknown_position_refused'
            and any(
                kw.arg == 'recording'
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in n.keywords
            )
        ]
        if not asks or min(asks) > first_read:
            offenders.append(key)

    assert offenders == [], (
        f'these save a live position without first asking whether the scope knows it: {offenders}'
    )
