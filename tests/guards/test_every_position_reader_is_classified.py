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
saves a position takes it from ``positions_to_keep`` and from no other read:
that read refuses an unknown or moving axis from the same snapshot as the
number, where a refusal asked before a separate read let a home that started
between the two pass the number Z left behind (``focus_z_um`` answered a
stale 3000.0 with Z homing).
"""

import ast

from tests.ast_seams import MemberNames, production_modules, walk_defs

_READERS = MemberNames(
    'modules/lumascope_api/motion.py',
    {
        'get_current_position',
        'get_current_positions',
        'get_target_positions',
        'get_actual_position',
        'axis_positions',
        'settled_positions',
        'positions_to_keep',
    },
)

_KEEPING_READ = 'positions_to_keep'

# Saves the position into a setting or a step. Must read it through
# positions_to_keep and no other reader.
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
    'modules/config_helpers.py::kept_plate_position': SAVES,
    'modules/lumascope_api/_lumascope.py::Lumascope.__init__._sim_stage_xy': DISPLAYS,
    'modules/lumascope_api/_lumascope.py::Lumascope.__init__._sim_sensor_inputs': DISPLAYS,
    'modules/lumascope_api/protocols.py::ProtocolsAPI.focus_z_um': SAVES,
    'modules/lumascope_api/runtime_state.py::RuntimeState.get_well_label': CAPTURE_METADATA,
    'modules/recording_frames.py::frame_fact': CAPTURE_METADATA,
    'modules/scope_session.py::ScopeSession.get_sequenced_capture_config': FEEDS_A_REFUSING_API,
    'modules/scope_session.py::ScopeSession.status_at_edge': DISPLAYS,
    'modules/protocol_step_runner.py::ProtocolStepRunner._grease_redist_w_pos': RUN,
    'ui/motion_settings.py::XYStageControl.get_xy_targets': DISPLAYS,
    'ui/protocol_settings.py::ProtocolSettings.insert_step_ex': DISPLAYS,
    'ui/shader.py::ShaderViewer._update_status_bar': DISPLAYS,
    'ui/stage.py::Stage.draw_labware_io_calculations': DISPLAYS,
    'ui/vertical_control.py::VerticalControl._show_z_target': DISPLAYS,
    'ui/vertical_control.py::VerticalControl.show_z': DISPLAYS,
}


def _own_nodes(fn):
    """The function's own body, nested defs excluded.

    A lambda is part of the body that writes it: it has no name of its own
    to classify, so its reads are the enclosing function's.
    """
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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
        f'saves the position, read it through {_KEEPING_READ}; then classify it in '
        'CLASSIFIED.'
    )
    assert set(CLASSIFIED) - found == set(), (
        'classified functions that no longer read a position: remove them'
    )


def test_a_function_that_saves_a_position_reads_it_through_the_keeping_read():
    offenders = {}
    for key, (_, nodes) in _readers_by_function().items():
        if CLASSIFIED.get(key) != SAVES:
            continue
        reads = {n.attr for n in nodes if isinstance(n, ast.Attribute) and n.attr in _READERS}
        if reads != {_KEEPING_READ}:
            offenders[key] = sorted(reads)

    assert offenders == {}, (
        f'these save a live position through a read other than {_KEEPING_READ}, '
        f'which alone refuses an unknown axis in the snapshot it reads: {offenders}'
    )
