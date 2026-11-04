# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every path that keeps the live position takes its number from a settled read.

``MotionAPI.positions_to_keep`` and ``settled_positions`` read each axis's
state and number in one snapshot and refuse a moving axis;
``positions_to_keep`` refuses an unknown or homing one there too. A keeper that read the bare cache
(``get_current_position(s)``) instead would save a place the axis was
passing through, and one that asked the state apart from the read would
let a move that ended in between pass a number taken while it ran: a step
was recorded at 5990.3 with the scope at 7000 that way. Each keeper below
reaches a settled read through the one link named beside it, and calls
no bare cache read of its own.
"""

from tests.ast_seams import MemberNames, direct_call_names, find_def

_SESSION = 'modules/scope_session.py'
_RUNNER = 'modules/protocol_runner.py'

# (file, class or None, keeper, the call that carries its number).
_KEEPERS = (
    ('modules/config_helpers.py', None, 'get_current_plate_position', 'settled_positions'),
    ('modules/config_helpers.py', None, 'kept_plate_position', 'positions_to_keep'),
    ('modules/lumascope_api/protocols.py', 'ProtocolsAPI', 'focus_z_um', 'positions_to_keep'),
    (_SESSION, 'ScopeSession', 'get_sequenced_capture_config', 'settled_positions'),
    (_SESSION, 'ScopeSession', 'save_bookmark', 'kept_plate_position'),
    (_SESSION, 'ScopeSession', 'get_current_plate_position', 'kept_plate_position'),
    (_SESSION, 'ScopeSession', 'plate_position_on', 'get_current_plate_position'),
    (_SESSION, 'ScopeSession', 'add_step', 'plate_position_on'),
    (_SESSION, 'ScopeSession', 'update_step', 'plate_position_on'),
    (_SESSION, 'ScopeSession', 'save_focus', 'focus_z_um'),
    (_SESSION, 'ScopeSession', 'save_all_bookmarks', 'focus_z_um'),
    (_SESSION, 'ScopeSession', 'apply_focus_to_channel_steps', 'focus_z_um'),
    (_RUNNER, 'ProtocolRunner', 'start_composite', 'plate_position_on'),
    (_RUNNER, 'ProtocolRunner', 'start_autofocus', 'plate_position_on'),
    (_RUNNER, 'ProtocolRunner', 'start_zstack', 'plate_position_on'),
)

_BARE_READS = MemberNames(
    ('modules/lumascope_api/motion.py',), {'get_current_position', 'get_current_positions'}
)


def test_every_keeper_reads_the_live_position_through_the_settled_read():
    offenders = []
    for rel_path, class_name, name, link in _KEEPERS:
        fn = find_def(rel_path, name, class_name)
        if fn is None:
            offenders.append(f'{rel_path}: {name} not found')
            continue
        calls = set(direct_call_names(fn))
        if link not in calls:
            offenders.append(f'{rel_path}: {name} does not call {link}')
        if bare := sorted(calls & _BARE_READS):
            offenders.append(f'{rel_path}: {name} reads the bare cache through {bare}')
    assert offenders == []
