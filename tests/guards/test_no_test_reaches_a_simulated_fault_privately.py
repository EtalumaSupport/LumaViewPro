# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Ratchet on tests that switch a simulated fault through the simulator's own
doors rather than through ``scope.simulation``.

A test of the scope injects a fault the way an integrator does, by name,
through ``scope.simulation``: the member answers on every board family that
produces the fault and refuses by name on one that cannot, so the test does
not depend on which simulator stands behind the scope. A test that sets a
fast board's ``_fail_on`` or a TMCM-6110's ``silent`` itself reaches past the
API into one simulator's internals, and is pinned to that simulator.

Counted: a read, write, call or ``setattr`` of a name only the simulators'
fault doors carry, from any test file, unless the receiver is reached
through ``.simulation``. A test whose subject IS a simulator or the driver
over one, with no scope, uses the doors as its fixture: such a file is named
in ``_OWN_DOORS`` with why, and is not counted.

The pin is an EQUALITY per file, at the guard's first run: a fall lowers the
file's entry in the same commit that moves its tests; an entry is never
raised.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules

# Names only the simulators' fault doors carry (the fast boards and camera,
# the wire sims' port and board, the TMCM-6110 and FX2 simulators).
_DOOR_NAMES = frozenset(
    {
        '_fail_on',
        'fail_after',
        'drop_replies',
        'stop_dropping',
        'hold_travel',
        'hold_frames',
        'release_frames',
        'leave_bus',
        'refuse_write',
        'allow_write',
        'unplug',
        'replug',
        'reboot',
        'drop_next_reply',
        'inject',
        'silent',
        'plugged',
        'lid_open',
        'powered',
        'ignore_stop',
        'break_right_switch',
        'go_silent',
    }
)

# Test files whose subject is a simulator, or a driver over its simulated
# board with no scope: the doors are what they test or their fixture.
_OWN_DOORS: dict[str, str] = {}


def _through_simulation(node: ast.Attribute) -> bool:
    receiver = node.value
    while isinstance(receiver, (ast.Attribute, ast.Call, ast.Subscript)):
        if isinstance(receiver, ast.Attribute) and receiver.attr == 'simulation':
            return True
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
    return False


def _setattr_name(node: ast.Call) -> str | None:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
    if name != 'setattr' or len(node.args) < 2:
        return None
    target = node.args[1]
    return target.value if isinstance(target, ast.Constant) else None


def private_fault_reaches() -> dict[str, list[str]]:
    """``{'tests/<file>.py': ['<line>: <name>', ...]}`` outside ``_OWN_DOORS``."""
    found: dict[str, list[str]] = {}
    for rel_path, tree in iter_package_modules(('tests',)):
        if rel_path.startswith('tests/guards/') or rel_path in _OWN_DOORS:
            continue
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute) and node.attr in _DOOR_NAMES:
                if not _through_simulation(node):
                    name = node.attr
            elif isinstance(node, ast.Call) and _setattr_name(node) in _DOOR_NAMES:
                name = _setattr_name(node)
            if name is not None:
                found.setdefault(rel_path, []).append(f'{node.lineno}: {name}')
    return found


# Pinned at the guard's first run. Lower a value in the commit that moves the
# file's tests onto scope.simulation; never raise one.
_PRIVATE_REACH_PIN: dict[str, int] = {
    'tests/capability/serial_p01_cable_pull.py': 2,
    'tests/test_a_camera_stream_stall_is_reported.py': 6,
    'tests/test_a_failed_home_raises_why.py': 1,
    'tests/test_a_failed_position_read_is_never_a_position.py': 7,
    'tests/test_a_home_is_believed_only_when_it_happened.py': 1,
    'tests/test_a_motion_status_read_never_answers_a_failure.py': 4,
    'tests/test_a_move_is_judged_when_it_ends.py': 1,
    'tests/test_a_position_is_answered_only_when_known.py': 4,
    'tests/test_a_relative_move_adds_to_the_boards_target.py': 2,
    'tests/test_a_stop_ends_the_move.py': 6,
    'tests/test_a_wire_client_is_refused_as_a_python_caller_is.py': 1,
    'tests/test_an_arrival_belongs_to_the_move_that_was_seen_arriving.py': 3,
    'tests/test_an_fx2_frame_ends_where_the_device_ends_it.py': 1,
    'tests/test_an_unplugged_fx2_is_removed.py': 8,
    'tests/test_audit_fixes.py': 3,
    'tests/test_capture_requires_streaming.py': 1,
    'tests/test_issue_539_motion_poll_no_traceback.py': 1,
    'tests/test_lumascope_api.py': 3,
    'tests/test_motion_state_gate.py': 2,
    'tests/test_no_reply_is_no_reply.py': 9,
    'tests/test_sim_wire_faults.py': 7,
    'tests/test_sim_wire_homing.py': 1,
    'tests/test_sim_wire_led.py': 3,
    'tests/test_sim_wire_led_link.py': 5,
    'tests/test_sim_wire_link.py': 9,
    'tests/test_sim_wire_port.py': 4,
    'tests/test_sim_wire_reconnect.py': 5,
    'tests/test_simulated_camera_stall.py': 1,
    'tests/test_simulators.py': 18,
    'tests/test_the_ls720_in_the_simulator.py': 10,
    'tests/test_the_target_is_what_was_commanded.py': 1,
    'tests/test_tmcm6110.py': 20,
}


def test_no_new_test_reaches_a_simulated_fault_privately():
    found = private_fault_reaches()
    counts = {rel_path: len(sites) for rel_path, sites in found.items()}
    risen = {
        rel_path: found[rel_path]
        for rel_path, count in counts.items()
        if count > _PRIVATE_REACH_PIN.get(rel_path, 0)
    }
    fallen = {
        rel_path: (pinned, counts.get(rel_path, 0))
        for rel_path, pinned in _PRIVATE_REACH_PIN.items()
        if counts.get(rel_path, 0) < pinned
    }
    assert not risen, (
        "A test switches a simulated fault through the simulator's own door. "
        'Inject it through scope.simulation, which names the fault and answers '
        'on every board family; a test whose subject is the simulator or the '
        f'driver over it goes in _OWN_DOORS with why. Sites: {risen}'
    )
    assert not fallen, f'Lower these entries in _PRIVATE_REACH_PIN to the count found: {fallen}'


def test_every_own_doors_file_exists():
    from tests.ast_seams import REPO_ROOT

    missing = [rel_path for rel_path in _OWN_DOORS if not (REPO_ROOT / rel_path).exists()]
    assert not missing, f'_OWN_DOORS names files that are gone: {missing}'
