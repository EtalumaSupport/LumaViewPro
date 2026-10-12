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
through ``.simulation`` or is a local named ``simulation``. A test whose
subject IS a simulator or the driver over one, with no scope, uses the
doors as its fixture: such a file is named in ``_OWN_DOORS`` with why, and
is not counted.

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
_OWN_DOORS: dict[str, str] = {
    'tests/test_simulators.py': 'the fast boards and camera, door by door',
    'tests/test_an_fx2_frame_ends_where_the_device_ends_it.py': (
        'the FX2 stream over a bare simulated FX2, no scope'
    ),
    'tests/test_simulated_camera_stall.py': "the simulated camera's stall, on a bare camera",
    'tests/test_tmcm6110.py': 'the TMCM-6110 driver over its simulated board, no scope',
    'tests/test_no_reply_is_no_reply.py': "the EL-0940 drivers' reading of a dropped reply, at the wire",
    'tests/test_sim_wire_port.py': 'the emulated port itself',
    'tests/test_sim_wire_link.py': "the motor board's USB link over the emulated port, no scope",
    'tests/test_sim_wire_led.py': 'the LED firmware behind its own emulated port, no scope',
    'tests/test_sim_wire_faults.py': "the firmware simulator's hardware faults and write oracle",
    'tests/test_sim_wire_homing.py': "the motor driver's home on the shipped firmware, no scope",
    'tests/test_a_home_is_believed_only_when_it_happened.py': (
        "the motor driver's reading of a home, on the firmware simulator"
    ),
}


def _through_simulation(node: ast.Attribute) -> bool:
    receiver = node.value
    while isinstance(receiver, (ast.Attribute, ast.Call, ast.Subscript)):
        if isinstance(receiver, ast.Attribute) and receiver.attr == 'simulation':
            return True
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
    # A test that took the member set into a local: ``simulation = scope.simulation``.
    return isinstance(receiver, ast.Name) and receiver.id == 'simulation'


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
    'tests/test_an_unplugged_fx2_is_removed.py': 1,
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
