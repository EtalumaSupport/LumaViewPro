# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulators stand in for the real boards, so their public surface
is held against the real drivers' by these guards: every name, every
parameter, and a ceiling on the names only the simulator has.

The behaviour tests for the simulators stay in tests/test_simulators.py;
this module holds only the surface guards, which measure the live
drivers and the production tree rather than one simulator's behaviour,
and so gate every commit from tests/guards/.
"""

import ast
import inspect

import pytest

from drivers.camera import Camera
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.simulated_camera import SimulatedCamera
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from tests.ast_seams import iter_package_modules


def test_simulated_led_board_has_every_public_name_of_the_real_board():
    """Simulated board has all public methods of the real board."""
    real_methods = {m for m in dir(LEDBoard) if not m.startswith('_')}
    sim_methods = {m for m in dir(SimulatedLEDBoard) if not m.startswith('_')}
    missing = real_methods - sim_methods
    assert not missing, f'SimulatedLEDBoard missing methods: {missing}'


def test_simulated_motor_board_has_every_public_name_of_the_real_board():
    """Simulated board has all public methods of the real board."""
    real_methods = {m for m in dir(MotorBoard) if not m.startswith('_')}
    sim_methods = {m for m in dir(SimulatedMotorBoard) if not m.startswith('_')}
    missing = real_methods - sim_methods
    assert not missing, f'SimulatedMotorBoard missing methods: {missing}'


def test_simulated_camera_has_every_public_name_of_the_camera_base():
    """Simulated camera implements all abstract methods from Camera ABC."""
    abstract_methods = {
        m for m in dir(Camera) if not m.startswith('_') and callable(getattr(Camera, m, None))
    }
    sim_methods = {m for m in dir(SimulatedCamera) if not m.startswith('_')}
    missing = abstract_methods - sim_methods
    assert not missing, f'SimulatedCamera missing methods: {missing}'


# ---------------------------------------------------------------------------
# Drop-in-replacement guards (all three board pairs)
# ---------------------------------------------------------------------------
#
# The three name-parity tests above
# ask one question -- does the simulator have every public NAME the real
# board has -- and all three passed while a documented SDK method was
# crashing against the simulator. `IlluminationAPI.wait_until_led_on()`
# calls `self._driver.wait_until_on(timeout_s)`; the real board accepts
# `timeout_s`, the simulator's override did not, so the call raised
# TypeError in every simulated run. A name-only comparison cannot see
# that, and the one test that touched the method on a sim scope set
# `_led_driver = None` first, so it never reached the driver.
#
# These guards close the two gaps a name check leaves.
#
# Why the drift exists at all, which is the part worth remembering:
#
#     SimulatedCamera(Camera)   inherits the ABC   -> 0 divergences
#     SimulatedLEDBoard         inherits nothing   -> 4 divergences
#     SimulatedMotorBoard       inherits nothing   -> 2 divergences
#
# The pair that shares a contract has never drifted; the two that share
# none hold every divergence. So the root is not six stale signatures, it
# is that nothing makes LED/motor drift UNREPRESENTABLE. Hand-mirroring
# the parameters would add defaults that do nothing on a simulator --
# decorative parity that reads as fixed while the next divergence is
# still free to appear. The five inert divergences are therefore
# allowlisted below against that structural fix, not patched here. The
# one that was actually crashing (`wait_until_on` missing `timeout_s`)
# is fixed in the simulator, where the parameter now bounds a loop that
# could previously spin forever.


_BOARD_PAIRS = (
    ('LEDBoard', LEDBoard, SimulatedLEDBoard),
    ('MotorBoard', MotorBoard, SimulatedMotorBoard),
    ('Camera', Camera, SimulatedCamera),
)

# Sim-only public names per board, recorded at introduction. This is a
# RATCHET, not an allowlist: it carries no per-name justification and
# exists only so new sim-only surface cannot accrete unnoticed. A
# deliberate addition raises the number here in the same commit.
#
# What the current entries cover: timing-mode controls and TIMING_*
# constants (test-harness affordances, on every simulator),
# `load_cycle_images` plus the camera's virtual-specimen focus modeling
# (`set_focal_z`, `set_blur_per_um`, ...), and six firmware-update
# methods on the motor simulator that anticipate the firmware-updating
# work landing on the FW branch. Those six have no caller yet; when that
# code calls them, `test_no_production_code_calls_simulator_only_names`
# below will require the REAL board to gain the same names, which is the
# point.
_SIM_ONLY_NAME_BUDGET = {
    'LEDBoard': 4,
    'MotorBoard': 12,
    'Camera': 13,
}

# Parameter divergences that stay until the simulators gain a shared
# contract with the real boards. Every one of these parameters is inert
# on a simulator -- there is no firmware to soft-reset, no empty response
# to stop on, no unsupported-command warning to suppress -- so adding
# them by hand would mean defaults that do nothing, and the next
# divergence would still be free to appear. Keyed by (board, method) so
# the entry survives reformatting.
_PARAM_DIVERGENCE_ALLOWLIST = {
    ('LEDBoard', 'enter_raw_repl'),
    ('LEDBoard', 'exchange_command'),
    ('LEDBoard', 'led_on'),
    ('MotorBoard', 'enter_raw_repl'),
    ('MotorBoard', 'exchange_command'),
}

# Production sites allowed to touch simulator-only surface. The single
# entry is the construction branch that BUILDS the simulator: inside
# `if simulate:` the driver provably is a SimulatedCamera, which is the
# one place production code can know that. Keyed by (file, name).
_SIM_ONLY_CALL_ALLOWLIST = {
    ('modules/lumascope_api/_lumascope.py', 'load_cycle_images'),
}


def _public_names(cls):
    return {name for name in dir(cls) if not name.startswith('_')}


def _sim_only_names(real, sim):
    return _public_names(sim) - _public_names(real)


@pytest.mark.parametrize('label,real,sim', _BOARD_PAIRS, ids=[p[0] for p in _BOARD_PAIRS])
def test_simulator_accepts_the_same_parameters_as_the_real_board(label, real, sim):
    """A call that works on the real board must work on the simulator.

    Compares PARAMETER NAMES only, deliberately. Annotations diverge
    harmlessly all over these classes -- the simulators are annotated
    more completely than the real boards, 15 such differences on the LED
    pair alone -- and an annotation never changes whether a call is
    accepted. A parameter name does.

    Bidirectional, because both directions are real hazards: a parameter
    the simulator lacks crashes every simulated run of a production code
    path, and a parameter only the simulator has invites test code to
    depend on something hardware will reject.
    """
    divergences = []
    for name in sorted(_public_names(real) & _public_names(sim)):
        try:
            real_params = list(inspect.signature(getattr(real, name)).parameters)
            sim_params = list(inspect.signature(getattr(sim, name)).parameters)
        except (TypeError, ValueError):
            continue  # C-implemented or otherwise non-introspectable
        if real_params != sim_params and (label, name) not in _PARAM_DIVERGENCE_ALLOWLIST:
            divergences.append(f'  {name}:\n      real={real_params}\n      sim ={sim_params}')

    assert not divergences, (
        f'{sim.__name__} is not a drop-in replacement for {label} -- these '
        f'methods take different parameters, so a caller written against one '
        f'breaks against the other:\n' + '\n'.join(divergences)
    )


@pytest.mark.parametrize('label,real,sim', _BOARD_PAIRS, ids=[p[0] for p in _BOARD_PAIRS])
def test_no_production_code_calls_simulator_only_names(label, real, sim):
    """Production code may not depend on surface only the simulator has.

    Simulators legitimately carry extra surface -- timing controls,
    virtual-specimen modeling, image-cycle loading -- so demanding an
    empty sim-only set would mean a large allowlist whose upkeep exceeds
    its signal. The hazard worth guarding is narrower and exact: code
    under `modules/` or `ui/` that calls a name only the simulator has
    works in every test and fails on hardware.

    Tests are excluded on purpose. Driving simulator-only affordances is
    what test code is FOR -- `conftest.sim_scope` calls `set_timing_mode`
    and `load_cycle_images`.
    """
    sim_only = _sim_only_names(real, sim)

    hits = []
    for rel_path, tree in iter_package_modules(('modules', 'ui')):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in sim_only:
                continue
            if (rel_path, node.attr) in _SIM_ONLY_CALL_ALLOWLIST:
                continue
            hits.append(f'  {rel_path}:{node.lineno}: .{node.attr}')

    assert not hits, (
        f'Production code references {sim.__name__}-only surface, which does '
        f'not exist on {label} and will fail on hardware:\n'
        + '\n'.join(sorted(hits))
        + f'\n\nFix by adding the name to {label} (making it real) or by '
        f'removing the production dependency.'
    )


def test_no_allowlist_entry_outlives_its_divergence():
    """Delete an exemption the moment the thing it excuses is gone.

    A stale entry is worse than no entry: it silently excuses the NEXT
    divergence on the same method, which is exactly the regression this
    guard exists to catch. Both allowlists are checked, so landing the
    shared-contract fix forces the entries out in the same commit.
    """
    stale_params = []
    for label, real, sim in _BOARD_PAIRS:
        for name in sorted(_public_names(real) & _public_names(sim)):
            if (label, name) not in _PARAM_DIVERGENCE_ALLOWLIST:
                continue
            try:
                real_params = list(inspect.signature(getattr(real, name)).parameters)
                sim_params = list(inspect.signature(getattr(sim, name)).parameters)
            except (TypeError, ValueError):
                continue
            if real_params == sim_params:
                stale_params.append((label, name))

    shared = {
        (label, name)
        for label, real, sim in _BOARD_PAIRS
        for name in _public_names(real) & _public_names(sim)
    }
    unknown = sorted(entry for entry in _PARAM_DIVERGENCE_ALLOWLIST if entry not in shared)

    assert not stale_params, (
        f'These methods now agree and their _PARAM_DIVERGENCE_ALLOWLIST '
        f'entries must be deleted: {sorted(stale_params)}'
    )
    assert not unknown, (
        f'These _PARAM_DIVERGENCE_ALLOWLIST entries name a method that no '
        f'longer exists on both classes: {unknown}'
    )

    sim_only_names = set()
    for _, real, sim in _BOARD_PAIRS:
        sim_only_names |= _sim_only_names(real, sim)
    stale_calls = sorted(
        entry for entry in _SIM_ONLY_CALL_ALLOWLIST if entry[1] not in sim_only_names
    )
    assert not stale_calls, (
        f'These _SIM_ONLY_CALL_ALLOWLIST entries name surface that is no '
        f'longer simulator-only, so the exemption must be deleted: {stale_calls}'
    )


@pytest.mark.parametrize('label,real,sim', _BOARD_PAIRS, ids=[p[0] for p in _BOARD_PAIRS])
def test_simulator_only_surface_does_not_grow(label, real, sim):
    """Ratchet: sim-only public surface may not accrete unnoticed.

    Every name here is one the real board does not have, so each is a
    place production code could come to depend on something hardware
    cannot do. Growth should be a decision, not a side effect.
    """
    sim_only = _sim_only_names(real, sim)
    budget = _SIM_ONLY_NAME_BUDGET[label]
    assert len(sim_only) <= budget, (
        f'{sim.__name__} now has {len(sim_only)} public names absent from '
        f'{label}, over the recorded {budget}: {sorted(sim_only)}\n\n'
        f'If the addition is deliberate, raise _SIM_ONLY_NAME_BUDGET '
        f"['{label}'] in this commit and say why in the message. If it is "
        f'not, the name probably belongs on {label} too.'
    )


# Announced at the end of every run (tests/ratchets.py).
from tests import ratchets as _ratchets

for _label, _real, _sim in _BOARD_PAIRS:
    _ratchets.register(
        f'simulators: sim-only public names on {_label}',
        (lambda real=_real, sim=_sim: len(_sim_only_names(real, sim))),
        _SIM_ONLY_NAME_BUDGET[_label],
    )
