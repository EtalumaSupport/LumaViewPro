# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Ratchet on parametrizes whose cases are computed rather than written.

A parametrize is for different behaviours, or for cases that each need
their own fixture or isolation. One assertion over a discovered set or a
data table (exception classes, API members, files, a register table, a dB
range) is ONE test that walks the set and names every offender in its
failure, the form of ``test_every_refusal_states_its_reason.py``. Fanned
out, it costs a collected test per item on every run: one guard's three
checks over about 120 exception classes were 359 tests.

The shape this counts is the one the source shows: a ``parametrize`` (or a
fixture's ``params``) whose cases are not written at the site, a call, a
comprehension, or a name bound to either or imported. Cases written out
as a list or tuple, at the site or in a module-level name, are not
counted. Where a computed set is fair, each case its own scope, model or
dialect, say so with ``# parametrize-justified: <why>`` on the decorator's
line or within the three lines above it.

The pin is an EQUALITY per file: a rise names the remedy, a fall lowers
the file's entry in the same commit, and an entry is never raised.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, iter_package_modules

_JUSTIFICATION = 'parametrize-justified'
_JUSTIFICATION_WINDOW = 3  # lines above the decorator that may carry it

_WRITTEN = (ast.List, ast.Tuple, ast.Set, ast.Dict)
_COMPUTED_INSIDE = (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp, ast.Starred)


def _module_bindings(tree):
    """``{name: value}`` for each module-level single-name assignment."""
    bound = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            bound[node.targets[0].id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            bound[node.target.id] = node.value
    return bound


def _cases(call):
    """The cases a ``parametrize`` or ``fixture(params=)`` call takes, else None."""
    if not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr == 'parametrize':
        for keyword in call.keywords:
            if keyword.arg == 'argvalues':
                return keyword.value
        return call.args[1] if len(call.args) > 1 else None
    if call.func.attr == 'fixture':
        for keyword in call.keywords:
            if keyword.arg == 'params':
                return keyword.value
    return None


def _written(cases, bindings):
    if isinstance(cases, ast.Name) and cases.id in bindings:
        cases = bindings[cases.id]
    if not isinstance(cases, _WRITTEN):
        return False
    return not any(isinstance(node, _COMPUTED_INSIDE) for node in ast.walk(cases))


def _justified(lines, lineno):
    window = lines[max(0, lineno - 1 - _JUSTIFICATION_WINDOW) : lineno]
    return any(_JUSTIFICATION in line for line in window)


def computed_parametrize_sites():
    """``{'tests/<file>.py': ['<line>: <cases>', ...]}`` for each unjustified site."""
    found: dict[str, list[str]] = {}
    for rel_path, tree in iter_package_modules(('tests',)):
        bindings = _module_bindings(tree)
        lines = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            cases = _cases(node)
            if cases is None or _written(cases, bindings):
                continue
            if lines is None:
                lines = (REPO_ROOT / rel_path).read_text(encoding='utf-8').splitlines()
            if not _justified(lines, node.lineno):
                found.setdefault(rel_path, []).append(f'{node.lineno}: {ast.unparse(cases)}')
    return found


# Pinned at its first run. Lower a value in the same commit that collapses a
# site into one test or justifies it; never raise one.
_COMPUTED_PIN = {
    'tests/capability/test_capability_probes.py': 1,
    'tests/guards/test_architecture_fixes.py': 2,
    'tests/guards/test_every_ranged_setting_is_refused_and_replaced.py': 2,
    'tests/guards/test_every_refusal_declares_its_cause.py': 1,
    'tests/test_a_cell_count_method_file_is_read_by_its_owner.py': 1,
    'tests/test_a_failed_move_is_shown_once.py': 1,
    'tests/test_a_folder_post_processing_refuses_is_logged_once.py': 1,
    'tests/test_a_layer_or_led_channel_the_scope_cannot_act_on_is_refused.py': 1,
    'tests/test_a_manual_scope_is_complete.py': 2,
    'tests/test_a_motion_argument_the_scope_cannot_act_on_is_refused.py': 1,
    'tests/test_a_protocols_schedule_is_refused_never_changed.py': 1,
    'tests/test_a_recording_refuses_what_falsifies_its_file.py': 2,
    'tests/test_a_run_reads_its_settings_once.py': 1,
    'tests/test_a_simulated_fx2_scope_runs_the_fx2_drivers.py': 1,
    'tests/test_a_stage_interlock_refuses_motion.py': 1,
    'tests/test_a_step_cell_is_its_columns_type.py': 1,
    'tests/test_an_fx2_exposure_is_the_setting_at_every_window.py': 3,
    'tests/test_an_interlock_at_a_home_is_reported_not_raised.py': 2,
    'tests/test_an_led_command_for_absent_hardware_is_refused.py': 2,
    'tests/test_api_examples.py': 3,
    'tests/test_attended_run_flag.py': 2,
    'tests/test_autofocus_setup_failure_does_not_latch.py': 2,
    'tests/test_camera_getter_sentinel_containment.py': 4,
    'tests/test_disk_space_helper.py': 2,
    'tests/test_going_to_a_step_is_the_sessions_move.py': 1,
    'tests/test_hyperstack_run_trigger.py': 2,
    'tests/test_labware_name_resolution.py': 3,
    'tests/test_led_cap_published_by_driver.py': 1,
    'tests/test_saving_an_unknown_position_is_refused_once.py': 1,
    'tests/test_sim_wire_conformance.py': 3,
    'tests/test_sim_wire_fast_tier_parity.py': 1,
    'tests/test_sim_wire_faults.py': 3,
    'tests/test_sim_wire_port.py': 4,
    'tests/test_sim_wire_reconnect.py': 2,
    'tests/test_simulator_tier.py': 2,
    'tests/test_step_name_builder.py': 1,
    'tests/test_the_catalogue_names_each_models_boards.py': 2,
    'tests/test_tmcm6110.py': 1,
    'tests/test_zprojection_numpy.py': 1,
}


def test_no_new_parametrize_fans_out_a_computed_set():
    found = computed_parametrize_sites()
    counts = {rel_path: len(sites) for rel_path, sites in found.items()}
    risen = {
        rel_path: found[rel_path]
        for rel_path, count in counts.items()
        if count > _COMPUTED_PIN.get(rel_path, 0)
    }
    fallen = {
        rel_path: (pinned, counts.get(rel_path, 0))
        for rel_path, pinned in _COMPUTED_PIN.items()
        if counts.get(rel_path, 0) < pinned
    }
    assert not risen, (
        'A parametrize fans a computed set out into one test per item. If it '
        'is one assertion over the set, make it one test that walks the set '
        'and names every offender; if each case is its own behaviour or needs '
        'its own fixture, write the cases out or mark the site '
        f'"# {_JUSTIFICATION}: <why>". Sites: {risen}'
    )
    assert not fallen, f'Lower these entries in _COMPUTED_PIN to the count found: {fallen}'
