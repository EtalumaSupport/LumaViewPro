# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A persisted objective the turret does not hold must not set image scale.

The session starts at turret position 1, so position 1's assignment is
the only honest starting objective on a turret model. The stored
objective_id is a leftover from the previous session (a session that
ended on another slot, or an assignment reset, leaves it naming glass
the turret does not hold); adopting the slot's objective BEFORE any
consumer reads settings is what keeps the scale bar and saved-image
metadata from carrying a fabricated pixel size.
"""

import ast
import threading

from modules.scope_session import ScopeSession
from tests.ast_seams import parse_module


def _session_with_settings(settings: dict) -> ScopeSession:
    session = ScopeSession.__new__(ScopeSession)
    session.settings = settings
    session.settings_lock = threading.Lock()
    return session


def _settings(objective_id, turret_objectives):
    return {
        'objective_id': objective_id,
        'turret_objectives': turret_objectives,
    }


class TestAdoptTurretSlot1Objective:
    def test_stale_objective_is_replaced_by_slot1_assignment(self):
        session = _session_with_settings(
            _settings('40x w/collar', {1: '1.25x Oly', 2: '20x Oly', 3: None, 4: None})
        )
        session.adopt_turret_slot1_objective(model_has_turret=True)
        assert session.settings['objective_id'] == '1.25x Oly'

    def test_matching_objective_is_untouched(self):
        session = _session_with_settings(_settings('20x Oly', {1: '20x Oly', 2: None}))
        session.adopt_turret_slot1_objective(model_has_turret=True)
        assert session.settings['objective_id'] == '20x Oly'

    def test_unassigned_slot1_keeps_stored_objective(self):
        # Nothing assigned at the starting position: no invented
        # objective; the unassigned-slot prompt owns resolving this.
        session = _session_with_settings(_settings('40x w/collar', {1: None, 2: '20x Oly'}))
        session.adopt_turret_slot1_objective(model_has_turret=True)
        assert session.settings['objective_id'] == '40x w/collar'

    def test_non_turret_model_is_a_no_op(self):
        # On non-turret models objective_id is the user's free choice --
        # even when stale slot assignments linger in settings from a
        # previous scope on the same machine.
        session = _session_with_settings(_settings('40x w/collar', {1: '20x Oly'}))
        session.adopt_turret_slot1_objective(model_has_turret=False)
        assert session.settings['objective_id'] == '40x w/collar'

    def test_missing_or_null_turret_config_is_a_no_op(self):
        for turret_objectives in (None, {}):
            session = _session_with_settings(_settings('20x Oly', turret_objectives))
            session.adopt_turret_slot1_objective(model_has_turret=True)
            assert session.settings['objective_id'] == '20x Oly'


def _call_linenos(rel_path: str, func_name: str, class_name: str | None = None) -> dict:
    """First lineno of each dotted call inside one function body."""
    module = parse_module(rel_path)
    scope = module.body
    if class_name is not None:
        scope = next(
            node.body
            for node in scope
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    func = next(
        node
        for node in scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
    )
    linenos: dict[str, int] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        parts = []
        target = node.func
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        dotted = '.'.join(reversed(parts))
        linenos.setdefault(dotted, node.lineno)
    return linenos


class TestAdoptionRunsBeforeTheStamps:
    """Call-order pins: adoption must precede every settings consumer.

    Both stamp paths (startup load_settings and the reconnect path,
    which never re-runs load_settings) build ScopeInitConfig from
    settings and push it into scope.initialize(); an adoption that runs
    after either build stamps the stale objective into runtime state.
    """

    def _assert_adopts_before_config(self, rel_path, method_name, class_name, adopt_key):
        linenos = _call_linenos(rel_path, method_name, class_name)
        adopt = linenos.get(adopt_key)
        build = linenos.get('ScopeInitConfig.from_settings')
        assert adopt is not None, f'{method_name} must adopt the slot-1 objective'
        assert build is not None, f'{method_name} must build ScopeInitConfig'
        assert adopt < build, (
            f'{method_name} adopts the slot-1 objective AFTER building '
            f'ScopeInitConfig -- the stale objective is already stamped'
        )

    def test_configure_scope_adopts_before_config_build(self):
        # The startup site's bring-up is the Session's now; the GUI calls
        # configure_scope() where its adopt call used to sit.
        self._assert_adopts_before_config(
            'modules/scope_session.py',
            'configure_scope',
            'ScopeSession',
            'self.adopt_turret_slot1_objective',
        )

    def test_reconnect_adopts_before_config_build(self):
        self._assert_adopts_before_config(
            'ui/microscope_settings.py',
            'reconnect',
            'MicroscopeSettings',
            'ctx.session.adopt_turret_slot1_objective',
        )

    def test_startup_session_no_longer_looks_up_the_position(self):
        # Startup is position 1 by rule; a lookup keyed on the stored
        # objective is the mechanism that let a stale objective steer
        # the physical turret.
        linenos = _call_linenos(
            'modules/scope_session.py', 'start_application_session', 'ScopeSession'
        )
        lookup = [name for name in linenos if 'get_turret_position_for_objective_id' in name]
        assert not lookup, (
            f'start_application_session consults the stored objective for '
            f'the startup position again: {lookup}'
        )
