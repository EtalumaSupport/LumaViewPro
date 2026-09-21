# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol naming glass this scope cannot reach is refused at the LOAD.

The run refused it; the load accepted it. So a user could open a protocol,
see its steps listed, click through them, and only find out at Run that
the scope was never able to perform it -- and every edit made in between
was made against a protocol that was never admissible.

Refusing at the load is the API's job rather than the panel's, which is
what makes it true for REST and a headless caller too. They get it the
same way they already get the refusal for a plate the installation's
labware catalogue does not have.

The reason a load gives is the reason a run gives, deliberately: one rule,
so a caller branching on `turret_objectives_unassigned` handles both
without knowing which surface it came from.
"""

from __future__ import annotations

import ast
import dataclasses

import pytest

from tests.ast_seams import find_def, iter_package_modules, parse_module

from modules.exceptions import ProtocolRunRefusedError
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executors,
    scope,
)


ON_TURRET = '10x Oly'
NOT_ON_TURRET = '20x Oly'


def _turret(scope, monkeypatch, *, has_turret=True, carries=(ON_TURRET,)):
    monkeypatch.setattr(
        scope, 'capabilities', dataclasses.replace(scope.capabilities, has_turret=has_turret)
    )
    monkeypatch.setattr(
        scope.runtime_state, 'get_turret_config', lambda: dict(enumerate(carries, start=1))
    )


def _write_protocol(tmp_path, *objectives):
    """A real protocol file on disk naming exactly these objectives."""
    protocol = _make_multi_step_protocol(
        [{'objective': o, 'name': f'step_{i}'} for i, o in enumerate(objectives)]
    )
    path = tmp_path / 'probe_protocol.tsv'
    protocol.to_file(file_path=path)
    return path


class TestTheLoadRefusesWhatTheRunWouldRefuse:
    def test_glass_no_slot_carries_is_refused_at_load(self, scope, monkeypatch, tmp_path):
        _turret(scope, monkeypatch, carries=(ON_TURRET,))
        path = _write_protocol(tmp_path, NOT_ON_TURRET)

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.load_protocol(file_path=path)

        assert refusal.value.reason == 'turret_objectives_unassigned'

    def test_an_unassigned_turret_refuses_at_load(self, scope, monkeypatch, tmp_path):
        _turret(scope, monkeypatch, carries=())
        path = _write_protocol(tmp_path, ON_TURRET)

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.load_protocol(file_path=path)

        assert refusal.value.reason == 'turret_objectives_unassigned'

    def test_a_turretless_scope_refuses_a_multi_objective_file(self, scope, monkeypatch, tmp_path):
        _turret(scope, monkeypatch, has_turret=False, carries=())
        path = _write_protocol(tmp_path, ON_TURRET, NOT_ON_TURRET)

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.load_protocol(file_path=path)

        assert refusal.value.reason == 'objectives_require_turret'

    def test_an_admissible_file_still_loads(self, scope, monkeypatch, tmp_path):
        _turret(scope, monkeypatch, carries=(ON_TURRET, NOT_ON_TURRET))
        path = _write_protocol(tmp_path, ON_TURRET, NOT_ON_TURRET)

        protocol = scope.protocols.load_protocol(file_path=path)

        assert protocol.num_steps() == 2

    def test_the_refusal_arrives_before_the_caller_holds_a_protocol(
        self, scope, monkeypatch, tmp_path
    ):
        """Nothing half-loaded: the caller gets an exception, not an object.

        A load that returned an inadmissible Protocol and warned beside it
        would leave every caller free to use it anyway.
        """
        _turret(scope, monkeypatch, carries=(ON_TURRET,))
        path = _write_protocol(tmp_path, NOT_ON_TURRET)

        loaded = None
        with pytest.raises(ProtocolRunRefusedError):
            loaded = scope.protocols.load_protocol(file_path=path)

        assert loaded is None, 'the refusal arrived after the protocol was built and returned'


class TestTheGuiNoLongerKeepsItsOwnCopy:
    def test_the_panel_has_no_objective_validator(self):
        """The GUI copy exempted single-objective protocols; the rule does not.

        Two answers to one question is how the load came to accept what the
        run refused. Pinned by absence because a re-added copy would pass
        every behavioural test while the divergence came back.
        """
        assert find_def('ui/protocol_settings.py', '_validate_objectives_in_protocol') is None, (
            'the GUI is holding its own version of the admissibility rule again'
        )

    def test_nothing_reads_or_writes_the_dead_protocol_field(self):
        """app_context.protocol had two writers and no readers at all.

        It was documented as the canonical owner of the current protocol
        while the live store was the panel's own attribute, so a reader who
        believed the comment would have got None. Swept as an ATTRIBUTE
        rather than as text: `ctx.protocol` and `ctx.protocols` differ by
        one character, and only one of them is the dead one.
        """
        offenders = []
        for rel_path, tree in iter_package_modules(('modules', 'ui')):
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Attribute) and node.attr == 'protocol'):
                    continue
                base = node.value
                # `ctx.protocol` reached either through a module alias
                # (_app_ctx.ctx.protocol) or a local name (ctx.protocol).
                named_ctx = (isinstance(base, ast.Attribute) and base.attr == 'ctx') or (
                    isinstance(base, ast.Name) and base.id == 'ctx'
                )
                if named_ctx:
                    offenders.append(f'{rel_path}:{node.lineno}')

        assert not offenders, 'app_context.protocol is referenced again: ' + ', '.join(offenders)

    def test_the_field_is_gone_from_the_context(self):
        """The declaration itself, not just its uses."""
        tree = parse_module('modules/app_context.py')
        declared = {
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

        assert 'protocol' not in declared, (
            'AppContext declares a protocol field again; the panel owns the live protocol'
        )
