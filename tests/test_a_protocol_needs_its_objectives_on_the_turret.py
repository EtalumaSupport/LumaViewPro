# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol may only name glass this scope can put in the light path.

A protocol names an objective per step; the scope can reach some set of
objectives and no others. When the two disagree the run does not fail --
it COMPLETES, and every file it writes takes its filename from the step's
objective and its metadata from the glass actually in the path. The file
lies about itself, and every scale derived from it follows the metadata,
so the name is the only thing that is wrong and nothing in the output
says so.

ONE rule, asked by every moment that admits a protocol:

- With a turret, an objective is addressable when a slot is assigned to
  it. An unassigned turret addresses NOTHING, so it refuses everything.
- Without a turret, the objective in the light path is whichever one is
  mounted, so a protocol may name only one.

The unassigned turret refusing a single-objective protocol is the part
that changed, and it is deliberate. The gate used to exempt that case to
keep a fresh install runnable, on the premise that the first protocol a
user makes names glass no slot holds. The premise was false: the startup
objective question assigns the current position before any protocol
loads, and its popup cannot be dismissed. A scope arriving here with four
empty slots is one whose slots were CLEARED, and it needs to be sent back
to the turret screen rather than allowed to photograph through whatever
happens to be mounted.

The table below is that rule, every cell written out. The cases beneath
it are the same rule reaching a caller through prepare().
"""

from __future__ import annotations

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.protocol_drives import autofocus_snapshot
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_autogain_settings,
    _make_image_capture_config,
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)


ON_TURRET = '10x Oly'
NOT_ON_TURRET = '20x Oly'

NOT_CARRIED = 'turret_objectives_unassigned'
NEEDS_TURRET = 'objectives_require_turret'


def _protocol(*objectives: str):
    """A protocol naming exactly the objectives given, one step each."""
    return _make_multi_step_protocol(
        [
            {'objective': objective, 'name': f'step_{i}_{objective}'}
            for i, objective in enumerate(objectives)
        ]
    )


def _turret(scope, monkeypatch, *, has_turret=True, carries=(ON_TURRET,)):
    """Point the scope's turret state at a known set of objectives.

    ScopeCapabilities is a frozen dataclass, so the capability is swapped
    by replacing the whole record rather than assigning into it -- the
    immutability is the point of that type and the test respects it.
    """
    import dataclasses

    monkeypatch.setattr(
        scope, 'capabilities', dataclasses.replace(scope.capabilities, has_turret=has_turret)
    )
    monkeypatch.setattr(
        scope.runtime_state,
        'get_turret_config',
        lambda: dict(enumerate(carries, start=1)),
    )


def _prepare(executor, protocol, tmp_path):
    return executor.prepare(
        protocol=protocol,
        run_trigger_source='scan',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='turret_objectives',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        autofocus_snapshot=autofocus_snapshot(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks={},
    )


# Every cell of the rule: whether the scope has a turret, what the turret
# carries, what the protocol names, and the answer. None is admitted; a
# string is the reason code the refusal must carry.
#
# Written out rather than computed, because a table that derives its own
# expectations from the same predicate the code uses cannot disagree with
# the code, and so pins nothing.
RULE_TABLE = [
    # A turret with nothing assigned addresses nothing.
    (True, (), (ON_TURRET,), NOT_CARRIED),
    (True, (), (NOT_ON_TURRET,), NOT_CARRIED),
    (True, (), (ON_TURRET, NOT_ON_TURRET), NOT_CARRIED),
    # A turret carrying one objective serves exactly that one.
    (True, (ON_TURRET,), (ON_TURRET,), None),
    (True, (ON_TURRET,), (NOT_ON_TURRET,), NOT_CARRIED),
    (True, (ON_TURRET,), (ON_TURRET, NOT_ON_TURRET), NOT_CARRIED),
    # A turret carrying both serves any combination of them.
    (True, (ON_TURRET, NOT_ON_TURRET), (ON_TURRET,), None),
    (True, (ON_TURRET, NOT_ON_TURRET), (NOT_ON_TURRET,), None),
    (True, (ON_TURRET, NOT_ON_TURRET), (ON_TURRET, NOT_ON_TURRET), None),
    # No turret: the mounted objective is whatever it is, so the count is
    # the only question and the turret's contents never enter it.
    (False, (), (ON_TURRET,), None),
    (False, (), (NOT_ON_TURRET,), None),
    (False, (), (ON_TURRET, NOT_ON_TURRET), NEEDS_TURRET),
    (False, (ON_TURRET,), (ON_TURRET,), None),
    (False, (ON_TURRET,), (NOT_ON_TURRET,), None),
    (False, (ON_TURRET,), (ON_TURRET, NOT_ON_TURRET), NEEDS_TURRET),
    (False, (ON_TURRET, NOT_ON_TURRET), (ON_TURRET,), None),
    (False, (ON_TURRET, NOT_ON_TURRET), (NOT_ON_TURRET,), None),
    (False, (ON_TURRET, NOT_ON_TURRET), (ON_TURRET, NOT_ON_TURRET), NEEDS_TURRET),
]


class TestTheRuleItself:
    """The rule through its owner, with no run and no protocol involved."""

    @pytest.mark.parametrize('has_turret, carries, names, expected', RULE_TABLE)
    def test_every_cell_of_the_rule(self, scope, monkeypatch, has_turret, carries, names, expected):
        _turret(scope, monkeypatch, has_turret=has_turret, carries=carries)

        if expected is None:
            scope.protocols.refuse_unaddressable_objectives(names)
            return

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.refuse_unaddressable_objectives(names)

        assert refusal.value.reason == expected, (
            f'has_turret={has_turret} carries={carries} names={names}: '
            f'expected {expected}, got {refusal.value.reason}'
        )

    def test_a_protocol_naming_nothing_is_not_this_rules_problem(self, scope, monkeypatch):
        """An empty protocol addresses no glass, so no glass is missing.

        The empty-protocol refusal owns that case and names it properly;
        answering it here would send the user to the turret screen to fix
        a protocol with no steps in it.
        """
        _turret(scope, monkeypatch, carries=())

        scope.protocols.refuse_unaddressable_objectives([])

    def test_the_turretless_refusal_does_not_send_the_user_to_a_turret(self, scope, monkeypatch):
        """A scope with no turret has no turret screen to be sent to."""
        _turret(scope, monkeypatch, has_turret=False, carries=())

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.refuse_unaddressable_objectives([ON_TURRET, NOT_ON_TURRET])

        assert 'turret' not in refusal.value.message.lower(), (
            f'the message names hardware this scope lacks: {refusal.value.message!r}'
        )

    def test_a_refusal_says_what_the_turret_carries(self, scope, monkeypatch):
        """A user told only 'not assigned' has to go and look."""
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.refuse_unaddressable_objectives([NOT_ON_TURRET])

        assert ON_TURRET in refusal.value.message, (
            f'the refusal must say what IS on the turret: {refusal.value.message!r}'
        )
        assert NOT_ON_TURRET in refusal.value.message, (
            f'the refusal must say what was asked for: {refusal.value.message!r}'
        )

    def test_an_unrenderable_objective_id_does_not_raise_out_of_the_boundary(
        self, scope, monkeypatch
    ):
        """A degraded id must still produce the refusal, not a TypeError.

        Objective validation does not run on the load path, so a blank or
        non-string cell from a hand-edited file reaches this rule. Sorting
        a mixed set of types to build the message would raise TypeError
        out of the API boundary, and a caller expecting a refusal would
        get a crash instead.
        """
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            scope.protocols.refuse_unaddressable_objectives([float('nan'), ''])

        assert refusal.value.reason == NOT_CARRIED


class TestTheEngineRefuses:
    def test_an_objective_the_turret_does_not_carry_is_refused(
        self, executor, scope, monkeypatch, tmp_path
    ):
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == NOT_CARRIED, (
            'the reason code is what a REST or SDK caller branches on'
        )

    def test_one_objective_the_turret_does_not_carry_is_refused_too(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """The defect: this run used to complete and write a lying filename.

        A turret carrying only 10x and a protocol naming only 20x was
        admitted, because the gate exempted anything with a single
        objective. The run moved nothing, captured through the 10x that
        was already in the light path, and named the files for the 20x
        the protocol asked for.
        """
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == NOT_CARRIED

    def test_a_multi_objective_protocol_on_an_unassigned_turret_is_still_refused(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """An empty turret cannot serve a protocol that changes objectives."""
        _turret(scope, monkeypatch, carries=())

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == NOT_CARRIED

    def test_a_single_objective_protocol_on_an_unassigned_turret_is_refused(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """The case the old gate exempted, and the reason it no longer does.

        The exemption existed to keep a fresh install runnable. It is the
        startup objective question that does that -- it assigns the
        current position before any protocol loads -- so the exemption
        was protecting a state that does not occur, while admitting every
        run on a turret someone had cleared.
        """
        _turret(scope, monkeypatch, carries=())

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == NOT_CARRIED

    def test_a_multi_objective_protocol_on_a_turretless_scope_is_refused(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """Nothing can change the objective between steps, so nothing may ask."""
        _turret(scope, monkeypatch, has_turret=False, carries=())

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == NEEDS_TURRET

    def test_a_protocol_whose_objectives_are_all_assigned_runs(
        self, executor, scope, monkeypatch, tmp_path
    ):
        _turret(scope, monkeypatch, carries=(ON_TURRET, NOT_ON_TURRET))

        plan = _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')


class TestAnObjectiveThatDoesNotExistIsNotATurretProblem:
    """The rule must only judge objectives that ARE objectives.

    A protocol saved on an install with a different objectives.json names
    glass this catalogue has never heard of; a hand-edited or legacy file
    can leave the Objective cell blank, which loads as the empty string.
    Neither is an addressability problem, and answering them with the
    turret's message tells the user to go and mount something that does
    not exist. Validation owns them, and it has a message that names the
    real defect -- which is why prepare() asks the rule BELOW validation.
    """

    def test_an_unknown_objective_refuses_as_a_validation_failure(
        self, executor, scope, monkeypatch, tmp_path
    ):
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol('zzz-no-such-objective'), tmp_path)

        assert refusal.value.reason == 'validation_failed', (
            'an objective outside the catalogue is a validation defect, not an '
            f'addressability one: {refusal.value.message!r}'
        )

    def test_a_blank_objective_refuses_as_a_validation_failure(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """A blank Objective cell in a protocol file loads as ''."""
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(''), tmp_path)

        assert refusal.value.reason == 'validation_failed'


class TestTheCasesThatMustNotChange:
    def test_a_single_objective_protocol_runs_when_the_turret_carries_it(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """The common case: one objective, mounted. Nothing refuses it."""
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        plan = _prepare(executor, _protocol(ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')

    def test_a_single_objective_protocol_runs_on_a_turretless_scope(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """One objective and no turret: nothing needs to change, so nothing does."""
        _turret(scope, monkeypatch, has_turret=False, carries=())

        plan = _prepare(executor, _protocol(ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')
