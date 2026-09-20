# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol needs the glass it names on the turret.

A protocol names an objective per step; the turret carries objectives in
its slots. When the two disagree the run does not fail -- it COMPLETES,
and every file it writes takes its filename from the step's objective and
its metadata from the turret's. The file lies about itself, and every
scale derived from it follows the metadata, so the name is the only thing
that is wrong and nothing in the output says so.

The check for that lived in a Kivy widget method, which meant a script,
the SDK and REST could all start the run the GUI would have refused. It
now refuses at the preparation chokepoint, where every caller meets it.

Two rules, not one:

- A turret that CARRIES something refuses a protocol naming anything it
  does not carry, whatever the step count.
- A turret with NOTHING assigned is the shipped first-run state: all four
  slots ship null. It still refuses a protocol that changes objectives
  mid-run, which cannot address a turret it has no assignments for; it
  does not refuse a single-objective protocol, or a fresh install could
  not run the shipped example.

The cases below are that contract, one test per configuration.
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


class TestTheEngineRefuses:
    def test_an_objective_the_turret_does_not_carry_is_refused(
        self, executor, scope, monkeypatch, tmp_path
    ):
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == 'turret_objectives_unassigned', (
            'the reason code is what a REST or SDK caller branches on'
        )

    def test_the_refusal_names_what_the_turret_carries(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """A user who is told only 'not assigned' has to go and look.

        The widget's message listed the assigned objectives, and that is
        the half a caller cannot reconstruct from the reason code.
        """
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert ON_TURRET in refusal.value.message, (
            f'the refusal must say what IS on the turret: {refusal.value.message!r}'
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

        assert refusal.value.reason == 'turret_objectives_unassigned'

    def test_a_multi_objective_protocol_on_an_unassigned_turret_is_still_refused(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """An empty turret cannot serve a protocol that changes objectives.

        The simpler single-rule predicate -- refuse only on a mismatch
        against what is actually assigned -- would let this one start,
        and it cannot finish: there is no slot to rotate to for either
        objective. It keeps refusing.
        """
        _turret(scope, monkeypatch, carries=())

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert refusal.value.reason == 'turret_objectives_unassigned'

    def test_a_protocol_whose_objectives_are_all_assigned_runs(
        self, executor, scope, monkeypatch, tmp_path
    ):
        _turret(scope, monkeypatch, carries=(ON_TURRET, NOT_ON_TURRET))

        plan = _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')


class TestTheCasesThatMustNotChange:
    def test_a_single_objective_protocol_runs_on_a_turret_with_no_assignments(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """The shipped first-run state, and the reason rule two exists.

        data/settings.json ships all four turret slots null against a
        96-step single-objective example protocol. A gate that refused
        on "not carried" alone would refuse that, so a fresh install
        could not run the protocol it came with.
        """
        _turret(scope, monkeypatch, carries=())

        plan = _prepare(executor, _protocol(NOT_ON_TURRET), tmp_path)

        assert plan is not None, 'an unassigned turret is a fresh install, not a mismatch'
        executor.reset(requester='scan')

    def test_a_single_objective_protocol_runs_when_the_turret_carries_it(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """The common case: one objective, mounted. Nothing refuses it."""
        _turret(scope, monkeypatch, carries=(ON_TURRET,))

        plan = _prepare(executor, _protocol(ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')

    def test_a_scope_with_no_turret_is_not_checked(self, executor, scope, monkeypatch, tmp_path):
        """No turret means nothing to assign objectives to."""
        _turret(scope, monkeypatch, has_turret=False, carries=())

        plan = _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')
