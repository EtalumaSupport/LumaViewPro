# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A multi-objective protocol needs every objective it names on the turret.

A protocol that switches objectives mid-run can only run if the turret
carries each one. The check for that lived in a Kivy widget method, which
meant a script, the SDK and REST could all start the run the GUI would
have refused -- and the run then failed partway through, after it had
moved the stage and taken images.

The predicate was never the problem: it already read the API's runtime
turret state rather than anything of the widget's own. Only its HOME was
wrong. It now refuses at the preparation chokepoint, where every caller
meets it.

Unchanged on purpose: a SINGLE-objective protocol is not checked at all.
That is legacy behaviour, and widening it here would be an undeclared
change riding along with a relocation -- a protocol with one objective
the turret does not carry still starts today, exactly as it did before.
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

    def test_a_protocol_whose_objectives_are_all_assigned_runs(
        self, executor, scope, monkeypatch, tmp_path
    ):
        _turret(scope, monkeypatch, carries=(ON_TURRET, NOT_ON_TURRET))

        plan = _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')


class TestTheCasesThatMustNotChange:
    def test_a_single_objective_protocol_is_not_checked(
        self, executor, scope, monkeypatch, tmp_path
    ):
        """Legacy, and carried deliberately rather than quietly widened."""
        _turret(scope, monkeypatch, carries=())

        plan = _prepare(executor, _protocol(NOT_ON_TURRET), tmp_path)

        assert plan is not None, 'one objective has never been validated against the turret'
        executor.reset(requester='scan')

    def test_a_scope_with_no_turret_is_not_checked(self, executor, scope, monkeypatch, tmp_path):
        """No turret means nothing to assign objectives to."""
        _turret(scope, monkeypatch, has_turret=False, carries=())

        plan = _prepare(executor, _protocol(ON_TURRET, NOT_ON_TURRET), tmp_path)

        assert plan is not None
        executor.reset(requester='scan')
