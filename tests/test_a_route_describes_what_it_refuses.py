# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A route's description says what it refuses and fails with, from its member's ``Raises:`` section.

Each class an entry names, with the reasons it quotes, is described under
the status each reason answers: a refusal's cause decides 422 or 409 per
reason, a fault is 500. An entry that quotes no reason is described with
every reason its class can carry. Each status carries the classes and
reasons as ``x-refusals``; the ``4XX`` and ``5XX`` ranges stay for what no
section declares.
"""

from __future__ import annotations

import gc
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from lib.lumascope_error import Fault
from modules.exceptions import HardwareCommandRefusedError, ProtocolRunRefusedError
from modules.scope_session import ScopeSession
from rest import problems
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture(scope='module')
def described(tmp_path_factory):
    live = tmp_path_factory.mktemp('live')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        with TestClient(build_app(session)) as client:
            yield client.get('/api/v1/openapi.json').json()
    finally:
        session.shutdown()


def _refusals(described: dict, path: str) -> dict[str, dict[str, list[str]]]:
    responses = described['paths'][path]['post']['responses']
    return {status: r['x-refusals'] for status, r in responses.items() if 'x-refusals' in r}


def _status(cls: type, reason: str) -> str:
    return str(problems._CAUSE_STATUS[cls.causes[reason]])


def test_each_reason_an_entry_quotes_is_described_under_the_status_it_answers(described):
    refusals = _refusals(described, '/api/v1/scope/motion/move_xy_by')

    for reason in ('not_connected', 'axis_absent', 'position_unread', 'lid_open_before_motion'):
        status = _status(HardwareCommandRefusedError, reason)
        assert reason in refusals[status]['HardwareCommandRefusedError'], (reason, status)
    assert {_status(HardwareCommandRefusedError, r) for r in ('axis_absent', 'not_connected')} == {
        '422',
        '409',
    }
    assert refusals['500']['MoveFaultError'] == [
        'driver_failed',
        'stalled',
        'board_lost',
        'position_unread_after_move',
        'status_unread',
        'monitor_ended_mid_move',
        'faulted',
    ]


def test_an_entry_that_quotes_no_reason_is_described_with_every_reason_of_its_class(described):
    refusals = _refusals(described, '/api/v1/handles/ProtocolRunner/{handle_id}/start_single_scan')

    said = {
        reason: status
        for status, classes in refusals.items()
        for reason in classes.get('ProtocolRunRefusedError', [])
    }
    assert said == {r: _status(ProtocolRunRefusedError, r) for r in ProtocolRunRefusedError.causes}


def test_the_ranges_stay_for_what_no_section_declares(described):
    responses = described['paths']['/api/v1/scope/motion/move_xy_by']['post']['responses']

    assert {'4XX', '5XX'} <= set(responses)


def test_a_server_refusal_is_described_under_the_status_the_server_answers():
    """The server's own refusal answers its reason's status, not its cause's: ``not_found`` is 404, not 422."""
    doc = (
        "Forget a handle.\n\nRaises:\n    ServerRefusedError: ``'not_found'``; ``'overloaded'``.\n"
    )

    described = problems.declared(doc, problems.outcome_classes())

    assert {status: d['x-refusals'] for status, d in described.items()} == {
        404: {'ServerRefusedError': ['not_found']},
        503: {'ServerRefusedError': ['overloaded']},
    }


def test_two_outcome_classes_of_one_name_fail_the_build():
    """A ``Raises:`` entry names its class by name, so a second class of that name is refused."""

    def twin(reason: str) -> type:
        class TwinError(Fault, Exception):
            causes: ClassVar[dict[str, None]] = {reason: None}

        return TwinError

    pair = (twin('zz_twin_one'), twin('zz_twin_two'))
    try:
        with pytest.raises(TypeError, match='TwinError'):
            problems.outcome_classes()
    finally:
        # Held by weak reference in __subclasses__: gone once dropped.
        del pair
        gc.collect()
    assert 'TwinError' not in problems.outcome_classes()
