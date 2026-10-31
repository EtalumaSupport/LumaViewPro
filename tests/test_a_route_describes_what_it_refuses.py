# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A route's description says what it refuses and fails with, from its member's ``Raises:`` section and the doors it enters.

Each class an entry names, with the reasons it quotes, is described under
the status each reason answers: a refusal's cause decides 422 or 409 per
reason, a fault is 500. An entry that quotes no reason is described with
every reason its class can carry. A member that enters a lane asking the
scope's activity claim, or a guarded write, adds that door's own entries
(``@api(admitted_by=...)``), so a route that a held control session
refuses says so. Each status carries the classes and reasons as
``x-refusals``; the ``4XX`` and ``5XX`` ranges stay for what no section
declares.
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


def _refusals(described: dict, path: str, method: str = 'post') -> dict[str, dict[str, list[str]]]:
    responses = described['paths'][path][method]['responses']
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
        '503',
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


def test_a_read_that_enters_a_lane_names_the_lanes_refusals_and_no_other(described):
    """``get_camera_temperatures_degc`` reads through the CAMERA lane; its own section names no lane refusal."""
    refusals = _refusals(
        described, '/api/v1/scope/diagnostics/get_camera_temperatures_degc', method='get'
    )

    assert refusals['409']['ControlSessionRefusedError'] == ['control_session_held']
    assert refusals['409']['HardwareCommandRefusedError'] == ['exclusive_activity_running']
    assert refusals['422']['HardwareCommandRefusedError'] == ['activity_ended']
    assert refusals['503']['HardwareCommandRefusedError'] == ['scope_disconnected']


def test_a_guarded_write_names_the_held_session_beside_its_own_refusals(described):
    refusals = _refusals(described, '/api/v1/delete_step')

    assert refusals == {
        '409': {'ControlSessionRefusedError': ['control_session_held']},
        '422': {'StepNotFoundError': ['no_such_step']},
    }


def test_the_remedy_door_is_the_one_route_to_a_remedy_and_names_the_held_session(described):
    """Each remedy is a guarded write reached only through ``apply_remedy``, so its
    route says a held control session refuses it, and no remedy has a route of its own."""
    refusals = _refusals(described, '/api/v1/apply_remedy')

    assert refusals['409']['ControlSessionRefusedError'] == ['control_session_held']
    remedy_routes = {'/api/v1/recover_file_writer', '/api/v1/repair_camera_driver'}
    assert remedy_routes.isdisjoint(described['paths'])


def test_a_stage_position_is_read_over_the_wire_through_axis_positions_alone(described):
    """``axis_positions`` is the position read a client has; the board's own read,
    ``get_actual_position``, is the bench's and has no route."""
    assert '/api/v1/scope/motion/axis_positions' in described['paths']
    assert '/api/v1/scope/motion/get_actual_position' not in described['paths']


def test_a_build_on_the_post_processing_lane_names_no_door(described):
    """The post-processing lane asks no claim, so a build is open to anyone, held session or not."""
    refusals = _refusals(described, '/api/v1/post_processing/build')

    named = {cls for classes in refusals.values() for cls in classes}
    assert named.isdisjoint({'ControlSessionRefusedError', 'HardwareCommandRefusedError'}), named


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
