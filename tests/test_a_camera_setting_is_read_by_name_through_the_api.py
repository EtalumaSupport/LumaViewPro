# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scope.diagnostics.read_camera_node`` reads one camera setting by the camera's own name, or refuses where it is asked.

A camera with no named settings -- the simulated camera, a Classic's FX2 --
is refused before anything reaches it (``camera_node_unsupported``), as is a
map the camera does not have or a name that is not one, and a scope with no
camera is refused naming the camera. A camera that answers gives its value
and unit, None for a name its map lacks, and a node that holds no value is
refused with its kind as the reason. The same holds over REST.

Which node is what is the driver's to read (``tests/test_a_camera_driver_reads_a_setting_by_its_name.py``);
here the simulated camera's reader stands in for one that has named
settings, so the API's half is proven on the real scope and lane.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _lumascope.drivers.camera import CameraNodeHasNoValue, CameraNodeValue
from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    CameraNodeNotReadableError,
    CameraSettingUnsupportedError,
    HardwareCommandRefusedError,
    MissingPart,
)
from _lumascope.modules.scope_parts import Part
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings

_ROUTE = '/api/v1/scope/diagnostics/read_camera_node'

# What the stand-in camera's maps hold, by map and name.
_NODES = {
    ('device', 'DeviceTemperature'): CameraNodeValue(41.5, 'C'),
    ('device', 'PixelFormat'): CameraNodeValue('Mono12', ''),
    ('device', 'AcquisitionStart'): CameraNodeHasNoValue.COMMAND,
    ('device', 'Root'): CameraNodeHasNoValue.CATEGORY,
    ('device', 'LUTValueAll'): CameraNodeHasNoValue.REGISTER,
    ('device', 'Device'): CameraNodeHasNoValue.PORT,
    ('device', 'EnumEntry_GainAuto_Off'): CameraNodeHasNoValue.ENUM_ENTRY,
    ('device', 'AcquisitionFrameRate'): CameraNodeHasNoValue.NOT_READABLE_NOW,
    ('stream', 'Statistic_Total_Buffer_Count'): CameraNodeValue(120, ''),
}


def _session(model='LS850', **kwargs) -> ScopeSession:
    return ScopeSession.create(
        complete_settings(microscope=model), simulate=True, warn_pre_release=False, **kwargs
    )


@pytest.fixture
def named(monkeypatch):
    """A session whose simulated camera has named settings, and the reads that reached it."""
    reached = []

    def read_node(self, name, node_map):
        reached.append((node_map, name))
        answer = _NODES.get((node_map, name))
        if name == 'DeviceTemperatureUnanswered':
            raise CameraReadFailedError('The camera did not answer a read of it')
        return answer

    # a stand-in by design: the simulated camera has no named settings (Eric's
    # ruling of 2026-11-08: the reader is proven on the Pylon driver against
    # Basler's emulator), so its reader is replaced to hand the API each
    # answer a camera's reader gives.
    monkeypatch.setattr(SimulatedCamera, 'node_maps', lambda self: ('device', 'stream'))
    monkeypatch.setattr(SimulatedCamera, 'read_node', read_node)
    session = _session()
    yield session, reached
    session.shutdown()


def test_a_camera_with_no_named_settings_is_refused_before_anything_reaches_it():
    for model in ('LS850', 'LS620'):  # the simulated camera; a Classic's simulated FX2
        session = _session(model)
        try:
            assert session.scope.capabilities.camera_node_maps == (), model
            with pytest.raises(CameraSettingUnsupportedError) as refused:
                session.scope.diagnostics.read_camera_node('Gain')
            assert refused.value.reason == 'camera_node_unsupported', model
        finally:
            session.shutdown()


def test_a_scope_with_no_camera_is_refused_naming_the_camera():
    session = _session(simulated=SimulatedScope(camera_present=False))
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.scope.diagnostics.read_camera_node('Gain')
    finally:
        session.shutdown()

    assert (refused.value.reason, refused.value.missing) == (
        'not_connected',
        MissingPart(Part.CAMERA),
    )


def test_a_setting_the_camera_has_answers_its_value_and_unit_on_its_map(named):
    session, _reached = named
    diagnostics = session.scope.diagnostics

    assert session.scope.capabilities.camera_node_maps == ('device', 'stream')
    assert diagnostics.read_camera_node('DeviceTemperature') == CameraNodeValue(41.5, 'C')
    assert diagnostics.read_camera_node('PixelFormat') == CameraNodeValue('Mono12', '')
    assert diagnostics.read_camera_node(
        'Statistic_Total_Buffer_Count', node_map='stream'
    ) == CameraNodeValue(120, '')


def test_a_name_the_map_lacks_answers_none(named):
    session, _reached = named

    assert session.scope.diagnostics.read_camera_node('BslChunkTimestampSelector') is None


def test_a_node_with_no_value_is_refused_with_its_kind_as_the_reason(named):
    session, _reached = named
    kinds = {
        name: kind
        for (_map, name), kind in _NODES.items()
        if isinstance(kind, CameraNodeHasNoValue)
    }

    reasons = {}
    for name in kinds:
        with pytest.raises(CameraNodeNotReadableError) as refused:
            session.scope.diagnostics.read_camera_node(name)
        reasons[name] = refused.value.reason

    assert reasons == {name: str(kind) for name, kind in kinds.items()}
    assert set(kinds.values()) == set(CameraNodeHasNoValue)


def test_a_read_the_camera_does_not_answer_raises_its_failure(named):
    session, _reached = named

    with pytest.raises(CameraReadFailedError) as raised:
        session.scope.diagnostics.read_camera_node('DeviceTemperatureUnanswered')

    assert raised.value.reason == 'camera_read_failed'


def test_a_map_or_name_the_camera_cannot_take_is_refused_before_anything_reaches_it(named):
    session, reached = named
    diagnostics = session.scope.diagnostics

    with pytest.raises(ArgumentRefusedError) as wrong_map:
        diagnostics.read_camera_node('Gain', node_map='instant_camera')
    names = {}
    for name in ('', '9Gain', 'Gain Raw', 'Gain;'):
        with pytest.raises(ArgumentRefusedError) as wrong_name:
            diagnostics.read_camera_node(name)
        names[name] = wrong_name.value.reason

    assert (wrong_map.value.reason, wrong_map.value.offered) == (
        'not_offered',
        ('device', 'stream'),
    )
    assert set(names.values()) == {'wrong_kind'}, names
    assert reached == []


def test_rest_reads_a_setting_and_states_each_refusal(named):
    session, _reached = named

    with TestClient(build_app(session)) as client:
        value = client.get(_ROUTE, params={'name': 'DeviceTemperature'})
        absent = client.get(_ROUTE, params={'name': 'BslChunkTimestampSelector'})
        command = client.get(_ROUTE, params={'name': 'AcquisitionStart'})
        not_now = client.get(_ROUTE, params={'name': 'AcquisitionFrameRate'})
        wrong_map = client.get(_ROUTE, params={'name': 'Gain', 'node_map': 'event'})

    assert (value.status_code, value.json()) == (200, {'value': 41.5, 'unit': 'C'})
    assert (absent.status_code, absent.json()) == (200, None)
    assert (command.status_code, command.json()['reason']) == (422, 'command')
    assert (not_now.status_code, not_now.json()['reason']) == (409, 'not_readable_now')
    assert (wrong_map.status_code, wrong_map.json()['reason']) == (422, 'not_offered')


def test_rest_refuses_a_camera_with_no_named_settings():
    session = _session()
    try:
        with TestClient(build_app(session)) as client:
            answer = client.get(_ROUTE, params={'name': 'Gain'})
    finally:
        session.shutdown()

    assert (answer.status_code, answer.json()['reason']) == (422, 'camera_node_unsupported')
