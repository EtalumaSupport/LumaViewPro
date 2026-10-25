# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A value outside its closed set is refused where it enters, in one word: ``not_offered``.

A closed set is a ``StrEnum``. Wherever one of its values enters -- a
member's parameter, a field of a record a caller builds (or an item or key
of one that holds several), an item of a typed container -- the member or
its plain string is taken and the body
gets the member, and anything else that is a string is refused
``ArgumentRefusedError`` ``'not_offered'``, naming the argument and offering
the set's values. Over REST the same judge runs: the wire hands the string
through, so a non-member is a 422 ``not_offered`` problem, not a request
pydantic refused. A value that is not a string at all stays the door's
``wrong_argument_type``.
"""

import dataclasses

import pytest
from fastapi.testclient import TestClient

from modules import wire_encoding
from modules.api_surface import api, api_fields
from modules.axes import StageAxis
from modules.exceptions import ArgumentRefusedError, ArgumentTypeRefusedError
from modules.lumascope_api.camera import CameraAutoMode
from modules.lumascope_api.illumination import LedReading, LedSnapshot, LedState
from modules.scope_session import ScopeSession
from rest.app import build_app
from rest.routes import routes
from tests.settings_fixtures import complete_settings

OFFERED = ('Off', 'Once', 'Continuous')


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def client(session):
    with TestClient(build_app(session)) as client:
        yield client


def _not_offered(call, argument, offered):
    with pytest.raises(ArgumentRefusedError) as refused:
        call()
    assert refused.value.reason == 'not_offered'
    assert refused.value.argument == argument
    assert refused.value.offered == offered
    return refused.value


def test_a_members_plain_string_is_taken_as_the_member(session):
    camera = session.camera
    camera.set_auto_exposure(CameraAutoMode.CONTINUOUS)

    camera.set_auto_exposure('Off')

    assert camera.read_auto_exposure() is CameraAutoMode.OFF


def test_a_string_outside_the_set_is_refused_offering_the_set(session):
    camera = session.camera
    camera.set_auto_exposure(CameraAutoMode.CONTINUOUS)

    _not_offered(lambda: camera.set_auto_exposure('Sometimes'), 'mode', OFFERED)

    assert camera.read_auto_exposure() is CameraAutoMode.CONTINUOUS


def test_a_value_that_is_no_string_is_refused_as_the_wrong_type(session):
    with pytest.raises(ArgumentTypeRefusedError) as refused:
        session.camera.set_auto_exposure(3)
    assert refused.value.reason == 'wrong_argument_type'


# A marked function of the test's own: no member takes an optional closed
# set or a container of one yet, and the door is the subject.
@api
def _optional(mode: CameraAutoMode | None) -> CameraAutoMode | None:
    return mode


@api
def _several(modes: tuple[CameraAutoMode, ...]) -> tuple[CameraAutoMode, ...]:
    return modes


def test_an_optional_closed_set_takes_none_and_refuses_a_non_string_by_type():
    assert _optional(None) is None
    assert _optional('Once') is CameraAutoMode.ONCE
    _not_offered(lambda: _optional('Twice'), 'mode', OFFERED)
    with pytest.raises(ArgumentTypeRefusedError):
        _optional(3)


def test_a_container_of_a_closed_set_takes_plain_strings_and_refuses_naming_the_bad_item():
    taken = _several(('Off', CameraAutoMode.ONCE))
    assert taken == (CameraAutoMode.OFF, CameraAutoMode.ONCE)
    assert all(type(m) is CameraAutoMode for m in taken)
    refused = _not_offered(lambda: _several(('Off', 'Twice')), 'modes[1]', OFFERED)
    assert refused.value == 'Twice'
    for wrong_type in ('Off', ('Off', 3)):
        with pytest.raises(ArgumentTypeRefusedError):
            _several(wrong_type)


def test_a_record_built_with_a_plain_string_holds_the_member():
    reading = LedReading(2, 'Red', 'lit', 10.0)
    assert reading.state is LedState.LIT


def test_a_record_built_with_a_string_outside_the_set_is_refused_naming_its_field():
    _not_offered(
        lambda: LedReading(2, 'Red', 'glowing', None),
        'LedReading.state',
        tuple(m.value for m in LedState),
    )


@api_fields('modes', 'by_mode')
@dataclasses.dataclass(frozen=True)
class _Held:
    """A record holding a closed set's members in a tuple and as a mapping's keys."""

    modes: tuple[CameraAutoMode, ...]
    by_mode: dict[CameraAutoMode, int]


def test_a_records_items_and_keys_built_with_plain_strings_hold_the_members():
    held = _Held(('Off', CameraAutoMode.ONCE), {'Continuous': 1})

    assert held.modes == (CameraAutoMode.OFF, CameraAutoMode.ONCE)
    assert [type(m) for m in held.modes] == [CameraAutoMode, CameraAutoMode]
    assert type(held.modes) is tuple
    assert [type(k) for k in held.by_mode] == [CameraAutoMode]
    assert held.by_mode == {CameraAutoMode.CONTINUOUS: 1}


def test_a_records_item_or_key_outside_the_set_is_refused_naming_its_place():
    _not_offered(lambda: _Held(('Off', 'Twice'), {}), '_Held.modes[1]', OFFERED)
    _not_offered(lambda: _Held((), {'Twice': 1}), "_Held.by_mode['Twice']", OFFERED)


def test_a_snapshot_a_caller_built_with_plain_strings_restores_its_lit_address(session):
    illumination = session.scope.illumination
    illumination.leds_off()
    snapshot = LedSnapshot(
        'by hand',
        tuple(
            LedReading(
                r.address,
                r.name,
                'lit' if r.name == 'Red' else 'dark',
                10.0 if r.name == 'Red' else None,
            )
            for r in illumination.get_led_states()
        ),
    )

    illumination.restore_led_state(snapshot)

    assert illumination.get_led_state('Red') == LedReading(2, 'Red', LedState.LIT, 10.0)


def test_over_rest_a_plain_string_is_taken_and_a_non_member_is_a_422_not_offered(client, session):
    route = '/api/v1/scope/camera/auto_exposure'
    assert client.put(route, json={'mode': 'Continuous'}).status_code == 200
    assert session.camera.read_auto_exposure() is CameraAutoMode.CONTINUOUS

    refused = client.put(route, json={'mode': 'Sometimes'})

    assert refused.status_code == 422
    body = refused.json()
    assert (body['reason'], body['argument'], body['offered']) == (
        'not_offered',
        'mode',
        list(OFFERED),
    )
    assert session.camera.read_auto_exposure() is CameraAutoMode.CONTINUOUS


def test_over_rest_a_records_field_outside_its_set_is_a_422_naming_the_field(client, session):
    snapshot = client.get(
        '/api/v1/scope/illumination/save_led_state', params={'tag': 'before'}
    ).json()
    snapshot['readings'][0].update(state='glowing', illumination_ma=None)

    refused = client.post(
        '/api/v1/scope/illumination/restore_led_state', json={'snapshot': snapshot}
    )

    assert refused.status_code == 422
    assert (refused.json()['reason'], refused.json()['argument']) == (
        'not_offered',
        'LedReading.state',
    )


def test_the_openapi_description_offers_the_sets_values_for_a_parameter(client):
    document = client.get('/api/v1/openapi.json').json()
    put = document['paths']['/api/v1/scope/camera/auto_exposure']['put']
    body = put['requestBody']['content']['application/json']['schema']
    model = document['components']['schemas'][body['$ref'].rsplit('/', 1)[-1]]
    assert model['properties']['mode']['enum'] == list(OFFERED)


def _closed_sets_in(alternatives) -> set[type]:
    """The closed sets a parameter takes: alone, optional, or as an array's or object's items."""
    found = set()
    for alternative in alternatives:
        if alternative.form == wire_encoding.ENUM:
            found.add(alternative.cls)
        found |= _closed_sets_in(alternative.items)
    return found


def _enums_in(schema: object) -> list[list[str]]:
    """Every ``enum`` a JSON schema fragment declares, however deeply nested."""
    if isinstance(schema, dict):
        own = [schema['enum']] if 'enum' in schema else []
        return own + [e for value in schema.values() for e in _enums_in(value)]
    if isinstance(schema, list):
        return [e for value in schema for e in _enums_in(value)]
    return []


def test_every_route_taking_a_closed_set_describes_the_sets_values():
    """Each parameter a wire client sends that takes a closed set is described
    with exactly that set's values, so a client reads the words it may send
    from the API's description; the door that judges them is the one
    ``test_over_rest_a_plain_string_is_taken_and_a_non_member_is_a_422_not_offered``
    pins."""
    session_routes = routes(
        ScopeSession,
        handed_out=wire_encoding.handed_out(
            ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
        ),
    )
    seen, undescribed = set(), []
    for route in session_routes:
        if route.body is None:
            continue
        properties = route.body.model_json_schema().get('properties', {})
        for parameter in route.member.parameters:
            for cls in _closed_sets_in(parameter.alternatives):
                seen.add(cls)
                values = [str(member) for member in cls]
                if values not in _enums_in(properties.get(parameter.name)):
                    undescribed.append(f'{route.member.name} {parameter.name}: {cls.__name__}')
    assert {CameraAutoMode, StageAxis} <= seen, 'the walk did not reach the closed-set parameters'
    assert not undescribed, f'described without their set: {undescribed}'
