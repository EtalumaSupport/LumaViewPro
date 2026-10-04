# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Add and Update measure the stage on the protocol's plate, not the scope's.

A step's X and Y are plate coordinates, and a run drives them through the
protocol's plate. Add Step and Update Step converted the live stage
position through the scope's selected plate instead, so with the scope on
a plate of other dimensions than the protocol's the step was stored in a
frame the run does not drive it in. In the shipped catalogue only the
Four-Slide Holder differs from the microplates (127.5 x 85.5 mm against
127.76 x 85.48), so the error is 0.26 mm in X and 0.02 mm in Y; an
installation's own plates can differ by more.
"""

import logging

import pytest

from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

PROTOCOL_PLATE = '96 well microplate'
SCOPE_PLATE = 'Four-Slide Holder'
STAGE_UM = {'X': 40000, 'Y': 30000}
# The stage at STAGE_UM on each plate (sim LS850, stage offset 5500 / 4000 um).
ON_PROTOCOL_PLATE = (82.26, 51.48)
ON_SCOPE_PLATE = (82.0, 51.5)


@pytest.fixture
def session(tmp_path):
    built = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'),
        simulate=True,
    )
    home_sim_scope(built.scope)
    built.set_layer_acquire('BF', 'image')
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def _protocol_on_one_plate_scope_on_another(session):
    protocol = session.create_empty_protocol()
    session.select_labware(SCOPE_PLATE)
    assert protocol.labware() == PROTOCOL_PLATE
    assert session.settings['protocol']['labware'] == SCOPE_PLATE
    return protocol


def _move_stage(session, position):
    for axis, um in position.items():
        session.scope.motion.move_absolute(axis, um, wait_until_complete=True)


def _xy(step):
    return (step['X'], step['Y'])


def test_add_stores_the_stage_on_the_protocols_plate(session):
    protocol = _protocol_on_one_plate_scope_on_another(session)
    _move_stage(session, STAGE_UM)

    session.add_step(protocol)

    assert _xy(protocol.step(0)) == ON_PROTOCOL_PLATE


def test_update_stores_the_stage_on_the_protocols_plate(session):
    protocol = _protocol_on_one_plate_scope_on_another(session)
    session.add_step(protocol)
    _move_stage(session, STAGE_UM)

    session.update_step(protocol, 0, layer='BF')

    assert _xy(protocol.step(0)) == ON_PROTOCOL_PLATE


def test_the_scopes_position_is_still_read_on_the_scopes_plate(session):
    _protocol_on_one_plate_scope_on_another(session)
    _move_stage(session, STAGE_UM)

    here = session.get_current_plate_position()

    assert (here['x'], here['y']) == ON_SCOPE_PLATE
