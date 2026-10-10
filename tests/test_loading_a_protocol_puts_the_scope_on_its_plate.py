# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Loading a protocol puts the scope on the protocol's plate, or refuses.

A protocol's positions are stated against the plate it names. The GUI's
Load parsed the file through the scope's API and then set the plate
spinner, whose handler wrote the store's plate into the protocol whether
the Session had taken the file's plate or refused it: under a recording,
a protocol saved on one plate was adopted on another, every well position
computed from the wrong geometry. The Session now loads a protocol and
selects its plate as one member, and a refused selection refuses the load.
"""

import logging

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_labware_name_resolution import _protocol_file

FILE_PLATE = '6 well microplate'


def _create(**overrides):
    return ScopeSession.create(complete_settings(**overrides), simulate=True)


def _shutdown(session):
    try:
        session.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


@pytest.fixture
def session():
    built = _create()
    yield built
    _shutdown(built)


def _scope_plate(session):
    return session.scope.runtime_state.get_labware().config


def _catalogue_plate(session, name):
    return session.wellplate_loader.get_plate(plate_key=name).config


class TestTheSessionLoad:
    def test_the_scope_takes_the_plate_the_file_names(self, tmp_path, session):
        assert session.settings['protocol']['labware'] != FILE_PLATE

        protocol = session.load_protocol(_protocol_file(tmp_path, FILE_PLATE))

        assert protocol.labware() == FILE_PLATE
        assert session.settings['protocol']['labware'] == FILE_PLATE
        assert _scope_plate(session) == _catalogue_plate(session, FILE_PLATE)

    def test_a_held_scope_refuses_a_protocol_on_another_plate(self, tmp_path, session):
        plate_before = session.settings['protocol']['labware']
        runtime_before = session.scope.runtime_state.get_labware()
        held = session.activity_claim.try_claim('recording')
        try:
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.load_protocol(_protocol_file(tmp_path, FILE_PLATE))
        finally:
            held.release()

        assert refused.value.holder == 'recording'
        assert session.settings['protocol']['labware'] == plate_before
        assert session.scope.runtime_state.get_labware().config == runtime_before.config

    def test_a_held_scope_loads_a_protocol_on_the_plate_in_place(self, tmp_path, session):
        in_place = session.settings['protocol']['labware']
        held = session.activity_claim.try_claim('recording')
        try:
            protocol = session.load_protocol(_protocol_file(tmp_path, in_place))
        finally:
            held.release()

        assert protocol.labware() == in_place


class TestAScopeWithNoStage:
    def test_the_protocol_and_the_scope_take_center_plate(self, tmp_path):
        session = _create(microscope='LS820')
        try:
            assert not session.scope.capabilities.has_xy_stage

            protocol = session.load_protocol(_protocol_file(tmp_path, FILE_PLATE))

            assert protocol.labware() == 'Center Plate'
            assert session.settings['protocol']['labware'] == 'Center Plate'
            assert _scope_plate(session) == _catalogue_plate(session, 'Center Plate')
        finally:
            _shutdown(session)
