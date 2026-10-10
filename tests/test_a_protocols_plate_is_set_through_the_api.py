# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's plate is set through the API, and a scope with no XY stage is on Center Plate.

The protocol panel wrote the scope's plate into its protocol after every
plate selection, even one the scope refused, and that write was also the
only thing that moved a stage-less scope's protocol onto Center Plate: the
rule had three homes (the panel's write, the microscope panel's Center
Plate call, and ``load_protocol``'s copy). Now bring-up puts a stage-less
scope on Center Plate, so every protocol made from the settings is born
there, and ``set_protocol_labware`` holds the rule for a protocol's plate
in one body that ``load_protocol`` also calls.
"""

from __future__ import annotations

import logging

import pytest

from modules.exceptions import CatalogueNameRefusedError
from modules.labware_loader import CENTER_PLATE
import modules.scope_session as scope_session_module
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

STORED_PLATE = '96 well microplate'


def _shutdown(session):
    try:
        session.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def _session(tmp_path, microscope):
    settings = complete_settings(live_folder=str(tmp_path), microscope=microscope)
    settings['protocol']['labware'] = STORED_PLATE
    return ScopeSession.create(settings, simulate=True)


@pytest.fixture
def stage_scope(tmp_path):
    session = _session(tmp_path, 'LS850')
    assert session.scope.capabilities.has_xy_stage
    yield session
    _shutdown(session)


@pytest.fixture
def stageless_scope(tmp_path):
    session = _session(tmp_path, 'LS820')
    assert not session.scope.capabilities.has_xy_stage
    yield session
    _shutdown(session)


class TestBringUp:
    def test_a_stageless_scope_comes_up_on_center_plate_and_says_so(self, tmp_path, monkeypatch):
        # The Session logs through lvp_logger's logger, which the suite's
        # conftest replaces; its lines are read where they are written.
        lines = []
        monkeypatch.setattr(
            scope_session_module.logger, 'info', lambda msg, *a, **kw: lines.append(str(msg))
        )
        session = _session(tmp_path, 'LS820')
        try:
            assert session.settings['protocol']['labware'] == CENTER_PLATE
            assert session.scope.runtime_state.get_labware().config == (
                session.wellplate_loader.get_plate(plate_key=CENTER_PLATE).config
            )
            assert any(
                f'stored plate {STORED_PLATE!r} replaced by {CENTER_PLATE!r}' in line
                for line in lines
            )
        finally:
            _shutdown(session)

    def test_a_scope_with_a_stage_keeps_its_stored_plate(self, stage_scope):
        assert stage_scope.settings['protocol']['labware'] == STORED_PLATE

    def test_a_stageless_scopes_protocols_are_born_on_center_plate(self, stageless_scope):
        assert stageless_scope.create_empty_protocol().labware() == CENTER_PLATE


class TestSetProtocolLabware:
    def test_the_protocol_takes_the_plate_and_the_scope_keeps_its_own(self, stage_scope):
        protocol = stage_scope.create_empty_protocol()

        taken = stage_scope.set_protocol_labware(protocol, '6 well microplate')

        assert taken == '6 well microplate'
        assert protocol.labware() == '6 well microplate'
        assert stage_scope.settings['protocol']['labware'] == STORED_PLATE

    def test_a_retired_spelling_is_stored_under_the_catalogues_key(self, stage_scope):
        protocol = stage_scope.create_empty_protocol()

        assert stage_scope.set_protocol_labware(protocol, 'Center Dish') == CENTER_PLATE
        assert protocol.labware() == CENTER_PLATE

    def test_a_plate_the_catalogue_lacks_is_refused_and_the_protocol_keeps_its_plate(
        self, stage_scope
    ):
        protocol = stage_scope.create_empty_protocol()

        with pytest.raises(CatalogueNameRefusedError, match='labware catalogue'):
            stage_scope.set_protocol_labware(protocol, 'a plate nobody makes')

        assert protocol.labware() == STORED_PLATE

    def test_a_stageless_scopes_protocol_takes_center_plate_and_says_so(
        self, stageless_scope, caplog
    ):
        protocol = stageless_scope.create_empty_protocol()

        with caplog.at_level(logging.INFO):
            taken = stageless_scope.set_protocol_labware(protocol, STORED_PLATE)

        assert taken == CENTER_PLATE
        assert protocol.labware() == CENTER_PLATE
        assert any(
            f'protocol plate {STORED_PLATE!r} replaced by {CENTER_PLATE!r}' in r.getMessage()
            for r in caplog.records
        )


def test_a_file_on_a_plate_loads_on_center_plate_on_a_stageless_scope(
    stage_scope, stageless_scope, tmp_path
):
    protocol = stage_scope.create_empty_protocol()
    path = tmp_path / 'on_a_plate.tsv'
    protocol.to_file(path)

    loaded = stageless_scope.load_protocol(path)

    assert loaded.labware() == CENTER_PLATE
    assert stageless_scope.settings['protocol']['labware'] == CENTER_PLATE
