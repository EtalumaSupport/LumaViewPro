# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored plate the catalogue does not have is refused, never replaced.

Every plate position is converted through the selected plate, so a
different plate's geometry would put every well in the wrong place while
the protocol reads as if it ran normally. Bring-up configures the scope
through the plate lookup, so an unusable stored plate stops the session
there, before anything else reads it; the GUI answers that refusal by
coming up on the shipped template and asking the user. After bring-up the
settings writers hold the same rule.
"""

import threading
import time

import pytest

from modules.exceptions import ConfigError, SettingRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# Measured: the workers of a normal shutdown are gone within 3 s.
_WORKERS_EXIT_BOUND_S = 10.0


def _settings_naming(plate, tmp_path):
    settings = complete_settings(live_folder=str(tmp_path))
    settings['protocol']['labware'] = plate
    return settings


def test_bring_up_refuses_an_unknown_plate_by_name(tmp_path):
    before = set(threading.enumerate())
    with pytest.raises(ConfigError, match="unknown labware 'Retired Plate'") as refused:
        ScopeSession.create(_settings_naming('Retired Plate', tmp_path), simulate=True)
    # The message says what the user can pick instead.
    assert '96 well microplate' in str(refused.value)
    # The factory tears down what it started before it lets the refusal out.
    # Shutdown stops the lanes without joining their workers -- a normal
    # bring-up and shutdown leaves them exiting the same way -- so this waits,
    # bounded, for them to go rather than asserting the instant it returns.
    deadline = time.monotonic() + _WORKERS_EXIT_BOUND_S
    while not set(threading.enumerate()) <= before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert set(threading.enumerate()) <= before


def test_the_shipped_template_plate_brings_up(tmp_path):
    # The GUI's answer to the refusal above is to come up on the template;
    # that only works if the template's own plate resolves.
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        assert session.scope.runtime_state.get_labware() is not None
    finally:
        session.shutdown()


class TestThePlateHasOneWriter:
    @pytest.fixture
    def session(self, tmp_path):
        s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        yield s
        s.shutdown()

    def test_the_settings_writer_refuses_the_plate_naming_its_member(self, session):
        stored = dict(session.settings['protocol'])
        with pytest.raises(SettingRefusedError, match='select_labware'):
            session.update_settings('protocol.labware', 'nonexistent')
        assert session.settings['protocol'] == stored

    def test_the_protocol_block_is_not_written_whole(self, session):
        stored = session.settings['protocol']
        with pytest.raises(SettingRefusedError, match='block'):
            session.update_settings('protocol', {**stored, 'labware': 'nonexistent'})
        assert session.settings['protocol'] is stored

    def test_an_unknown_plate_is_refused_by_its_member(self, session):
        stored = dict(session.settings['protocol'])
        with pytest.raises(ConfigError, match="unknown labware 'nonexistent'"):
            session.select_labware('nonexistent')
        assert session.settings['protocol'] == stored

    def test_a_retired_spelling_is_stored_under_the_catalogue_key(self, session):
        session.select_labware('Center Dish')
        assert session.settings['protocol']['labware'] == 'Center Plate'
