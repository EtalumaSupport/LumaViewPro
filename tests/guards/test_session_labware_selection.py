# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session owns the active labware, for every host.

Bring-up sets the plate from settings and offers no way back, so before this
surface existed a caller that was not the GUI could start with a plate and
never switch one -- the plate was reachable only through a Kivy spinner
handler. These cases run headless, which is the point: they are the proof that
the capability is not GUI-bound.

The plate decides every well position the program computes, so the settings
store and the scope's runtime state have to move together or not at all. A
name the loader cannot resolve is refused before either store is written: a
write that half-lands leaves the two describing different plates, and the
wrongness is silent -- a capture still happens, at the wrong place.
"""

import logging

import pytest

from modules import labware_loader
from modules.exceptions import ConfigError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


STARTING_PLATE = '96 well microplate'
OTHER_PLATE = '384 well microplate'
# data/labware.json's alias map: this plate was renamed, and protocols saved
# under the old name still have to load.
RENAMED_PLATE_OLD_NAME = '384 well Corning Spheroid Microplate'


@pytest.fixture
def sessions():
    """Build sessions and shut every one of them down."""
    built = []

    def make(**overrides):
        settings = complete_settings(**overrides)
        settings.setdefault('protocol', {})['labware'] = overrides.pop('labware', STARTING_PLATE)
        session = ScopeSession.create_headless(settings=settings)
        built.append(session)
        return session

    yield make
    for session in reversed(built):
        try:
            session.shutdown()
        except Exception:
            logging.getLogger(__name__).debug(
                'teardown noise is not the measurement', exc_info=True
            )


def _plate_dimensions(session):
    """What the runtime object reports, which is all it can be asked.

    The stored ``WellPlate`` carries no name of its own, so 'which plate is
    the scope on' can only be answered by the geometry it hands out.
    """
    return session.scope.runtime_state.get_labware().get_dimensions()


def _expected_dimensions(plate_name):
    return (
        labware_loader.WellPlateLoader(source_path='.')
        .get_plate(plate_key=plate_name)
        .get_dimensions()
    )


class TestSelectingAPlate:
    def test_a_new_plate_moves_both_stores(self, sessions):
        session = sessions()

        assert session.select_labware(OTHER_PLATE) is True

        assert session.settings['protocol']['labware'] == OTHER_PLATE
        assert _plate_dimensions(session) == _expected_dimensions(OTHER_PLATE)

    def test_selecting_the_held_plate_is_a_no_op(self, sessions):
        session = sessions()

        assert session.select_labware(STARTING_PLATE) is False

        assert session.settings['protocol']['labware'] == STARTING_PLATE

    def test_a_renamed_plate_is_accepted_under_its_old_name(self, sessions):
        """A protocol saved before the rename still names a real plate.

        Validation goes through the loader's membership test rather than its
        list of names, because the list holds canonical keys only -- checking
        against it would refuse a plate the lookup resolves perfectly well.
        """
        session = sessions()

        assert session.select_labware(RENAMED_PLATE_OLD_NAME) is True

        assert _plate_dimensions(session) == _expected_dimensions(OTHER_PLATE)


class TestRefusals:
    def test_an_unknown_plate_is_refused_and_moves_nothing(self, sessions):
        session = sessions()
        before = _plate_dimensions(session)

        with pytest.raises(ConfigError, match='unknown labware'):
            session.select_labware('Acme 1536 Ultra Plate')

        assert session.settings['protocol']['labware'] == STARTING_PLATE
        assert _plate_dimensions(session) == before

    def test_settings_with_no_protocol_block_are_refused_before_anything_moves(self, sessions):
        """A store that cannot hold the plate is not one to write half of.

        Settings handed straight to a factory skip the template merge that
        puts the protocol block there, so it can genuinely be absent. Writing
        the runtime state first and discovering it at the settings write would
        leave the two stores describing different plates -- the one outcome
        this member exists to make impossible.
        """
        session = sessions()
        before = _plate_dimensions(session)
        del session.settings['protocol']

        with pytest.raises(ConfigError, match='no protocol block'):
            session.select_labware(OTHER_PLATE)

        assert _plate_dimensions(session) == before

    def test_a_non_string_is_refused_by_name_not_by_TypeError(self, sessions):
        """A wire payload decodes to whatever it decodes to.

        The loader resolves through a dict lookup, so an unhashable value
        raises TypeError out of the membership test instead of answering it.
        A caller handing one over gets the same refusal as any other bad name.
        """
        session = sessions()

        with pytest.raises(ConfigError, match='must be a string'):
            session.select_labware(['96 well microplate'])

        assert session.settings['protocol']['labware'] == STARTING_PLATE
