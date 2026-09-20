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

    def make(labware=STARTING_PLATE, **overrides):
        # The starting plate is named here rather than laid over the template,
        # because it belongs inside the protocol block and an override would
        # put it at the top level where nothing reads it.
        settings = complete_settings(**overrides)
        settings.setdefault('protocol', {})['labware'] = labware
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


def _runtime_plate_shape(session):
    """Which plate the scope is actually on, as its well grid.

    The stored ``WellPlate`` carries no name, so the question has to be put
    to something the object exposes -- and it must be something that TELLS
    PLATES APART. ``get_dimensions()`` does not: nine of the ten plates in
    ``data/labware.json`` are the same SBS footprint, so assertions written
    against it pass whether or not the runtime store was ever written. The
    well grid does discriminate: 96-well is 8x12, 384-well is 16x24.
    """
    plate = session.scope.runtime_state.get_labware()
    return (plate.config['rows'], plate.config['columns'])


def _expected_plate_shape(plate_name):
    plate = labware_loader.WellPlateLoader(source_path='.').get_plate(plate_key=plate_name)
    return (plate.config['rows'], plate.config['columns'])


class TestSelectingAPlate:
    def test_a_new_plate_moves_both_stores(self, sessions):
        session = sessions()

        assert session.select_labware(OTHER_PLATE) is True

        assert session.settings['protocol']['labware'] == OTHER_PLATE
        assert _runtime_plate_shape(session) == _expected_plate_shape(OTHER_PLATE)

    def test_selecting_the_held_plate_is_a_no_op(self, sessions):
        session = sessions()

        assert session.select_labware(STARTING_PLATE) is False

        assert session.settings['protocol']['labware'] == STARTING_PLATE

    def test_a_renamed_plate_is_accepted_under_its_old_name_and_stored_under_its_key(
        self, sessions
    ):
        """A protocol saved before the rename still names a real plate.

        The old spelling is accepted and translated here, once: the settings
        store carries the catalogue key, so nothing downstream ever compares
        an old spelling to a key, and a spinner restored from settings finds
        the name in its list.
        """
        session = sessions()

        assert session.select_labware(RENAMED_PLATE_OLD_NAME) is True

        assert session.settings['protocol']['labware'] == OTHER_PLATE
        assert _runtime_plate_shape(session) == _expected_plate_shape(OTHER_PLATE)

    def test_the_old_name_of_the_held_plate_is_a_no_op(self, sessions):
        # "Changed" is decided on the key, not the spelling handed in.
        session = sessions(labware=OTHER_PLATE)

        assert session.select_labware(RENAMED_PLATE_OLD_NAME) is False

        assert session.settings['protocol']['labware'] == OTHER_PLATE


class TestTheSettingsStoreIsNotAProxyForTheScope:
    """The no-op gate reads the settings store, so a writer that gets there
    first can make the member believe there is nothing to do.

    This is not hypothetical: loading a protocol used to write the labware
    key and only then set the spinner text, and the spinner's text event is
    what reaches this member. The member saw its own answer already stored,
    reported no change, and left the scope on the previous plate -- settings
    naming one plate while every well position came from another.
    """

    def test_a_settings_write_that_beats_the_member_does_not_silence_it(self, sessions):
        session = sessions()
        assert _runtime_plate_shape(session) == _expected_plate_shape(STARTING_PLATE)

        # Exactly what the out-of-band writer did, one line ahead of the call.
        session.settings['protocol']['labware'] = OTHER_PLATE
        session.select_labware(OTHER_PLATE)

        assert _runtime_plate_shape(session) == _expected_plate_shape(OTHER_PLATE), (
            'the scope is still on the old plate while settings name the new one'
        )


class TestRefusals:
    def test_an_unknown_plate_is_refused_and_moves_nothing(self, sessions):
        session = sessions()
        before = _runtime_plate_shape(session)

        with pytest.raises(ConfigError, match='unknown labware'):
            session.select_labware('Acme 1536 Ultra Plate')

        assert session.settings['protocol']['labware'] == STARTING_PLATE
        assert _runtime_plate_shape(session) == before

    def test_settings_with_no_protocol_block_are_refused_before_anything_moves(self, sessions):
        """A store that cannot hold the plate is not one to write half of.

        Settings handed straight to a factory skip the template merge that
        puts the protocol block there, so it can genuinely be absent. Writing
        the runtime state first and discovering it at the settings write would
        leave the two stores describing different plates -- the one outcome
        this member exists to make impossible.
        """
        session = sessions()
        before = _runtime_plate_shape(session)
        del session.settings['protocol']

        with pytest.raises(ConfigError, match='no usable protocol block'):
            session.select_labware(OTHER_PLATE)

        assert _runtime_plate_shape(session) == before

    @pytest.mark.parametrize('block', ['a string', ['a', 'list'], 7])
    def test_a_protocol_block_that_is_not_a_mapping_is_refused_the_same_way(self, sessions, block):
        """Absent and unusable are the same answer to the caller.

        A hand-edited settings file can put anything in that slot. Checking
        only for None would let the rest through to a reach that raises
        AttributeError instead of the refusal this member promises.
        """
        session = sessions()
        before = _runtime_plate_shape(session)
        session.settings['protocol'] = block

        with pytest.raises(ConfigError, match='no usable protocol block'):
            session.select_labware(OTHER_PLATE)

        assert _runtime_plate_shape(session) == before

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
