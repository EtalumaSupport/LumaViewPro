# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The selected plate is the scope's answer: ``runtime_state.get_labware()``.

The GUI's stage map and every other caller ask the scope which plate is
selected. The answer is the catalogue's plate for the name the settings
hold; a stored name the catalogue does not have is refused by name, never
replaced by another plate, whose geometry would move every well.
"""

import pytest


@pytest.fixture
def scope_and_settings():
    """A simulated LS850 bound to the template's settings; a test names its plate there."""
    from tests.scope_fakes import bind_settings_like_a_session, build_scope

    scope = build_scope(simulate=True, sim_model='LS850', register_atexit=False)
    settings = bind_settings_like_a_session(scope)
    yield scope, settings
    scope.disconnect()


def test_the_answer_is_the_plate_the_settings_name(scope_and_settings):
    scope, settings = scope_and_settings
    settings['protocol']['labware'] = '6 well microplate'

    assert scope.runtime_state.get_labware() == scope.labware['6 well microplate']


def test_a_stored_plate_the_catalogue_lacks_is_refused_by_name(scope_and_settings):
    # 'New' was the old KV spinner default that leaked into settings.
    from modules.exceptions import CatalogueNameRefusedError

    scope, settings = scope_and_settings
    settings['protocol']['labware'] = 'New'

    with pytest.raises(CatalogueNameRefusedError) as refused:
        scope.runtime_state.get_labware()
    assert (refused.value.reason, refused.value.value) == ('labware_unknown', 'New')
