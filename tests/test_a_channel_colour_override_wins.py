# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer's ``display_color`` setting overrides the colour it is shown and written in.

``ScopeSession.channel_colors`` is the one read of a channel's colour: a
layer's ``display_color``, a ``'#rrggbb'`` colour at full brightness, wins
over the unit's (its board block's, else the release catalogue's), and null
is no override. A value the store
cannot use as a colour is refused where it is written, never stored.
"""

import pytest
import tifffile

from _lumascope.modules import channel_record
from _lumascope.modules.exceptions import SettingRefusedError
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_manual_capture_member import _files, _open_session, _settings
from tests.stock_channels import LED0, LED2

MAGENTA = '#ff00ff'


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
    )
    yield s
    s.shutdown()
    s.scope.disconnect()


def test_an_override_is_the_channels_colour_and_null_gives_the_release_back(session):
    release = channel_record.release_colors()

    session.set_channel_settings(LED2, display_color=MAGENTA)
    overridden = session.channel_colors()
    session.set_channel_settings(LED2, display_color=None)

    assert overridden == {**release, LED2: (1.0, 0.0, 1.0)}
    assert session.channel_colors() == release


def test_a_unit_colour_its_board_gives_is_the_channels_unless_overridden(tmp_path):
    # The board's own Channels block colours LED 0's channel cyan; a layer the block
    # leaves out keeps the release's colour, and the override wins over both.
    from _lumascope.modules.lumascope_api.simulation import SimulatedScope
    from tests.test_a_simulated_scope_comes_up_as_a_units_own_config import FIELD_CONFIG

    release = channel_record.release_colors()
    unit = {
        **FIELD_CONFIG,
        'Channels': {LED0: {'led_channel': 0, 'excitation_nm': 405, 'display_color': '#00ffff'}},
    }
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), simulator_tier='firmware'),
        simulate=True,
        simulated=SimulatedScope(unit_config=unit),
        warn_pre_release=False,
    )
    try:
        from_the_unit = session.channel_colors()
        session.set_channel_settings(LED0, display_color=MAGENTA)
        overridden = session.channel_colors()
    finally:
        session.shutdown()

    assert from_the_unit == {**release, LED0: (0.0, 1.0, 1.0)}
    assert overridden == {**release, LED0: (1.0, 0.0, 1.0)}


def test_a_colour_the_store_cannot_use_is_refused_at_the_write(session):
    refused = {}
    # A value of another kind is the door's type refusal, before the store.
    for value in ('magenta', '#800080', '#FF00FF'):
        with pytest.raises(SettingRefusedError) as raised:
            session.set_channel_settings(LED2, display_color=value)
        refused[repr(value)] = raised.value.reason

    assert refused == {
        "'magenta'": 'out_of_range',
        "'#800080'": 'out_of_range',
        "'#FF00FF'": 'out_of_range',
    }
    assert session.channel_settings(LED2).display_color is None


def test_a_still_is_saved_in_its_channels_override(tmp_path):
    # An 8-bit fluorescence still carries its colour in the TIFF's palette:
    # the palette's brightest entry is the colour at full weight.
    with _open_session(_settings(tmp_path)) as session:
        session.show_channel(LED2)
        session.set_channel_settings(LED2, display_color=MAGENTA)
        session.set_channel_settings(LED2, false_color=True)

        (path,) = _files(session)

        with tifffile.TiffFile(path) as tif:
            brightest = [int(v) for v in tif.pages[0].colormap[:, -1]]
    assert brightest == [65535, 0, 65535]
