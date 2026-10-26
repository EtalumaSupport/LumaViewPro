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

from modules import layer_record
from modules.exceptions import SettingRefusedError
from modules.scope_session import ScopeSession
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
    release = layer_record.release_colors()

    session.update_settings(f'{LED2}.display_color', MAGENTA)
    overridden = session.channel_colors()
    session.update_settings(f'{LED2}.display_color', None)

    assert overridden == {**release, LED2: (1.0, 0.0, 1.0)}
    assert session.channel_colors() == release


def test_a_unit_colour_its_board_gives_is_the_channels_unless_overridden(session):
    # The board's own Channels block colours LED 0's channel cyan; a layer the block
    # leaves out keeps the release's colour, and the override wins over both.
    release = layer_record.release_colors()
    session.scope._motion_driver.motorconfig.update_from_board(
        {'Channels': {LED0: {'led_channel': 0, 'excitation_nm': 405, 'display_color': '#00ffff'}}}
    )
    session.scope.refresh_layer_identity()

    from_the_unit = session.channel_colors()
    session.update_settings(f'{LED0}.display_color', MAGENTA)

    assert from_the_unit == {**release, LED0: (0.0, 1.0, 1.0)}
    assert session.channel_colors() == {**release, LED0: (1.0, 0.0, 1.0)}


def test_a_colour_the_store_cannot_use_is_refused_at_the_write(session):
    refused = {}
    for value in ([1, 0, 1], 'magenta', '#800080', '#FF00FF'):
        with pytest.raises(SettingRefusedError) as raised:
            session.update_settings(f'{LED2}.display_color', value)
        refused[repr(value)] = raised.value.reason

    assert refused == {
        '[1, 0, 1]': 'wrong_kind',
        "'magenta'": 'out_of_range',
        "'#800080'": 'out_of_range',
        "'#FF00FF'": 'out_of_range',
    }
    assert session.get_setting(f'{LED2}.display_color') is None


def test_a_still_is_saved_in_its_channels_override(tmp_path):
    # An 8-bit fluorescence still carries its colour in the TIFF's palette:
    # the palette's brightest entry is the colour at full weight.
    with _open_session(_settings(tmp_path)) as session:
        session.show_channel(LED2)
        session.update_settings(f'{LED2}.display_color', MAGENTA)

        (path,) = _files(session, false_color_on=True)

        with tifffile.TiffFile(path) as tif:
            brightest = [int(v) for v in tif.pages[0].colormap[:, -1]]
    assert brightest == [65535, 0, 65535]
