# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where a channel's manual sliders stop is the session's answer, and only a hint.

``ScopeSession.channel_slider_maxima`` answers each channel's exposure and
illumination slider maximum with the connected part's cap and the layer's
manual ceiling already combined, so the GUI shows the answer and computes
nothing, and a REST or SDK caller reads the same numbers. Neither is a
bound: a typed value above one is still accepted, up to what the camera or
the LED board can be asked for. With no part to cap a slider the answer is
None, never a stand-in maximum.
"""

from __future__ import annotations

import dataclasses

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.channel_record import ChannelSliderMaxima
from tests.stock_channels import LED0
from tests.test_a_camera_command_with_no_camera_is_refused import _camera_less
from tests.test_a_command_for_absent_motion_hardware_is_refused import (
    make_session,  # noqa: F401 -- the fixture this file's tests take
)


def test_each_maximum_is_the_parts_cap_narrowed_to_the_layers_ceiling(make_session):
    session = make_session('LS850')
    caps = session.scope.capabilities
    camera_ms, board_ma = session.scope.camera.setting_limits.exposure_max_ms, caps.led_max_ma
    # The ceilings (scopes.json) sit under the simulated parts' caps for BF
    # and above them for fluorescence; luminescence has none and no LED.
    assert (camera_ms, board_ma) == (1000.0, 1000)

    gain_db = session.scope.camera.setting_limits.gain_max_db
    assert gain_db == 48.0

    assert session.channel_slider_maxima('BF') == ChannelSliderMaxima(50.0, 50, gain_db)
    assert session.channel_slider_maxima(LED0) == ChannelSliderMaxima(camera_ms, board_ma, gain_db)
    assert session.channel_slider_maxima('Lumi') == ChannelSliderMaxima(camera_ms, None, gain_db)


def test_with_no_camera_the_exposure_maximum_is_none(make_session, monkeypatch):
    session = _camera_less(make_session, 'LS850')

    # With no camera the gain slider keeps the documented 48 dB.
    assert session.channel_slider_maxima('BF') == ChannelSliderMaxima(None, 50, 48.0)


def test_with_no_led_board_every_illumination_maximum_is_none(make_session, monkeypatch):
    session = make_session('LS850')
    # The capabilities of a scope that came up without its board, as
    # test_an_led_command_for_absent_hardware_is_refused builds them: the
    # simulator cannot bring a session up with its LED board absent.
    monkeypatch.setattr(
        session.scope,
        'capabilities',
        dataclasses.replace(session.scope.capabilities, led_max_ma=None, led_addresses=None),
    )

    assert session.channel_slider_maxima('BF') == ChannelSliderMaxima(50.0, None, 48.0)
    assert session.channel_slider_maxima(LED0).slider_max_illumination_ma is None


def test_a_channel_with_no_led_on_this_scope_has_no_illumination_maximum(make_session):
    # The LS560's unit carries no Ex405 LED; its board is present.
    session = make_session('LS560')
    assert session.scope.channel_identity.find(LED0) is None

    assert session.channel_slider_maxima(LED0).slider_max_illumination_ma is None


def test_an_unknown_channel_is_refused(make_session):
    with pytest.raises(ArgumentRefusedError) as refused:
        make_session('LS850').channel_slider_maxima('Ex999')
    assert refused.value.reason == 'channel_unknown'


def test_a_typed_exposure_above_the_slider_maximum_is_accepted(make_session):
    session = make_session('LS850')
    slider_max = session.channel_slider_maxima('BF').slider_max_exposure_ms

    session.scope.camera.set_exposure_ms(slider_max * 4)

    assert session.scope.camera.read_exposure_ms() == pytest.approx(slider_max * 4)
