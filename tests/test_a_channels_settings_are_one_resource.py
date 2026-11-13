# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A channel's settings are one resource: one read answers them all, one write takes any of them.

A channel's settings were written through six per-channel members and dotted
paths, the GUI's sliders writing one nested key at a time and a caller
assembling a channel from many reads. ``ScopeSession.channel_settings``
answers every one of a channel's settings as a ``ChannelSettings``, and
``set_channel_settings`` writes any of them through the one channel write:
judged whole, so a refusal stores nothing; a field left out left as it is,
None a value; the record as stored answered. On the wire, a key left out is
a field not written and null is the value.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from fastapi.testclient import TestClient

from _lumascope.modules import common_utils
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.exceptions import HardwareCommandRefusedError, SettingRefusedError
from _lumascope.modules.lumascope_api.camera import CameraAutoMode
from _lumascope.modules.protocol import CHANNEL_SETTING_KEYS, ChannelSettings
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.scope_fakes import channel_reading
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture

STIM_CHANNEL = 'Ex405'


def _bf_led_ma(session):
    reading = channel_reading(session.scope.illumination, 'BF')
    return reading.illumination_ma if reading.state else None


def test_a_write_stores_the_fields_named_and_answers_the_record_the_read_answers(session):
    before = session.channel_settings(STIM_CHANNEL)

    answered = session.set_channel_settings(
        STIM_CHANNEL, gain_db=7.5, frames_summed=3, video_duration_s=8.0, stim_frequency_hz=2.0
    )

    assert answered == session.channel_settings(STIM_CHANNEL)
    assert answered == dataclasses.replace(
        before, gain_db=7.5, frames_summed=3, video_duration_s=8.0, stim_frequency_hz=2.0
    )
    assert session.settings[STIM_CHANNEL]['sum'] == 3
    assert session.settings[STIM_CHANNEL]['video_config']['duration'] == 8.0
    assert session.settings[STIM_CHANNEL]['stim_config']['frequency'] == 2.0


def test_none_is_a_value_written(session):
    session.set_channel_settings(
        'BF', acquire=AcquireMode.IMAGE, focus_z_um=5000.0, display_color='#00ff00'
    )

    answered = session.set_channel_settings('BF', acquire=None, focus_z_um=None, display_color=None)

    assert (answered.acquire, answered.focus_z_um, answered.display_color) == (None, None, None)
    assert session.settings['BF']['acquire'] is None
    assert session.settings['BF']['focus'] is None


def test_a_write_with_one_value_refused_stores_none_of_them(session):
    before = session.get_settings_snapshot()

    with pytest.raises(SettingRefusedError) as refused:
        session.set_channel_settings('BF', gain_db=4.0, frames_summed=0)

    assert (refused.value.reason, refused.value.path) == ('out_of_range', 'BF.sum')
    assert session.get_settings_snapshot() == before


def test_an_illumination_above_the_led_board_is_refused_and_nothing_changes(session):
    board_max = session.scope.capabilities.led_max_ma
    before = session.get_settings_snapshot()

    with pytest.raises(SettingRefusedError) as refused:
        session.set_channel_settings('BF', gain_db=4.0, illumination_ma=board_max + 1)

    assert (refused.value.reason, refused.value.path) == ('out_of_range', 'BF.illumination_ma')
    assert session.get_settings_snapshot() == before


def test_a_stimulation_current_above_the_led_board_is_refused_and_nothing_changes(session):
    board_max = session.scope.capabilities.led_max_ma
    before = session.get_settings_snapshot()

    with pytest.raises(SettingRefusedError, match="LED board's maximum") as refused:
        session.set_channel_settings(STIM_CHANNEL, stim_illumination_ma=board_max + 1)

    assert (refused.value.reason, refused.value.path) == (
        'out_of_range',
        f'{STIM_CHANNEL}.stim_config.illumination_ma',
    )
    assert session.get_settings_snapshot() == before
    session.set_channel_settings(STIM_CHANNEL, stim_illumination_ma=board_max)
    assert session.channel_settings(STIM_CHANNEL).stim_illumination_ma == board_max


def test_a_field_the_channel_does_not_carry_is_refused_and_read_none(session):
    before = session.get_settings_snapshot()

    with pytest.raises(SettingRefusedError) as refused:
        session.set_channel_settings('BF', gain_db=4.0, stim_frequency_hz=2.0)

    assert (refused.value.reason, refused.value.path) == ('not_offered', 'BF.stim_config.frequency')
    assert session.get_settings_snapshot() == before
    record = session.channel_settings('BF')
    assert (record.stim_frequency_hz, record.composite_brightness_threshold_pct) == (None, None)


def test_one_stimulation_field_leaves_the_others(session):
    stim = dict(session.settings[STIM_CHANNEL]['stim_config'])

    session.set_channel_settings(STIM_CHANNEL, stim_pulse_count=42)

    assert session.settings[STIM_CHANNEL]['stim_config'] == {**stim, 'pulse_count': 42}


def test_the_shown_lit_channel_takes_its_values_on_the_camera_and_led(session):
    session.show_channel('BF')
    session.light_shown_channel(True)

    session.set_channel_settings(
        'BF', illumination_ma=37.0, gain_db=2.5, exposure_ms=12.0, auto_gain=False
    )

    camera = session.scope.camera
    assert (camera.read_gain_db(), camera.read_exposure_ms()) == pytest.approx((2.5, 12.0))
    assert _bf_led_ma(session) == pytest.approx(37.0)


def test_auto_gain_off_alone_answers_what_the_loop_reached(session):
    session.show_channel('BF')
    session.set_channel_settings('BF', auto_gain=True)

    answered = session.set_channel_settings('BF', auto_gain=False)

    lock = session.channel_auto_gain_lock('BF')
    assert lock is not None and lock.gain_db is not None
    assert (answered.auto_gain, answered.gain_db) == (False, round(lock.gain_db, 1))
    assert session.scope.camera.auto_gain == CameraAutoMode.OFF


def test_acquiring_stops_the_channel_stimulating(session):
    session.set_channel_settings(STIM_CHANNEL, stim_enabled=True)

    answered = session.set_channel_settings(STIM_CHANNEL, acquire=AcquireMode.IMAGE)

    assert (answered.acquire, answered.stim_enabled) == (AcquireMode.IMAGE, False)


def test_a_channel_the_scope_lacks_is_refused_acquiring(session):
    absent = next(
        name
        for name in common_utils.get_channels()
        if name not in {record.key_name for record in session.scope.channel_identity.channels}
    )

    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.set_channel_settings(absent, acquire=AcquireMode.IMAGE)

    assert refused.value.reason == 'not_on_this_scope'
    assert session.settings[absent]['acquire'] is None


def test_every_stored_channel_setting_is_a_field_of_the_one_write(session):
    """A key a channel's block holds with no field would have no writer once the dotted paths close."""
    fields = [field for field, _ in CHANNEL_SETTING_KEYS]
    keywords = [
        name
        for name, parameter in inspect.signature(
            ScopeSession.set_channel_settings
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    template = session.scope.settings_template
    unwritten = {}
    for channel in common_utils.get_channels():
        keys = set()
        for key, value in template[channel].items():
            keys |= {f'{key}.{sub}' for sub in value} if isinstance(value, dict) else {key}
        unwritten[channel] = sorted(keys - {key for _, key in CHANNEL_SETTING_KEYS})

    assert fields == [f.name for f in dataclasses.fields(ChannelSettings)] == keywords
    assert {channel: keys for channel, keys in unwritten.items() if keys} == {}


class TestOnTheWire:
    @pytest.fixture
    def client(self, session):
        with TestClient(build_app(session)) as client:
            yield client

    def test_a_key_left_out_is_not_written_and_null_is_the_value(self, client, session):
        session.set_channel_settings('BF', focus_z_um=5000.0, gain_db=3.0)

        answer = client.put('/api/v1/channel_settings', json={'channel': 'BF', 'focus_z_um': None})

        assert answer.status_code == 200, answer.text
        assert answer.json()['focus_z_um'] is None
        assert answer.json()['gain_db'] == 3.0
        assert (
            client.get('/api/v1/channel_settings', params={'channel': 'BF'}).json() == answer.json()
        )
