# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every write of a channel's settings is judged whole, then stored and applied by one write.

A channel's fields had several writers, each with its own subset of the
rules: a protocol Load stored the shown channel's gain, exposure and
illumination without putting them on the camera or the LED, so a still then
recorded settings that were not used; a Load and a step stored an
illumination above the LED board's maximum; a Load turned auto-gain off
without the lock and left a channel acquiring and stimulating; Set All
Bookmarks stored a focus past the travel check; auto-gain was stored on
before the camera refused its inverted range; and a write held back while a
run or a diagnostic held the scope never reached the LED, or the camera
after a diagnostic. ``ScopeSession`` now has one channel write: every field
is judged before any is stored, and the channel shown when the write ends is
on the camera, and on its LED when lit. A write held back during a run or a
diagnostic is applied at its end; one to the shown channel's hardware during
a home is refused, as every other request during a home is.
"""

from __future__ import annotations

import datetime
import threading

import pandas as pd
import pytest

from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    PositionOutOfRangeError,
    ProtocolRunRefusedError,
)
from _lumascope.modules.lumascope_api.camera import CameraAutoMode
from _lumascope.modules.protocol import ChannelSettings, Protocol
from tests.protocol_fixtures import protocol_of
from tests.scope_fakes import channel_reading
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture
from tests.test_a_home_holds_the_scope import HOME_S, _HeldHome
from tests.test_going_to_a_step_is_the_sessions_move import _PLATE, _objective_of, _step

_RUN_WAIT_S = 120.0
STIM_CHANNEL = 'Ex405'


def _layers(*steps: dict, **rows: ChannelSettings) -> Protocol:
    """A protocol built in memory whose Layer Settings are ``rows``."""
    return protocol_of(
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(list(steps) or [_step(0.0, 0.0, 0.0, '')]),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(hours=1.0),
            'labware_id': _PLATE,
            'capture_root': '',
            'tiling': '1x1',
            'channel_settings': rows,
        },
    )


def _row(acquire=AcquireMode.IMAGE, **cells) -> ChannelSettings:
    fields = dict.fromkeys(
        ('illumination_ma', 'gain_db', 'auto_gain', 'exposure_ms', 'false_color', 'frames_summed')
    )
    return ChannelSettings(acquire=acquire, **{**fields, **cells})


def _lit_shown_bf(session, illumination_ma=80.0, gain_db=9.0, exposure_ms=30.0):
    """BF shown and lit at ``illumination_ma``, the camera holding it at the given values."""
    session.show_channel('BF')
    session.update_settings('BF.illumination_ma', illumination_ma)
    session.update_settings('BF.gain_db', gain_db)
    session.update_settings('BF.exposure_ms', exposure_ms)
    session.light_shown_channel(True)


def _bf_led_ma(session):
    reading = channel_reading(session.scope.illumination, 'BF')
    return reading.illumination_ma if reading.state else None


def _invert_the_gain_range(session):
    session.update_settings('protocol.autogain.max_gain_db', 1.0)
    session.update_settings('protocol.autogain.min_gain_db', 5.0)


class TestALoadedLayerIsAppliedLikeAnyWrite:
    def test_the_shown_lit_channels_layer_is_on_the_camera_and_its_led(self, session):
        # OP-61: the Load stored the row and applied nothing.
        _lit_shown_bf(session)

        session.apply_channel_settings(
            _layers(BF=_row(illumination_ma=37.0, gain_db=2.5, exposure_ms=12.0, auto_gain=False))
        )

        camera = session.scope.camera
        assert (camera.read_gain_db(), camera.read_exposure_ms()) == pytest.approx((2.5, 12.0))
        assert _bf_led_ma(session) == pytest.approx(37.0)
        assert camera.get_held_channel() == 'BF'

    def test_a_layer_above_the_led_board_is_refused_and_nothing_changes(self, session):
        # OP-62: the board refuses to drive it, and over-driving an LED is a damage mode.
        board_max = session.scope.capabilities.led_max_ma
        before = session.get_settings_snapshot()

        with pytest.raises(ProtocolRunRefusedError) as refused:
            session.apply_channel_settings(_layers(BF=_row(illumination_ma=board_max + 1)))

        assert refused.value.reason == 'values_out_of_range'
        assert "LED board's maximum" in str(refused.value)
        assert session.get_settings_snapshot() == before

    def test_auto_gain_off_with_its_values_is_those_values_with_the_loop_off(self, session):
        # OP-63: "these values, the loop off" -- no lock takes the loop's values over the row's.
        session.show_channel('BF')
        session.set_channel_auto_gain('BF', True)

        session.apply_channel_settings(
            _layers(BF=_row(gain_db=2.5, exposure_ms=12.0, auto_gain=False))
        )

        camera = session.scope.camera
        assert camera.auto_gain == CameraAutoMode.OFF
        assert (camera.read_gain_db(), camera.read_exposure_ms()) == pytest.approx((2.5, 12.0))
        assert (session.settings['BF']['gain_db'], session.settings['BF']['exposure_ms']) == (
            2.5,
            12.0,
        )

    def test_auto_gain_off_alone_keeps_what_the_loop_reached(self, session):
        # OP-63: named alone, auto-gain off is the lock, as the Auto Gain box's untick is.
        session.show_channel('BF')
        session.set_channel_auto_gain('BF', True)

        session.apply_channel_settings(_layers(BF=_row(auto_gain=False)))

        lock = session.channel_auto_gain_lock('BF')
        assert lock is not None and lock.gain_db is not None
        assert session.settings['BF']['auto_gain'] is False
        assert session.settings['BF']['gain_db'] == round(lock.gain_db, 1)
        assert session.scope.camera.auto_gain == CameraAutoMode.OFF

    def test_a_layer_that_acquires_stops_its_channel_stimulating(self, session):
        # OP-64: one channel does not capture and stimulate at once.
        session.update_settings(f'{STIM_CHANNEL}.stim_config.enabled', True)

        session.apply_channel_settings(_layers(**{STIM_CHANNEL: _row()}))

        assert session.settings[STIM_CHANNEL]['acquire'] == AcquireMode.IMAGE
        assert session.settings[STIM_CHANNEL]['stim_config']['enabled'] is False


def test_a_step_above_the_led_board_is_refused_and_nothing_changes(session):
    # OP-62: a step's first visit stored a current the board will not drive.
    board_max = session.scope.capabilities.led_max_ma
    step = {**_step(60.0, 40.0, 5000.0, _objective_of(session)), 'Illumination': board_max + 1}
    protocol = _layers(step, BF=_row())
    before = session.get_settings_snapshot()

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.go_to_step(protocol, 0)

    assert refused.value.reason == 'values_out_of_range'
    assert "LED board's maximum" in str(refused.value)
    assert session.get_settings_snapshot() == before


def test_set_all_bookmarks_refuses_a_z_outside_travel_and_stores_nothing(session):
    # OP-65: every focus is judged before the bookmark and the foci are stored.
    motion = session.scope.motion
    beyond = motion.get_axis_limits('Z').max_um + 100.0
    motion.move_z(beyond, ignore_limits=True)
    before = session.get_settings_snapshot()

    with pytest.raises(PositionOutOfRangeError):
        session.save_all_bookmarks()

    assert session.get_settings_snapshot() == before


class TestAutoGainIsNotStoredOnWithARangeNoArmCanTake:
    """The Next row's harm: stored on, every later show or still of the channel was refused."""

    @pytest.mark.parametrize('channel', ['BF', 'Ex488'], ids=['shown', 'hidden'])
    def test_the_setter_refuses_and_stores_nothing(self, session, channel):
        # parametrize-justified: the shown channel arms the camera, a hidden one never does;
        # each must refuse before the store.
        session.show_channel('BF')
        _invert_the_gain_range(session)

        with pytest.raises(ArgumentRefusedError) as refused:
            session.set_channel_auto_gain(channel, True)

        assert refused.value.reason == 'gain_range_inverted'
        assert session.settings[channel]['auto_gain'] is False

    def test_a_loaded_layer_turning_it_on_is_refused_whole(self, session):
        _invert_the_gain_range(session)
        before = session.get_settings_snapshot()

        with pytest.raises(ArgumentRefusedError) as refused:
            session.apply_channel_settings(
                _layers(BF=_row(gain_db=2.5), Ex488=_row(auto_gain=True))
            )

        assert refused.value.reason == 'gain_range_inverted'
        assert session.get_settings_snapshot() == before


class TestAWriteHeldBackByAnActivityIsAppliedAtItsEnd:
    def test_a_diagnostic_ends_with_the_shown_channel_on_the_camera_and_led(self, session):
        _lit_shown_bf(session)

        with session.diagnostic_claim():
            session.update_settings('BF.gain_db', 7.5)
            session.update_settings('BF.illumination_ma', 45.0)
            # Stored only while the diagnostic holds the scope (D3).
            assert session.scope.camera.read_gain_db() == pytest.approx(9.0)

        assert session.scope.camera.read_gain_db() == pytest.approx(7.5)
        assert _bf_led_ma(session) == pytest.approx(45.0)

    @pytest.mark.slow
    def test_an_illumination_written_during_a_zstack_is_on_the_led_at_its_end(
        self, session, monkeypatch, tmp_path
    ):
        # OP-67: the run's end relit the LED at its pre-run current.
        _lit_shown_bf(session, illumination_ma=37.0)
        session.update_settings('zstack.range', 20.0)
        session.update_settings('zstack.step_size', 10.0)
        entered, release = threading.Event(), threading.Event()
        driver = session.scope._motion_driver
        real = driver.move_abs_pos

        def held(axis, position, *args, **kwargs):
            if axis == 'Z' and not entered.is_set():
                entered.set()
                assert release.wait(_RUN_WAIT_S), 'the held move was never released'
            return real(axis, position, *args, **kwargs)

        monkeypatch.setattr(driver, 'move_abs_pos', held)
        run = session.protocol_runner().start_zstack(
            'BF', parent_dir=str(tmp_path), enable_image_saving=False
        )
        try:
            assert entered.wait(_RUN_WAIT_S)
            session.update_settings('BF.illumination_ma', 80.0)
            session.update_settings('BF.gain_db', 4.0)
        finally:
            release.set()
        assert run.result(timeout=_RUN_WAIT_S).status == 'completed'

        assert _bf_led_ma(session) == pytest.approx(80.0)
        assert session.scope.camera.read_gain_db() == pytest.approx(4.0)


@pytest.mark.slow
def test_a_write_to_the_shown_channels_hardware_during_a_home_is_refused(session, monkeypatch):
    # Held back, it never reached the camera: a home ends on the IO lane, which cannot
    # wait on the camera's. Refused like every other request during a home; a hidden
    # channel's write is stored, since it goes on the hardware when that channel is shown.
    session.show_channel('BF')
    held = _HeldHome(session.scope, monkeypatch)
    home = session.scope.motion.start_home('ALL')
    try:
        assert held.entered.wait(HOME_S)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.update_settings('BF.gain_db', 7.5)
        assert refused.value.reason == 'exclusive_activity_running'
        session.update_settings('Ex488.gain_db', 7.5)
    finally:
        held.release.set()
        home.exception(timeout=HOME_S)

    assert session.settings['BF']['gain_db'] != 7.5
    assert session.settings['Ex488']['gain_db'] == 7.5


@pytest.mark.slow
def test_an_auto_gain_bound_the_shown_camera_takes_during_a_home_is_refused(session, monkeypatch):
    # The bound goes on the camera while the shown channel is on auto-gain, so it
    # is refused during a home as the channel's own fields are.
    session.show_channel('BF')
    session.set_channel_auto_gain('BF', True)
    before = session.get_setting('protocol.autogain.max_gain_db')
    held = _HeldHome(session.scope, monkeypatch)
    home = session.scope.motion.start_home('ALL')
    try:
        assert held.entered.wait(HOME_S)
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.update_settings('protocol.autogain.max_gain_db', before - 1.0)
        assert refused.value.reason == 'exclusive_activity_running'
    finally:
        held.release.set()
        home.exception(timeout=HOME_S)

    assert session.get_setting('protocol.autogain.max_gain_db') == before
