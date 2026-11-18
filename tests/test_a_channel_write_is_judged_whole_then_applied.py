# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every write of a channel's settings is judged whole, then stored and applied by one write.

A channel's fields had several writers, each with its own subset of the
rules: a protocol Load stored the shown channel's gain, exposure and
illumination without putting them on the camera or the LED, so a still then
recorded settings that were not used; a Load and a step stored an
illumination above the LED board's maximum as asked, where the one write now
stores it at the maximum and tells; a Load turned auto-gain off
without the lock and left a channel acquiring and stimulating; Set All
Bookmarks stored a focus past the travel check; auto-gain was stored on
before the camera refused its inverted range; and a write held back while a
run or a diagnostic held the scope never reached the LED, or the camera
after a diagnostic. ``ScopeSession`` now has one channel write: every field
is judged before any is stored, and the channel shown when the write ends is
on the camera, and on its LED when lit. A write held back during a run or a
diagnostic is applied at its end; one to the shown channel's hardware during
a home is refused, as every other request during a home is. The shown
channel's camera and LED take a write before it is stored, so a value either
refuses is not stored.
"""

from __future__ import annotations

import datetime
import threading

import pandas as pd
import pytest

from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.image_mode import ImageMode
from _lumascope.drivers.exceptions import BoardPortFailedError
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    CameraSettingRejected,
    HardwareCommandRefusedError,
    PositionOutOfRangeError,
    ValueClampedNotice,
)
from _lumascope.modules.lumascope_api.camera import CameraAutoMode
from _lumascope.modules.protocol import ChannelSettings, Protocol
from _lumascope.modules.scope_session import ScopeSession
from tests.protocol_fixtures import protocol_of
from tests.scope_fakes import channel_reading
from tests.settings_fixtures import complete_settings
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
    session.set_channel_settings('BF', illumination_ma=illumination_ma)
    session.set_channel_settings('BF', gain_db=gain_db)
    session.set_channel_settings('BF', exposure_ms=exposure_ms)
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

    def test_a_layer_above_the_led_board_is_stored_at_its_maximum_and_told(
        self, session, centre_posts
    ):
        # The board does not drive past its maximum, so the layer's current is
        # set at it, as a value typed into the box is.
        board_max = session.scope.capabilities.led_max_ma

        session.apply_channel_settings(_layers(BF=_row(illumination_ma=board_max + 1)))

        assert session.settings['BF']['illumination_ma'] == board_max
        assert _clamp_notices(centre_posts) == [
            str(ValueClampedNotice([('BF', 'illumination_ma', board_max + 1, float(board_max))]))
        ]

    def test_auto_gain_off_with_its_values_is_those_values_with_the_loop_off(self, session):
        # OP-63: "these values, the loop off" -- no lock takes the loop's values over the row's.
        session.show_channel('BF')
        session.set_channel_settings('BF', auto_gain=True)

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
        session.set_channel_settings('BF', auto_gain=True)

        session.apply_channel_settings(_layers(BF=_row(auto_gain=False)))

        lock = session.channel_auto_gain_lock('BF')
        assert lock is not None and lock.gain_db is not None
        assert session.settings['BF']['auto_gain'] is False
        assert session.settings['BF']['gain_db'] == round(lock.gain_db, 1)
        assert session.scope.camera.auto_gain == CameraAutoMode.OFF

    def test_a_layer_that_acquires_stops_its_channel_stimulating(self, session):
        # OP-64: one channel does not capture and stimulate at once.
        session.set_channel_settings(STIM_CHANNEL, stim_enabled=True)

        session.apply_channel_settings(_layers(**{STIM_CHANNEL: _row()}))

        assert session.settings[STIM_CHANNEL]['acquire'] == AcquireMode.IMAGE
        assert session.settings[STIM_CHANNEL]['stim_config']['enabled'] is False


def test_a_step_above_the_led_board_is_stored_at_its_maximum_and_told(session, centre_posts):
    # A step's visit writes its current into its channel through the one
    # channel write, which sets it at the board's maximum.
    board_max = session.scope.capabilities.led_max_ma
    step = {**_step(60.0, 40.0, 5000.0, _objective_of(session)), 'Illumination': board_max + 1}
    protocol = _layers(step, BF=_row())

    session.go_to_step(protocol, 0)

    assert session.settings['BF']['illumination_ma'] == board_max
    assert protocol.step_row(step_index=0)['Illumination'] == board_max + 1
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('BF', 'illumination_ma', board_max + 1.0, float(board_max))]))
    ]


def _clamp_notices(centre_posts):
    return [post.message for post in centre_posts if post.reason == ValueClampedNotice.reason]


def test_set_all_bookmarks_refuses_a_z_outside_travel_and_stores_nothing(session):
    # OP-65: every focus is judged before the bookmark and the foci are stored.
    motion = session.scope.motion
    beyond = motion.axis_limits('Z').max_um + 100.0
    motion.move_z(beyond, ignore_limits=True)
    before = session.settings_snapshot()

    with pytest.raises(PositionOutOfRangeError):
        session.save_all_bookmarks()

    assert session.settings_snapshot() == before


class TestAutoGainIsNotStoredOnWithARangeNoArmCanTake:
    """The Next row's harm: stored on, every later show or still of the channel was refused."""

    @pytest.mark.parametrize('channel', ['BF', 'Ex488'], ids=['shown', 'hidden'])
    def test_the_setter_refuses_and_stores_nothing(self, session, channel):
        # parametrize-justified: the shown channel arms the camera, a hidden one never does;
        # each must refuse before the store.
        session.show_channel('BF')
        _invert_the_gain_range(session)

        with pytest.raises(ArgumentRefusedError) as refused:
            session.set_channel_settings(channel, auto_gain=True)

        assert refused.value.reason == 'gain_range_inverted'
        assert session.settings[channel]['auto_gain'] is False

    def test_a_loaded_layer_turning_it_on_is_refused_whole(self, session):
        _invert_the_gain_range(session)
        before = session.settings_snapshot()

        with pytest.raises(ArgumentRefusedError) as refused:
            session.apply_channel_settings(
                _layers(BF=_row(gain_db=2.5), Ex488=_row(auto_gain=True))
            )

        assert refused.value.reason == 'gain_range_inverted'
        assert session.settings_snapshot() == before

    def test_a_step_turning_it_on_is_refused_before_the_stage_moves(self, session):
        # Going to a step asks the channel write before any move starts, so
        # the refusal leaves the stage where it was as well as the store.
        _invert_the_gain_range(session)
        step = {**_step(60.0, 40.0, 5000.0, _objective_of(session)), 'Auto_Gain': True}
        protocol = _layers(step, BF=_row())
        before = session.settings_snapshot()
        targets = session.scope.motion.target_positions()

        with pytest.raises(ArgumentRefusedError) as refused:
            session.start_go_to_step(protocol, 0)

        assert refused.value.reason == 'gain_range_inverted'
        assert session.scope.motion.target_positions() == targets
        assert session.settings_snapshot() == before


class TestAWriteHeldBackByAnActivityIsAppliedAtItsEnd:
    def test_a_diagnostic_ends_with_the_shown_channel_on_the_camera_and_led(self, session):
        _lit_shown_bf(session)

        with session.diagnostic_claim():
            session.set_channel_settings('BF', gain_db=7.5)
            session.set_channel_settings('BF', illumination_ma=45.0)
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
            session.set_channel_settings('BF', illumination_ma=80.0)
            session.set_channel_settings('BF', gain_db=4.0)
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
            session.set_channel_settings('BF', gain_db=7.5)
        assert refused.value.reason == 'exclusive_activity_running'
        session.set_channel_settings('Ex488', gain_db=7.5)
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
    session.set_channel_settings('BF', auto_gain=True)
    before = session.setting('protocol.autogain.max_gain_db')
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

    assert session.setting('protocol.autogain.max_gain_db') == before


def _shown_bf_at(session, gain_db=9.0, exposure_ms=30.0):
    """BF shown, the camera holding it at the given gain and exposure, auto-gain off."""
    session.show_channel('BF')
    session.set_channel_settings('BF', auto_gain=False, gain_db=gain_db, exposure_ms=exposure_ms)


class TestAValueTheCameraRefusesIsNotStored:
    """Rule 9: the shown channel's camera is asked before the store.

    Stored first, a value the camera refused stayed in the channel, so the
    box, the settings file and every later still of the channel held a value
    the camera never took, and each still asked for it again.
    """

    @pytest.mark.parametrize(
        ('node', 'field', 'refused'),
        [('gain', 'gain_db', 12.0), ('exposure', 'exposure_ms', 45.0)],
        ids=['gain', 'exposure'],
    )
    def test_the_channel_keeps_the_earlier_value_and_the_next_still_is_taken_at_it(
        self, session, node, field, refused
    ):
        # parametrize-justified: each refusal is armed on its own camera node.
        _shown_bf_at(session)
        earlier = getattr(session.channel_settings('BF'), field)
        before = session.settings_snapshot()
        session.scope.simulation.refuse_camera_write(node, 'raise')

        with pytest.raises(CameraSettingRejected) as rejected:
            session.set_channel_settings('BF', **{field: refused})

        assert rejected.value.reason == 'camera_rejected'
        assert getattr(session.channel_settings('BF'), field) == earlier
        assert session.settings_snapshot() == before
        record = session.scope.imaging.capture_still('BF', accept_dark=True).record
        assert getattr(record, field) == pytest.approx(earlier)

    def test_a_write_whose_gain_alone_is_refused_stores_neither_value(self, session):
        _shown_bf_at(session)
        before = session.settings_snapshot()
        session.scope.simulation.refuse_camera_write('gain', 'raise')

        with pytest.raises(CameraSettingRejected):
            session.set_channel_settings('BF', gain_db=12.0, exposure_ms=45.0)

        assert session.settings_snapshot() == before

    def test_a_loaded_layer_the_camera_refuses_stores_no_row(self, session):
        _shown_bf_at(session)
        before = session.settings_snapshot()
        session.scope.simulation.refuse_camera_write('gain', 'raise')

        with pytest.raises(CameraSettingRejected):
            session.apply_channel_settings(
                _layers(BF=_row(gain_db=2.5, exposure_ms=12.0, auto_gain=False))
            )

        assert session.settings_snapshot() == before

    def test_an_auto_gain_bound_the_camera_refuses_is_not_stored(self, session):
        session.show_channel('BF')
        session.set_channel_settings('BF', auto_gain=True)
        before = session.settings_snapshot()
        session.scope.simulation.refuse_camera_write('auto_gain', 'raise')

        with pytest.raises(CameraSettingRejected):
            session.update_settings(
                'protocol.autogain.max_gain_db',
                session.setting('protocol.autogain.max_gain_db') - 1.0,
            )

        assert session.settings_snapshot() == before


def _fail_the_range_read(session):
    """The camera's gain and exposure range left unread: its re-read after a readout change fails."""
    camera = session.scope.camera
    camera.set_image_mode(ImageMode.EIGHT_BIT)
    session.scope.simulation.refuse_camera_read('minimum_exposure', 'failed', count=1)
    camera.set_image_mode(ImageMode.SCIENTIFIC)


_RANGE_UNREAD_WORDS = (
    "The camera's gain and exposure range could not be read, so the gain and exposure "
    'changes were not sent. It is read again at the next change of binning, frame size '
    'or pixel format, else when LumaViewPro restarts.'
)


class TestAWriteWithTheCameraRangeUnreadIsRefusedInItsWords:
    """With the camera's range unread, a channel's gain and exposure are refused as the camera refuses them.

    The channel write held them to the range through the camera's reads,
    which raise the bare read failure, so the person was told only that the
    operation did not complete; the camera's own writes say the range could
    not be read, which settings were not sent, and what reads it again.
    """

    # parametrize-justified: the shown channel goes to the camera, one not shown only to the store.
    @pytest.mark.parametrize('channel', ['BF', 'Lumi'])
    def test_typing_a_gain_and_exposure_is_refused_and_nothing_is_stored(self, session, channel):
        session.show_channel('BF')
        _fail_the_range_read(session)
        before = session.settings_snapshot()

        with pytest.raises(CameraSettingRejected) as refused:
            session.set_channel_settings(channel, gain_db=12.0, exposure_ms=45.0)

        assert refused.value.reason == 'camera_read_failed'
        assert str(refused.value) == _RANGE_UNREAD_WORDS
        assert session.settings_snapshot() == before

    def test_a_loaded_layer_is_refused_and_stores_no_row(self, session):
        session.show_channel('BF')
        _fail_the_range_read(session)
        before = session.settings_snapshot()

        with pytest.raises(CameraSettingRejected) as refused:
            session.apply_channel_settings(
                _layers(BF=_row(gain_db=2.5, exposure_ms=12.0, auto_gain=False))
            )

        assert refused.value.reason == 'camera_read_failed'
        assert str(refused.value) == _RANGE_UNREAD_WORDS
        assert session.settings_snapshot() == before


@pytest.fixture
def ls620():
    # The FX2's LED board is the simulated one that takes a refused write.
    built = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield built
    built.shutdown()


def test_a_light_the_led_refuses_after_the_camera_took_the_write_stores_nothing_and_puts_the_camera_back(
    ls620,
):
    channel = ls620.shown_channel.channel
    ls620.set_channel_settings(channel, auto_gain=False, gain_db=3.0, illumination_ma=10.0)
    ls620.light_shown_channel(True)
    camera = ls620.scope.camera
    held_gain = camera.read_gain_db()  # the FX2's step nearest the 3.0 asked
    before = ls620.settings_snapshot()
    ls620.scope.simulation.refuse_led_write()

    with pytest.raises(BoardPortFailedError):
        ls620.set_channel_settings(channel, gain_db=6.0, illumination_ma=20.0)

    assert ls620.settings_snapshot() == before
    assert camera.read_gain_db() == pytest.approx(held_gain)
    assert camera.get_held_channel() == channel
