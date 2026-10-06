"""A camera's black level is read, set where the camera offers it, and recorded.

The black level is the camera's own offset parameter, in the camera's own
units. A setter makes it a capture variable, so every frame's record states
it beside the gain. A camera that does not offer it refuses before anything
reaches the camera; a value outside the camera's live range is refused the
same way; a camera holding its black level automatically refuses a manual
one.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pypylon import genicam

from drivers.exceptions import HardwareError
from modules.exceptions import (
    CameraSettingOutOfRangeError,
    CameraSettingRejected,
    CameraSettingUnsupportedError,
)
from modules.scope_session import ScopeSession
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera
from tests.settings_fixtures import complete_settings


def test_the_simulated_camera_offers_it_and_offsets_its_frames(sim_scope):
    imaging = sim_scope.imaging
    assert sim_scope.capabilities.camera_supports_black_level is True
    assert imaging.get_black_level() == 0.0
    sim_scope._camera_driver.set_test_pattern(enabled=True, pattern='Black')

    assert imaging.set_black_level(4.0) == 4.0
    assert imaging.get_black_level() == 4.0
    image = imaging.capture_and_wait(force_to_8bit=False, accept_dark=True, timeout_s=2.0)

    assert image is not None
    assert np.all(image == 4)
    assert imaging.last_capture_info['frame_record'].black_level == 4.0


def test_a_full_scale_pixel_saturates_under_the_offset_and_does_not_wrap(sim_scope):
    imaging = sim_scope.imaging
    sim_scope._camera_driver.set_test_pattern(enabled=True, pattern='White')
    full_scale = imaging.capture_and_wait(force_to_8bit=False, timeout_s=2.0)

    imaging.set_black_level(4.0)
    image = imaging.capture_and_wait(force_to_8bit=False, timeout_s=2.0)

    assert full_scale is not None and image is not None
    assert np.array_equal(image, full_scale)


def test_a_camera_refusal_is_a_rejection_in_the_black_levels_words(sim_scope):
    with (
        patch.object(sim_scope._camera_driver, 'set_black_level', return_value=False),
        pytest.raises(CameraSettingRejected, match='did not accept the black level 4'),
    ):
        sim_scope.imaging.set_black_level(4.0)


def test_the_simulator_reloads_its_default_black_level_at_connect(sim_scope):
    driver = sim_scope._camera_driver
    sim_scope.imaging.set_black_level(4.0)
    driver.init_camera_config()
    assert driver.get_black_level() == 0.0


def test_a_value_outside_the_range_is_refused_and_nothing_moves(sim_scope):
    imaging = sim_scope.imaging
    with pytest.raises(CameraSettingOutOfRangeError):
        imaging.set_black_level(1000.0)
    assert imaging.get_black_level() == 0.0


def test_the_range_is_the_one_the_setter_refuses_outside(sim_scope):
    imaging = sim_scope.imaging
    low, high = imaging.get_black_level_range()
    assert (low, high) == sim_scope._camera_driver.get_black_level_range()
    assert imaging.set_black_level(high) == high
    with pytest.raises(CameraSettingOutOfRangeError) as refused:
        imaging.set_black_level(high + 1.0)
    assert refused.value.maximum == high


def test_a_failed_range_read_raises(sim_scope):
    with (
        patch.object(
            sim_scope._camera_driver,
            'get_black_level_range',
            side_effect=HardwareError('BlackLevel range read failed'),
        ),
        pytest.raises(HardwareError, match='BlackLevel range read failed'),
    ):
        sim_scope.imaging.get_black_level_range()


def test_no_active_camera_has_no_range(sim_scope):
    with patch.object(sim_scope._camera_driver, '_active', None):
        assert sim_scope.imaging.get_black_level_range() is None


def test_a_failed_read_fails_the_capture_and_records_no_frame(sim_scope):
    imaging = sim_scope.imaging
    driver = sim_scope._camera_driver
    with (
        patch.object(
            driver, 'get_black_level', side_effect=HardwareError('BlackLevel read failed')
        ),
        pytest.raises(HardwareError, match='BlackLevel read failed'),
    ):
        imaging.capture_and_wait(accept_dark=True, timeout_s=2.0)


@pytest.fixture(scope='module')
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_the_fx2_reports_its_row_black_target_and_offers_no_setting(fx2_session):
    scope = fx2_session.scope
    assert scope.capabilities.camera_supports_black_level is False
    assert scope.imaging.get_black_level() == 0.0
    assert scope.imaging.get_black_level_range() is None
    with pytest.raises(CameraSettingUnsupportedError):
        scope.imaging.set_black_level(4.0)
    assert scope.imaging.capture_and_wait(accept_dark=True, timeout_s=2.0) is not None
    assert scope.imaging.last_capture_info['frame_record'].black_level == 0.0


# --- Pylon --------------------------------------------------------------------


def _pylon(nodes=('BlackLevel', 'BlackLevelSelector')):
    cam = bare_pylon_camera()

    def get_node(name):
        if name not in nodes:
            raise genicam.LogicalErrorException('Node not existing')
        return MagicMock()

    cam.active.GetNodeMap.return_value.GetNode.side_effect = get_node
    return cam


def test_pylon_sets_the_selector_then_the_value_and_reads_it_back():
    cam = _pylon()
    cam.active.BlackLevel.GetValue.return_value = 4.0625
    assert cam.set_black_level(4.07) == 4.0625
    cam.active.BlackLevelSelector.SetValue.assert_called_once_with('All')
    cam.active.BlackLevel.SetValue.assert_called_once_with(4.07)


def test_pylon_without_the_selector_writes_the_value_alone():
    cam = _pylon(nodes=('BlackLevel',))
    cam.active.BlackLevel.GetValue.return_value = 2.0
    assert cam.set_black_level(2.0) == 2.0
    cam.active.BlackLevelSelector.SetValue.assert_not_called()


def test_pylon_lost_comms_on_the_write_is_a_refusal_and_a_disconnect():
    cam = _pylon()
    cam.active.BlackLevel.SetValue.side_effect = genicam.RuntimeException('gone')
    assert cam.set_black_level(2.0) is False
    cam._mark_disconnected.assert_called_once()


def test_pylon_failed_write_raises_and_keeps_the_camera():
    cam = _pylon()
    cam.active.BlackLevel.SetValue.side_effect = genicam.OutOfRangeException('refused')
    with pytest.raises(HardwareError, match=r'BlackLevel 2\.0 write failed'):
        cam.set_black_level(2.0)
    cam._mark_disconnected.assert_not_called()


def test_pylon_failed_read_back_raises():
    cam = _pylon()
    cam.active.BlackLevel.GetValue.side_effect = genicam.OutOfRangeException('refused')
    with pytest.raises(HardwareError, match=r'BlackLevel 2\.0 write failed'):
        cam.set_black_level(2.0)


def test_pylon_failed_probe_raises():
    cam = bare_pylon_camera()
    cam.active.GetNodeMap.side_effect = genicam.RuntimeException('transient')
    with pytest.raises(HardwareError, match='BlackLevel probe failed'):
        cam.supports_black_level()
    cam._mark_disconnected.assert_not_called()


def test_pylon_failed_read_raises_and_keeps_the_camera():
    cam = _pylon()
    cam.active.BlackLevel.GetValue.side_effect = genicam.RuntimeException('transient')
    with pytest.raises(HardwareError, match='BlackLevel read failed'):
        cam.get_black_level()
    cam._mark_disconnected.assert_not_called()


def test_pylon_without_the_node_reports_none_and_offers_nothing():
    cam = _pylon(nodes=())
    assert cam.get_black_level() is None
    assert cam.get_black_level_range() is None
    assert cam.supports_black_level() is False


# --- IDS ----------------------------------------------------------------------


def _ids(auto):
    cam = bare_ids_camera()
    nodes = {'BlackLevel': MagicMock(), 'BlackLevelAuto': MagicMock()}
    nodes['BlackLevelAuto'].CurrentEntry.return_value.SymbolicValue.return_value = auto
    nodes['BlackLevel'].Value.return_value = 3.0
    cam.remote_nodemap.HasNode.side_effect = lambda name: name in nodes
    cam.remote_nodemap.FindNode.side_effect = nodes.__getitem__
    return cam, nodes['BlackLevel']


def test_ids_with_auto_black_level_on_refuses_and_writes_nothing():
    cam, node = _ids(auto='Continuous')
    assert cam.set_black_level(3.0) is False
    node.SetValue.assert_not_called()


def test_ids_with_auto_black_level_off_writes_and_reads_back():
    cam, node = _ids(auto='Off')
    assert cam.set_black_level(3.0) == 3.0
    node.SetValue.assert_called_once_with(3.0)


def test_ids_failed_write_raises():
    cam, node = _ids(auto='Off')
    node.SetValue.side_effect = RuntimeError('peak: write failed')
    with pytest.raises(HardwareError, match=r'BlackLevel 3\.0 write failed'):
        cam.set_black_level(3.0)


def test_ids_failed_probe_raises():
    cam, _node = _ids(auto='Off')
    cam.remote_nodemap.HasNode.side_effect = RuntimeError('peak: probe failed')
    with pytest.raises(HardwareError, match='BlackLevel probe failed'):
        cam.supports_black_level()
