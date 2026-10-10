"""A camera reports the frame rate its current settings allow.

The figure is the camera's own: Basler's resulting acquisition frame rate,
IDS's AcquisitionFrameRate maximum, the simulator's pacing rate. The FX2
reports none. A failed read raises; it is never turned into None.
"""

from unittest.mock import MagicMock

import pytest
from pypylon import genicam

from drivers.exceptions import HardwareError
from modules.scope_session import ScopeSession
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera
from tests.settings_fixtures import complete_settings


def test_the_simulator_reports_the_rate_it_paces_frames_at(sim_scope):
    imaging = sim_scope.imaging
    imaging.set_exposure_ms(100.0)
    assert imaging.get_resulting_frame_rate() == pytest.approx(10.0)
    imaging.set_exposure_ms(1.0)
    assert imaging.get_resulting_frame_rate() == pytest.approx(
        sim_scope._camera_driver._MAX_DELIVERY_FPS
    )


@pytest.fixture(scope='module')
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_the_fx2_reports_none(fx2_session):
    assert fx2_session.scope.imaging.get_resulting_frame_rate() is None


# --- Pylon --------------------------------------------------------------------


def _pylon(nodes):
    cam = bare_pylon_camera()

    def get_node(name):
        if name not in nodes:
            raise genicam.LogicalErrorException('Node not existing')
        return MagicMock()

    cam.active.GetNodeMap.return_value.GetNode.side_effect = get_node
    return cam


def test_pylon_reads_the_bsl_node_first():
    cam = _pylon(nodes=('BslResultingAcquisitionFrameRate', 'ResultingFrameRate'))
    cam.active.BslResultingAcquisitionFrameRate.GetValue.return_value = 19.4
    cam.active.ResultingFrameRate.GetValue.return_value = 99.0
    assert cam.get_resulting_frame_rate() == 19.4


def test_pylon_falls_back_to_the_legacy_node():
    cam = _pylon(nodes=('ResultingFrameRate',))
    cam.active.ResultingFrameRate.GetValue.return_value = 35.0
    assert cam.get_resulting_frame_rate() == 35.0


def test_pylon_without_either_node_reports_none():
    assert _pylon(nodes=()).get_resulting_frame_rate() is None


def test_pylon_failed_read_raises_and_keeps_the_camera():
    cam = _pylon(nodes=('BslResultingAcquisitionFrameRate',))
    cam.active.BslResultingAcquisitionFrameRate.GetValue.side_effect = genicam.RuntimeException(
        'transient'
    )
    with pytest.raises(HardwareError, match='Resulting frame rate read failed'):
        cam.get_resulting_frame_rate()
    cam._mark_disconnected.assert_not_called()


# --- IDS ----------------------------------------------------------------------


def _ids(nodes):
    cam = bare_ids_camera()
    cam.remote_nodemap.HasNode.side_effect = lambda name: name in nodes
    cam.remote_nodemap.FindNode.side_effect = nodes.__getitem__
    return cam


def test_ids_reports_the_acquisition_frame_rate_maximum():
    node = MagicMock()
    node.Maximum.return_value = 61.5
    node.Value.return_value = 12.0
    assert _ids({'AcquisitionFrameRate': node}).get_resulting_frame_rate() == 61.5


def test_ids_without_the_node_reports_none():
    assert _ids({}).get_resulting_frame_rate() is None


def test_ids_failed_read_raises():
    node = MagicMock()
    node.Maximum.side_effect = RuntimeError('peak: read failed')
    with pytest.raises(HardwareError, match='AcquisitionFrameRate maximum read failed'):
        _ids({'AcquisitionFrameRate': node}).get_resulting_frame_rate()
