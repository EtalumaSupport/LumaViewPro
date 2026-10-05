"""A camera reports its link: transport, negotiated speed, and GigE's stream
packet size and inter-packet delay, each None where the camera reports none.

A read that fails raises; it is never turned into None.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pypylon import genicam

from drivers import fx2driver
from drivers.camera import Camera, link_info
from drivers.exceptions import HardwareError
from modules.scope_session import ScopeSession
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera
from tests.settings_fixtures import complete_settings


def test_the_simulated_camera_is_on_usb3_and_reports_nothing_more(sim_scope):
    assert sim_scope.diagnostics.get_camera_link_info() == link_info(transport='USB3')


@pytest.fixture(scope='module')
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


def test_the_fx2_is_usb2_at_the_speed_it_negotiated(fx2_session):
    assert fx2_session.scope.diagnostics.get_camera_link_info() == {
        'transport': 'USB2',
        'link_speed': 480.0,
        'link_speed_unit': 'Mbps',
        'packet_size_bytes': None,
        'inter_packet_delay': None,
    }


@pytest.mark.parametrize('libusb_speed, mbps', [(0, None), (2, 12.0), (3, 480.0), (4, 5000.0)])
def test_the_fx2_transport_reads_libusbs_speed(libusb_speed, mbps):
    transport = fx2driver._PyusbTransport()
    assert transport.link_speed_mbps() is None
    transport._dev = SimpleNamespace(speed=libusb_speed)
    assert transport.link_speed_mbps() == mbps


def test_a_camera_that_reports_no_link_answers_none_for_every_field():
    assert Camera.get_link_info(SimpleNamespace(active=True)) == link_info()
    assert Camera.get_link_info(SimpleNamespace(active=False)) is None


def test_unknown_sdk_transport_names_pass_through():
    assert link_info(transport='CoaXPress')['transport'] == 'CoaXPress'


# --- Pylon --------------------------------------------------------------------


def _pylon(values, device_class='BaslerUsb', speed_unit='bps'):
    cam = bare_pylon_camera()

    def get_node(name):
        if name not in values:
            raise genicam.LogicalErrorException('Node not existing')
        return MagicMock()

    cam.active.GetNodeMap.return_value.GetNode.side_effect = get_node
    cam.active.GetDeviceInfo.return_value.GetDeviceClass.return_value = device_class
    for name, value in values.items():
        getattr(cam.active, name).GetValue.return_value = value
    cam.active.DeviceLinkSpeed.GetUnit.return_value = speed_unit
    return cam


def test_pylon_usb3_reports_its_link_speed_in_the_unit_it_declares():
    cam = _pylon({'DeviceLinkSpeed': 5_000_000_000})
    assert cam.get_link_info() == {
        'transport': 'USB3',
        'link_speed': 5_000_000_000,
        'link_speed_unit': 'bps',
        'packet_size_bytes': None,
        'inter_packet_delay': None,
    }


def test_pylon_speed_with_no_declared_unit_says_its_unit_is_unknown():
    cam = _pylon({'DeviceLinkSpeed': 400_000_000}, speed_unit='')
    info = cam.get_link_info()
    assert info['link_speed'] == 400_000_000
    assert info['link_speed_unit'] is None


def test_pylon_gige_reports_its_stream_packet_settings():
    cam = _pylon(
        {'DeviceLinkSpeed': 125_000_000, 'GevSCPSPacketSize': 9000, 'GevSCPD': 1000},
        device_class='BaslerGigE',
        speed_unit='Bps',
    )
    assert cam.get_link_info() == {
        'transport': 'GigE',
        'link_speed': 125_000_000,
        'link_speed_unit': 'Bps',
        'packet_size_bytes': 9000,
        'inter_packet_delay': 1000,
    }


def test_pylon_failed_read_raises_and_keeps_the_camera():
    cam = _pylon({'DeviceLinkSpeed': 0})
    cam.active.DeviceLinkSpeed.GetValue.side_effect = genicam.RuntimeException('transient')
    with pytest.raises(HardwareError, match='Link info read failed'):
        cam.get_link_info()
    cam._mark_disconnected.assert_not_called()


# --- IDS ----------------------------------------------------------------------


def _ids(values, tl_type='USB3Vision'):
    cam = bare_ids_camera()
    nodes = {name: MagicMock() for name in values}
    for name, value in values.items():
        nodes[name].Value.return_value = value
        nodes[name].Unit.return_value = 'Bps'
    if tl_type is not None:
        nodes['DeviceTLType'] = MagicMock()
        nodes['DeviceTLType'].CurrentEntry.return_value.SymbolicValue.return_value = tl_type
    cam.remote_nodemap.HasNode.side_effect = lambda name: name in nodes
    cam.remote_nodemap.FindNode.side_effect = nodes.__getitem__
    return cam, nodes


def test_ids_usb3_reports_its_transport_and_link_speed():
    cam, _nodes = _ids({'DeviceLinkSpeed': 400_000_000})
    assert cam.get_link_info() == {
        'transport': 'USB3',
        'link_speed': 400_000_000,
        'link_speed_unit': 'Bps',
        'packet_size_bytes': None,
        'inter_packet_delay': None,
    }


def test_ids_speed_with_no_declared_unit_says_its_unit_is_unknown():
    cam, nodes = _ids({'DeviceLinkSpeed': 400_000_000})
    nodes['DeviceLinkSpeed'].Unit.return_value = ''
    assert cam.get_link_info()['link_speed_unit'] is None


def test_ids_without_the_nodes_reports_none_for_each():
    cam, _nodes = _ids({}, tl_type=None)
    assert cam.get_link_info() == link_info()


def test_ids_failed_read_raises():
    cam, nodes = _ids({'DeviceLinkSpeed': 0})
    nodes['DeviceLinkSpeed'].Value.side_effect = RuntimeError('peak: read failed')
    with pytest.raises(HardwareError, match='Link info read failed'):
        cam.get_link_info()
