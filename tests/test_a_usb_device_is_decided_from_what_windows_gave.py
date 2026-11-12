# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A USB device's binding is decided from the properties Windows' one walk read.

The walk of Windows' USB device list (``winusb_iso.usb_devices``) reads each
wanted device's properties as Windows gives them and decides nothing;
``binding_from`` decides the binding on any host. A property read for the
log alone that came in a type LumaViewPro does not read is recorded on its
device, which is still read; one the binding is decided by fails the read,
since whether the device is usable is then not known.
"""

import pytest

from _lumascope.drivers.usb_driver_binding import DeviceProperties, binding_from

_OURS = {'oem12.inf'}


def _code_31_on_ours(**unreadable) -> DeviceProperties:
    """Our package bound to the running firmware and not started, as the build PC read it on 2026-11-11."""
    return DeviceProperties(
        instance_id='USB\\VID_04B4&PID_EA17\\5&1A2B3C&0&2',
        values={
            'present': True,
            'service': 'WinUSB',
            'provider': 'Etaluma, Inc.',
            'inf': 'OEM12.INF',
            'version': '1.0.0.3',
            'date': '2026-11-11',
            'problem_code': 31,
        },
        unreadable=unreadable,
    )


def test_a_log_only_property_of_an_unread_type_is_recorded_and_the_device_still_read():
    binding = binding_from(_code_31_on_ours(problem_status='type 0x19'), _OURS)

    assert binding.unreadable == ('problem status (type 0x19)',)
    assert binding.problem_status is None
    assert binding.ours and not binding.started
    assert binding.problem_code == 31


def test_a_deciding_property_of_an_unread_type_fails_the_read_naming_it():
    with pytest.raises(Exception) as refused:
        binding_from(_code_31_on_ours(service='type 0x19'), _OURS)

    assert refused.value.reason == 'camera_usb_property_unread'
    assert 'service as type 0x19' in str(refused.value)
