# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A failure that ends at its caller is shown once, by its caller's reporter.

``save_image`` and the SDK tuning setters logged a traceback, posted their
own popup and re-raised, so the caller's reporter showed the same failure a
second time. Now they raise and post nothing. A disk failure in the save is
typed with the words that point at the disk; any other save failure keeps its
own type and words, since "check disk space" would be untrue for it.
"""

from __future__ import annotations

import pathlib

import pytest

from drivers.exceptions import HardwareError
from modules import image_save, image_utils
from modules.exceptions import CaptureError, ImageSaveError
from modules.lumascope_api.runtime_state import RuntimeState
from tests.test_jpg_export import _bright_mono, _scope_with_depth


@pytest.fixture
def posted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.notify',
        lambda *a, **kw: calls.append(a),
    )
    return calls


def _save(tmp_path):
    return image_save.save_image(
        _scope_with_depth(),
        _bright_mono(),
        save_folder=str(tmp_path),
        file_root='snap_',
        append='BF',
        channel='BF',
        false_color_on=False,
        tail_id_mode=None,
        output_format='JPG',
        save_encoding='8bit',
        significant_bits=8,
        objective_id='4x Oly',
    )


class TestSaveImage:
    def test_a_disk_failure_raises_the_typed_fault_and_posts_nothing(
        self, tmp_path, monkeypatch, posted
    ):
        denied = PermissionError('read-only folder')

        def refuse(self, data):
            raise denied

        monkeypatch.setattr(pathlib.Path, 'write_bytes', refuse)

        with pytest.raises(ImageSaveError) as excinfo:
            _save(tmp_path)

        assert isinstance(excinfo.value, CaptureError), (
            'the reporter shows a CaptureError in its words'
        )
        assert excinfo.value.title == 'Image Save Failed'
        assert 'Check disk space and permissions' in str(excinfo.value)
        assert excinfo.value.__cause__ is denied
        assert posted == []

    def test_a_failure_that_is_not_the_disk_keeps_its_own_words(
        self, tmp_path, monkeypatch, posted
    ):
        def bad_data(*args, **kwargs):
            raise ValueError('cannot encode this array as JPEG')

        monkeypatch.setattr(image_utils, 'encode_display_jpg', bad_data)

        with pytest.raises(ValueError, match='cannot encode') as excinfo:
            _save(tmp_path)

        assert not isinstance(excinfo.value, ImageSaveError)
        assert posted == []


# Each bench-tool SDK setter, the driver method it calls and its arguments.
SDK_SETTERS = [
    ('_set_acquisition_stop_mode', 'set_acquisition_stop_mode', ('Complete',)),
    ('_set_bandwidth_reserve_mode', 'set_bandwidth_reserve_mode', ('Performance',)),
    ('_set_device_link_throughput_limit', 'set_device_link_throughput_limit', ('Off',)),
    ('_set_max_transfer_size', 'set_max_transfer_size', (1048576,)),
    ('_set_num_max_queued_urbs', 'set_num_max_queued_urbs', (64,)),
    ('_set_max_num_buffer', 'set_max_num_buffer', (10,)),
    ('_set_gev_packet_size', 'set_gev_packet_size', (8192,)),
    ('_set_gev_inter_packet_delay', 'set_gev_inter_packet_delay', (1000,)),
]


class _FailingCamera:
    active = True

    def __init__(self, method):
        def fail(**kwargs):
            raise HardwareError(f'{method} failed: RuntimeException: node write timed out')

        setattr(self, method, fail)


@pytest.mark.parametrize(('setter', 'method', 'args'), SDK_SETTERS)
def test_an_sdk_setter_raises_the_driver_error_once_and_posts_nothing(setter, method, args, posted):
    from modules.lumascope_api import Lumascope
    from modules.lumascope_api.imaging import ImagingAPI

    camera = _FailingCamera(method)
    scope = Lumascope.__new__(Lumascope)
    scope.runtime_state = RuntimeState(scope)
    scope._camera_driver = camera
    imaging = ImagingAPI(scope, camera)

    with pytest.raises(HardwareError, match='node write timed out'):
        getattr(imaging, setter)(*args)

    assert posted == []
