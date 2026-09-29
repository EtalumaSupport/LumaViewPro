# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A failure that ends at its caller is shown once, by its caller's reporter.

``save_image`` logged a traceback, posted its own popup and re-raised, so the
caller's reporter showed the same failure a second time. Now it raises and
posts nothing. A disk failure is typed with the words that point at the disk;
any other save failure keeps its own type and words, since "check disk space"
would be untrue for it.
"""

from __future__ import annotations

import pathlib

import pytest

from modules import image_save, image_utils
from modules.exceptions import CaptureError, ImageSaveError
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
