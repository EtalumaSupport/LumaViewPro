# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``save_image`` refuses None as its array, before anything is written.

A capture that gets no frame raises why (``CaptureFailedError``) and never
answers None, so None reaching the save is a caller's defect, refused as an
argument, not reported as a camera that returned nothing.
"""

from __future__ import annotations

import pytest

from modules import image_save
from modules.exceptions import ArgumentRefusedError
from modules.layer_record import release_colors
from tests.frame_records import frame_record, plate
from tests.test_jpg_export import _scope_with_depth


def test_a_none_array_is_refused_as_the_wrong_kind_and_nothing_is_written(tmp_path):
    with pytest.raises(ArgumentRefusedError) as refused:
        image_save.save_image(
            _scope_with_depth(),
            None,
            save_folder=str(tmp_path),
            file_root='snap_',
            append='BF',
            channel='BF',
            false_color_on=False,
            tail_id_mode=None,
            output_format='TIFF',
            save_encoding='8bit',
            significant_bits=8,
            objective_id='4x Oly',
            frame_record=frame_record(),
            labware=plate(),
            well_label=None,
            channel_colors=release_colors(),
        )

    assert (refused.value.reason, refused.value.argument) == ('wrong_kind', 'array')
    assert list(tmp_path.iterdir()) == []
