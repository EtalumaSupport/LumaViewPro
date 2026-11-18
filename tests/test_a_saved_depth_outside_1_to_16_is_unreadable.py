# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved file whose depth or white lies outside a frame's is unreadable, naming the value.

A frame's depth is 1 to 16 bits (``image_utils.FRAME_DEPTH_BITS``). A file
another program wrote, or a damaged one, can state any depth: the readers
resolve it from the OME ``SignificantBits``, LVP's private tag, or the pixel
type when the file states neither, and before this a depth past 16 sized
``convert_to_8bit``'s table at 2**depth entries (40 bits killed the process,
a float32 file with no tag built 32 GB), 17 to about 30 rendered dark with
no error, and a tag of 0 was read as no tag. Each is now refused where the
file is read, as ``InputFramesUnusableError('unreadable')``, so every reader
of saved pixels refuses it alike. The files here are written with tifffile
directly, as another program would write them; LVP's own writer cannot
write them.
"""

from __future__ import annotations

import datetime

import cv2
import numpy as np
import pytest
import tifffile as tf

from _lumascope.modules import image_utils
from _lumascope.modules.channel_record import release_color
from _lumascope.modules.exceptions import InputFramesUnusableError

_DEPTH_TAG = 65123
_WHITE_TAG = 65124
_PIXELS = np.zeros((8, 8), np.uint16)


def _tags(*pairs):
    return [(code, 'H', 1, value, True) for code, value in pairs]


def _depth_tag(depth):
    return lambda path: tf.imwrite(path, _PIXELS, extratags=_tags((_DEPTH_TAG, depth)))


def _ome_depth(path):
    tf.imwrite(path, _PIXELS, ome=True, metadata={'SignificantBits': 20})


def _float32_untagged(path):
    tf.imwrite(path, np.zeros((8, 8), np.float32))


def _uint32_untagged(path):
    tf.imwrite(path, np.zeros((8, 8), np.uint32))


def _white_tag(white):
    return lambda path: tf.imwrite(
        path, _PIXELS, extratags=_tags((_DEPTH_TAG, 12), (_WHITE_TAG, white))
    )


# Each case writes the file another program could, and the value the refusal names.
_UNUSABLE = {
    'depth_tag_0': (_depth_tag(0), 'depth of 0 bits'),
    'depth_tag_17': (_depth_tag(17), 'depth of 17 bits'),
    'depth_tag_20': (_depth_tag(20), 'depth of 20 bits'),
    'ome_significant_bits_20': (_ome_depth, 'depth of 20 bits'),
    'float32_with_no_tag': (_float32_untagged, 'depth of 32 bits'),
    'uint32_with_no_tag': (_uint32_untagged, 'depth of 32 bits'),
}

_UNUSABLE_WHITE = {
    'white_tag_0': (_white_tag(0), 'white of 0 bits'),
    'white_tag_above_its_depth': (_white_tag(13), 'white of 13 bits'),
}


def _assert_unreadable_naming(reader, path, named):
    with pytest.raises(InputFramesUnusableError) as refused:
        reader(path)

    assert refused.value.reason == 'unreadable'
    assert named in str(refused.value)
    assert path.name in str(refused.value)


# parametrize-justified: each case is its own file on disk, and each reader its own door.
@pytest.mark.parametrize('case', _UNUSABLE, ids=str)
@pytest.mark.parametrize(
    'reader',
    [image_utils.load_pixels, image_utils.read_tiff_depth_and_record],
    ids=lambda f: f.__name__,
)
def test_a_file_stating_a_depth_no_frame_has_is_unreadable(tmp_path, reader, case):
    write, named = _UNUSABLE[case]
    path = tmp_path / f'{case}.tif'
    write(path)

    _assert_unreadable_naming(reader, path, named)


# parametrize-justified: each case is its own file on disk. The header read
# (read_tiff_depth_and_record) returns no white, so the white is judged by the
# pixel read, the one that hands it to a renderer.
@pytest.mark.parametrize('case', _UNUSABLE_WHITE, ids=str)
def test_a_file_stating_a_white_outside_its_depth_is_unreadable(tmp_path, case):
    write, named = _UNUSABLE_WHITE[case]
    path = tmp_path / f'{case}.tif'
    write(path)

    _assert_unreadable_naming(image_utils.load_pixels, path, named)


def test_a_non_tiff_whose_pixels_are_wider_than_a_frame_is_unreadable(tmp_path):
    path = tmp_path / 'radiance.hdr'
    assert cv2.imwrite(str(path), np.zeros((8, 8, 3), np.float32))

    with pytest.raises(InputFramesUnusableError) as refused:
        image_utils.load_pixels(path)

    assert refused.value.reason == 'unreadable'
    assert 'float32' in str(refused.value)


def _lvp_save(path, depth, ome, white):
    data = np.full((8, 8), (1 << depth) - 1, np.uint8 if depth == 8 else np.uint16)
    image_utils.write_tiff(
        data,
        path,
        {
            'pixel_size_um': 0.5,
            'channel': 'BF',
            'objective': {'id': '10x'},
            'significant_bits': depth,
            'datetime': datetime.datetime(2026, 11, 17, 12, 0, 0),
        },
        ome=ome,
        color=release_color('BF'),
        significant_bits=depth,
        save_encoding='right_aligned',
        white_bits=white,
    )


def test_every_depth_and_white_lvp_saves_still_reads(tmp_path):
    saves = [
        (depth, ome, white)
        for depth in (8, 12, 16)
        for ome in (False, True)
        for white in (None, depth)
    ]
    misread = []
    for depth, ome, white in saves:
        path = tmp_path / f'{depth}_{ome}_{white}.tif'
        _lvp_save(path, depth, ome, white)
        _, read_depth, read_white = image_utils.load_pixels(path)
        header_depth, _ = image_utils.read_tiff_depth_and_record(path)
        if (read_depth, header_depth, read_white) != (depth, depth, white):
            misread.append(((depth, ome, white), (read_depth, header_depth, read_white)))

    assert not misread, misread
