# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every TIFF LVP writes is compressed one way: deflate level 1, the standard
code 8, with the horizontal predictor on every file but a 16-bit RGB one, which
ImageJ's File > Open refuses with it.

The codec belongs to the two writers, ``image_utils.write_tiff`` and
``image_utils.write_hyperstack_tiff`` (``image_utils._tiff_codec``); no caller
chooses it. Each case writes through the surface its producer uses -- a
recorded frame through ``image_save.write_recorded_frame``, a still through
``image_save.save_image``, a post-processing output through ``write_tiff``, a
hyperstack through ``StackBuilder`` -- and reads the file back through
tifffile: every page deflate, with the predictor but on a 16-bit RGB file, the
pixels unchanged.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from _lumascope.modules import image_mode, image_save, image_utils
from _lumascope.modules.channel_record import release_color, release_colors
from _lumascope.modules.stack_builder import StackBuilder
from tests.frame_records import frame_record, plate, recorded_record, write_recorded
from tests.stock_channels import LED1

_RNG = np.random.default_rng(0)

_THE_CODEC = (tf.COMPRESSION.ADOBE_DEFLATE, tf.PREDICTOR.HORIZONTAL)
_NO_PREDICTOR = (tf.COMPRESSION.ADOBE_DEFLATE, tf.PREDICTOR.NONE)


def _codec_and_pixels(path: pathlib.Path) -> tuple[set, np.ndarray]:
    with tf.TiffFile(str(path)) as tif:
        codecs = {(page.compression, page.predictor) for page in tif.pages}
        pixels = tif.asarray()
    return codecs, pixels


def _mono12(shape=(64, 64)) -> np.ndarray:
    return _RNG.integers(0, 4096, shape).astype(np.uint16)


def _mono8(shape=(64, 64)) -> np.ndarray:
    return _RNG.integers(0, 256, shape).astype(np.uint8)


# Each recorded frame's container and depth: OME or plain (ImageJ at 16 bits,
# a palette at 8), as its file name states.
_RECORDED = {
    'ome_8bit': ('frame.ome.tiff', _mono8, '8bit'),
    'ome_12bit': ('frame.ome.tiff', _mono12, 'right_aligned'),
    'plain_8bit_palette': ('frame.tiff', _mono8, '8bit'),
    'plain_12bit_imagej': ('frame.tiff', _mono12, 'right_aligned'),
}

# Each still's container and shape, by the save's output format and encoding,
# and the codec its file is written with: a false-colour still is 16-bit RGB.
_STILLS = {
    'ome_12bit': (image_mode.OutputFormat.OME_TIFF, _mono12, 'right_aligned', 12, _THE_CODEC),
    'imagej_12bit': (image_mode.OutputFormat.TIFF, _mono12, 'right_aligned', 12, _THE_CODEC),
    'plain_8bit_palette': (image_mode.OutputFormat.TIFF, _mono8, '8bit', 8, _THE_CODEC),
    'plain_false_colour_rgb': (image_mode.OutputFormat.TIFF, _mono12, 'rgb', 12, _NO_PREDICTOR),
    'ome_false_colour_rgb': (image_mode.OutputFormat.OME_TIFF, _mono12, 'rgb', 12, _NO_PREDICTOR),
}


def test_every_recorded_frame_is_deflate_with_the_predictor(sim_scope, tmp_path):
    """One test over the recorded-frame containers; the failure names each
    container whose codec or pixels are wrong."""
    offenders = []
    for case, (name, make, encoding) in _RECORDED.items():
        frame = make()
        path = tmp_path / case / name
        path.parent.mkdir()
        write_recorded(
            sim_scope,
            frame,
            path,
            channel=LED1,
            save_encoding=encoding,
            false_color_on=True,
            record=recorded_record(frame_significant_bits=8 if frame.dtype == np.uint8 else 12),
        )
        codecs, pixels = _codec_and_pixels(path)
        if codecs != {_THE_CODEC} or not np.array_equal(pixels, frame):
            offenders.append(
                f'{case}: codecs {codecs}, pixels equal {np.array_equal(pixels, frame)}'
            )
    assert not offenders, offenders


def test_every_still_is_deflate_with_the_predictor_but_a_16_bit_rgb_one(sim_scope, tmp_path):
    """One test over the still containers; a false-colour still is 16-bit RGB,
    written without the predictor in either container, and its pixels are the
    channel's colour baked in, so its shape is asserted; the others' pixels
    equal the frame."""
    offenders = []
    for case, (output_format, make, encoding, bits, codec) in _STILLS.items():
        frame = make()
        path = image_save.save_image(
            sim_scope,
            frame.copy(),
            save_folder=str(tmp_path),
            file_root=f'{case}_',
            append='a',
            tail_id_mode=None,
            channel=LED1,
            false_color_on=True,
            channel_colors=release_colors(),
            save_encoding=encoding,
            output_format=output_format,
            significant_bits=bits,
            white_bits=None,
            objective_id=sim_scope.runtime_state.get_current_objective_id(),
            frame_record=frame_record(frame_significant_bits=bits),
            labware=plate(),
            well_label=None,
            fallback_focus=None,
        )
        codecs, pixels = _codec_and_pixels(path)
        if encoding == 'rgb':
            pixels_right = pixels.shape == (*frame.shape, 3) and pixels.dtype == np.uint16
        else:
            pixels_right = np.array_equal(pixels, frame)
        if codecs != {codec} or not pixels_right:
            offenders.append(f'{case}: codecs {codecs}, pixels right {pixels_right}')
    assert not offenders, offenders


def test_a_post_processing_output_is_deflate_with_the_predictor(tmp_path):
    """A composite: an 8-bit RGB image of remapped values, written plain, keeps
    the predictor; only a 16-bit RGB file goes without."""
    composite = _RNG.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    path = tmp_path / 'composite.tiff'
    image_utils.write_tiff(
        data=composite,
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'Composite',
            'significant_bits': 8,
            'datetime': recorded_record().captured_at.at,
        },
        ome=False,
        color=release_color(LED1),
        significant_bits=8,
        save_encoding='8bit',
        white_bits=None,
    )
    codecs, pixels = _codec_and_pixels(path)
    assert codecs == {_THE_CODEC}
    assert np.array_equal(pixels, composite)


def test_a_hyperstack_is_deflate_with_the_predictor(sim_scope, tmp_path):
    frames = []
    rows = []
    for z_idx in range(2):
        name = f'frame_z{z_idx}.tiff'
        frame = _mono12((16, 16))
        frames.append(frame)
        write_recorded(
            sim_scope, frame, tmp_path / name, channel=LED1, save_encoding='right_aligned'
        )
        rows.append(
            {
                'Filepath': name,
                'channel': LED1,
                'Scan Count': 0,
                'Z-Slice': z_idx,
                'X': 10.0,
                'Y': 20.0,
                'Z': float(z_idx) * 10,
            }
        )
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=pd.DataFrame(rows),
        output_file_loc=pathlib.Path('out.ome.tiff'),
        save_encoding='right_aligned',
        channel_colors=release_colors(),
    )
    assert result['status'], result.get('error')

    codecs, pixels = _codec_and_pixels(tmp_path / 'out.ome.tiff')
    assert codecs == {_THE_CODEC}
    assert np.array_equal(pixels.reshape(2, 16, 16), np.stack(frames))


def test_a_caller_cannot_choose_the_hyperstack_codec(tmp_path):
    plane = np.zeros((4, 4), dtype=np.uint8)
    reference = tmp_path / 'ref.tiff'
    tf.imwrite(reference, plane)
    metadata = image_utils.build_hyperstack_output_metadata(
        input_records=[image_utils.read_file_record(reference)],
        channel_names=[LED1],
        channel_colors=[release_color(LED1)],
        plane_positions={
            'PositionX': [0.0],
            'PositionY': [0.0],
            'PositionZ': [0.0],
            'DeltaT': None,
        },
        significant_bits=8,
        pixel_size_um=None,
    )
    with pytest.raises(TypeError, match='compression'):
        image_utils.write_hyperstack_tiff(
            planes=iter([plane]),
            file_loc=tmp_path / 'chosen.ome.tiff',
            shape=(1, 1, 1, 4, 4),
            dtype=np.uint8,
            hyperstack_metadata=metadata,
            hyperstack_options={'photometric': 'minisblack', 'compression': 'lzw'},
        )
    assert not (tmp_path / 'chosen.ome.tiff').exists()
