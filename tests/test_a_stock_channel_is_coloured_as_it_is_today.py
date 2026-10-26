# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each colour writer colours a stock channel as it does today.

A transmitted channel (BF, PC, DF) is grey; LED 0's channel and Lumi are the
image on the blue plane, LED 1's on the green, LED 2's on the red. Every writer that colours
a frame says so: false colour, Quick Enhance's derived image, the TIFF's
photometric and colour map (plain and OME), the JPEG the display exports and
the video. These pins hold the stock channels' output while colour stops
being chosen by key and becomes a value each writer multiplies by (W1c part
B): nothing a user sees for the shipped channels may change.

The JPEG and the video are lossy, so they are held to the expected colour
within a tolerance, never to their bytes, which differ between codec builds.
"""

from __future__ import annotations

import av
import cv2
import numpy as np
import tifffile

from modules import image_utils
from modules.layer_record import release_color
from modules.quick_enhance import QuickEnhancer
from modules.video_writer import VideoWriter
from modules.layer_record import release_colors
from tests.stock_channels import LED0, LED1, LED2

# None is grey.
_COLOUR = {
    'BF': None,
    'PC': None,
    'DF': None,
    LED0: (0, 0, 1),
    LED1: (0, 1, 0),
    LED2: (1, 0, 0),
    'Lumi': (0, 0, 1),
}
_MONO = (np.arange(64 * 48, dtype=np.uint32) * 37 % 256).astype(np.uint8).reshape(48, 64)
_METADATA = {
    'pixel_size_um': 0.5,
    'objective': '20x Oly',
    'exposure_time_ms': 2.0,
    'gain_db': 1.0,
    'illumination_ma': 5.0,
    'z_pos_um': 5000.0,
    'plate_pos_mm': {'x': 1.0, 'y': 2.0},
    'datetime': '2026:10:13 12:00:00',
    'camera_make': 'Test',
    'microscope': 'TestScope',
    'well_label': '',
}
# The lossy writers' frame: smooth, since chroma subsampling smears a dense
# pattern's colour by about 18 levels where a gradient loses under 2
# (measured), and a wrong colour is off by about 85.
_SMOOTH = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (48, 1))
# Mean absolute difference, in 8-bit levels, a lossy encode may add.
_LOSSY_LEVELS = 3.0


def _rgb(mono: np.ndarray, colour: tuple[int, int, int] | None) -> np.ndarray:
    if colour is None:
        return np.stack([mono] * 3, axis=-1)
    return np.stack([mono * c for c in colour], axis=-1).astype(np.uint8)


def _far(got: np.ndarray, want: np.ndarray) -> bool:
    return float(np.abs(got.astype(float) - want.astype(float)).mean()) > _LOSSY_LEVELS


def test_false_colour_puts_each_channel_on_its_plane_or_grey():
    wrong = []
    for channel, colour in _COLOUR.items():
        rgb = image_utils.mono_to_rgb_falsecolor(_MONO, release_color(channel))
        if not np.array_equal(rgb, _rgb(_MONO, colour)):
            wrong.append(f'mono_to_rgb_falsecolor {channel}')
        added = image_utils.add_false_color(_MONO.copy(), release_color(channel))
        want = _MONO if colour is None else _rgb(_MONO, colour)
        if not np.array_equal(added, want):
            wrong.append(f'add_false_color {channel}')
    assert wrong == []


def test_quick_enhance_colours_a_fluorescence_image_and_leaves_a_transmitted_one_grey():
    wrong = []
    for channel, colour in _COLOUR.items():
        got = QuickEnhancer.colorize_mono_for_visual_output(
            _MONO, channel, channel_colors=release_colors()
        )
        want = _MONO if colour is None else _rgb(_MONO, colour)
        if not np.array_equal(got, want):
            wrong.append(channel)
    assert wrong == []


def test_a_tiff_carries_the_channels_colour_map_or_none(tmp_path):
    wrong = []
    for channel, colour in _COLOUR.items():
        for ome in (False, True):
            path = tmp_path / f'{channel}_{ome}.tiff'
            image_utils.write_tiff(
                data=_MONO.copy(),
                file_loc=path,
                metadata={**_METADATA, 'channel': channel},
                ome=ome,
                color=release_color(channel),
                significant_bits=8,
                save_encoding='right_aligned',
                white_bits=None,
            )
            with tifffile.TiffFile(path) as written:
                page = written.pages[0]
                pixels, photometric, colour_map = (
                    page.asarray(),
                    int(page.photometric),
                    page.colormap,
                )
            if not np.array_equal(pixels, _MONO):
                wrong.append(f'{channel} ome={ome}: pixels')
            if colour is None:
                if photometric != 1 or colour_map is not None:
                    wrong.append(f'{channel} ome={ome}: not grey ({photometric})')
            elif photometric != 3 or [int(v) for v in colour_map[:, -1]] != [
                65535 * c for c in colour
            ]:
                wrong.append(f'{channel} ome={ome}: palette {photometric}')
    assert wrong == []


def test_the_display_jpeg_shows_the_channels_colour():
    wrong = []
    for channel, colour in _COLOUR.items():
        encoded = image_utils.encode_display_jpg(_SMOOTH.copy(), release_color(channel), 8)
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if _far(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), _rgb(_SMOOTH, colour)):
            wrong.append(channel)
    assert wrong == []


def test_a_video_shows_the_channels_colour(tmp_path):
    wrong = []
    frames = [np.roll(_SMOOTH, i, axis=1).copy() for i in range(3)]
    for channel, colour in _COLOUR.items():
        writer = VideoWriter(
            tmp_path / f'{channel}.mp4',
            fps=10,
            width=64,
            height=48,
            color=release_color(channel),
        )
        for frame in frames:
            writer.add_frame(frame, significant_bits=8)
        writer.close()
        with av.open(str(writer.output_path)) as container:
            decoded = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
        if len(decoded) != len(frames) or any(
            _far(got, _rgb(want, colour)) for got, want in zip(decoded, frames, strict=True)
        ):
            wrong.append(channel)
    assert wrong == []


def test_a_16_bit_imagej_tiff_carries_the_channels_lut(tmp_path):
    """A 16-bit frame is stored grey, and FIJI is told its colour by the LUT."""
    mono16 = _MONO.astype(np.uint16) * 16
    ramp = np.arange(256)
    wrong = []
    for channel, colour in _COLOUR.items():
        path = tmp_path / f'{channel}.tiff'
        image_utils.write_tiff(
            data=mono16.copy(),
            file_loc=path,
            metadata={**_METADATA, 'channel': channel},
            ome=False,
            color=release_color(channel),
            significant_bits=12,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tifffile.TiffFile(path) as written:
            page = written.pages[0]
            pixels, photometric = page.asarray(), int(page.photometric)
            imagej = written.imagej_metadata
        # One LUT is read back as the (3, 256) array itself, not a list of one.
        lut = np.asarray(imagej['LUTs']).reshape(3, 256)
        want = np.stack([ramp * (1 if colour is None else c) for c in (colour or (1, 1, 1))])
        if not np.array_equal(pixels, mono16) or photometric != 1:
            wrong.append(f'{channel}: pixels or photometric {photometric}')
        if not np.array_equal(lut, want):
            wrong.append(f'{channel}: lut')
        if (imagej.get('mode') == 'color') != (colour is not None):
            wrong.append(f'{channel}: mode {imagej.get("mode")}')
    assert wrong == []


def test_a_16_bit_false_colour_tiff_bakes_the_channels_colour(tmp_path):
    """Saved with the RGB encoding, a fluorescence frame is widened onto its
    plane after its payload fills the container; a transmitted one stays grey."""
    mono16 = _MONO.astype(np.uint16) * 16
    filled = mono16 << 4
    wrong = []
    for channel, colour in _COLOUR.items():
        path = tmp_path / f'{channel}.tiff'
        image_utils.write_tiff(
            data=mono16.copy(),
            file_loc=path,
            metadata={**_METADATA, 'channel': channel},
            ome=False,
            color=release_color(channel),
            significant_bits=12,
            save_encoding='rgb',
            white_bits=None,
        )
        pixels = tifffile.imread(path)
        want = filled if colour is None else np.stack([filled * c for c in colour], axis=-1)
        if not np.array_equal(pixels, want):
            wrong.append(channel)
    assert wrong == []


def test_a_hyperstack_names_each_channels_colour_in_its_ome_metadata(tmp_path):
    """OME Channel.Color is the signed RGBA of the channel's colour, white for grey."""
    reference = tmp_path / 'reference.tiff'
    image_utils.write_tiff(
        data=_MONO.copy(),
        file_loc=reference,
        metadata={**_METADATA, 'channel': 'BF'},
        ome=True,
        color=release_color('BF'),
        significant_bits=8,
        save_encoding='right_aligned',
        white_bits=None,
    )

    def rgba(colour):
        r, g, b = (255 * c for c in (colour or (1, 1, 1)))
        unsigned = (r << 24) | (g << 16) | (b << 8) | 0xFF
        return unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned

    metadata = image_utils.build_hyperstack_output_metadata(
        reference,
        channel_names=list(_COLOUR),
        channel_colors=[release_color(channel) for channel in _COLOUR],
        plane_positions={
            'PositionX': [0.0] * len(_COLOUR),
            'PositionY': [0.0] * len(_COLOUR),
            'PositionZ': [0.0] * len(_COLOUR),
            'DeltaT': None,
        },
        significant_bits=8,
        pixel_size_um=0.5,
    )
    assert metadata['Channel']['Color'] == [rgba(c) for c in _COLOUR.values()]


def test_a_stock_false_colour_image_reads_back_as_its_mono():
    """The composite and stack inputs collapse a baked image to one plane."""
    wrong = [
        channel
        for channel, colour in _COLOUR.items()
        if not np.array_equal(image_utils.rgb_image_to_gray(_rgb(_MONO, colour)), _MONO)
    ]
    assert wrong == []
