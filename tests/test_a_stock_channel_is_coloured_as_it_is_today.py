# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each colour writer colours a stock channel as it does today.

A transmitted channel (BF, PC, DF) is grey; Blue and Lumi are the image on
the blue plane, Green on the green, Red on the red. Every writer that colours
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
from modules.quick_enhance import QuickEnhancer
from modules.video_writer import VideoWriter

# None is grey.
_COLOUR = {
    'BF': None,
    'PC': None,
    'DF': None,
    'Blue': (0, 0, 1),
    'Green': (0, 1, 0),
    'Red': (1, 0, 0),
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
        rgb = image_utils.mono_to_rgb_falsecolor(_MONO, channel)
        if not np.array_equal(rgb, _rgb(_MONO, colour)):
            wrong.append(f'mono_to_rgb_falsecolor {channel}')
        added = image_utils.add_false_color(_MONO.copy(), channel)
        want = _MONO if colour is None else _rgb(_MONO, colour)
        if not np.array_equal(added, want):
            wrong.append(f'add_false_color {channel}')
    assert wrong == []


def test_quick_enhance_colours_a_fluorescence_image_and_leaves_a_transmitted_one_grey():
    wrong = []
    for channel, colour in _COLOUR.items():
        got = QuickEnhancer.colorize_mono_for_visual_output(_MONO, channel)
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
                color=channel,
                significant_bits=8,
                save_encoding='right_aligned',
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
        encoded = image_utils.encode_display_jpg(_SMOOTH.copy(), channel, 8)
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if _far(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), _rgb(_SMOOTH, colour)):
            wrong.append(channel)
    assert wrong == []


def test_a_video_shows_the_channels_colour(tmp_path):
    wrong = []
    frames = [np.roll(_SMOOTH, i, axis=1).copy() for i in range(3)]
    for channel, colour in _COLOUR.items():
        writer = VideoWriter(
            tmp_path / f'{channel}.mp4', fps=10, width=64, height=48, color=channel
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
