# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The overlays a still is saved with are rendered below the GUI.

A headless capture asking for an overlay copy must get the pixels the
Capture button gets, so the renderers live in modules/. Drawing the
crosshairs must not touch the frame it was given: with no bullseye drawn
first, that frame is the raw capture itself.
"""

import numpy as np

from modules import capture_overlays


def test_crosshairs_leave_the_input_unmarked():
    raw = np.zeros((64, 80), dtype=np.uint8)

    marked = capture_overlays.add_crosshairs(raw)

    assert not raw.any(), 'the raw frame was drawn into'
    assert marked is not raw
    assert marked[:, 39:41].min() == 255, 'vertical line, 2 px, at the centre column'
    assert marked[31:33, :].min() == 255, 'horizontal line, 2 px, at the centre row'


def test_crosshairs_on_a_colour_image_mark_every_channel():
    raw = np.zeros((64, 80, 3), dtype=np.uint8)

    marked = capture_overlays.add_crosshairs(raw)

    assert not raw.any()
    assert marked[:, 39:41, :].min() == 255
    assert marked.shape == raw.shape


def test_bullseye_bands():
    levels = np.array([[0, 10, 130, 140, 250]], dtype=np.uint8)

    rgb = capture_overlays.transform_to_bullseye(levels)

    assert rgb.shape == (1, 5, 3)
    assert rgb[0].tolist() == [
        [0, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 0, 0],
        [255, 0, 0],
    ]
