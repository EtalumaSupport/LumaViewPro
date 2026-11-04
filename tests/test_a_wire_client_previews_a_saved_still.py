# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wire client previews a saved still by the name the live folder gives it.

``GET /api/v1/previews/<name>?max_width=`` answers the file downscaled, as a
JPEG for display: 8-bit grey, as the live view is, rendered from the file's
own depth and white, so a summed still is as bright as it is shown. The name
goes through ``ScopeSession.live_folder_path``, as a download's does.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import modules.image_mode as image_mode
from modules import image_utils
from rest.app import build_app
from tests.test_manual_capture_member import _capture, _open_session, _settings
from tests.stock_channels import LED0


def _jpeg(answer) -> np.ndarray:
    assert answer.status_code == 200, answer.text
    assert answer.headers['content-type'] == 'image/jpeg'
    return cv2.imdecode(np.frombuffer(answer.content, np.uint8), cv2.IMREAD_UNCHANGED)


def _name(session, still) -> str:
    return still.file.relative_to(session.get_setting('live_folder')).as_posix()


@pytest.fixture
def summed(tmp_path):
    """A lit 12-bit frame summed three times, saved, with a client."""
    settings = _settings(tmp_path)
    settings['image_mode'] = image_mode.ImageMode.SCIENTIFIC
    settings['BF']['sum'] = 3
    with _open_session(settings) as session:
        session.scope.motion.move_z(session.scope._camera_driver.get_nominal_focal_z())
        session.scope.illumination.led_on('BF', 200.0)
        stills = _capture(session)
        with TestClient(build_app(session)) as client:
            yield session, client, stills


def test_a_summed_still_previews_as_bright_as_it_is_shown(summed):
    session, client, (still,) = summed
    pixels, bits, _ = image_utils.load_pixels(still.file)
    width = pixels.shape[1]

    shown = _jpeg(client.get(f'/api/v1/previews/{_name(session, still)}?max_width={width}'))

    as_shown = image_utils.convert_sum_to_8bit(
        pixels, still.frame.frames_summed, still.frame.frame_significant_bits
    )
    against_the_sums_depth = image_utils.convert_to_8bit(pixels, bits)
    assert shown.shape == as_shown.shape
    assert abs(float(shown.mean()) - float(as_shown.mean())) < 2.0
    assert float(as_shown.mean()) - float(against_the_sums_depth.mean()) > 5.0, (
        'the sum is no brighter shown than at its own depth, so this cannot tell them apart'
    )


def test_the_preview_is_no_wider_than_asked_and_half_the_image_by_default(summed):
    session, client, (still,) = summed
    width = image_utils.load_pixels(still.file)[0].shape[1]
    name = _name(session, still)

    narrow = _jpeg(client.get(f'/api/v1/previews/{name}?max_width=100'))
    default = _jpeg(client.get(f'/api/v1/previews/{name}'))

    assert narrow.shape[1] == 100
    assert default.shape[1] == width // 2


def test_a_colour_file_previews_grey(tmp_path):
    # A JPG still bakes its channel's false colour into its pixels.
    with _open_session(_settings(tmp_path, live_format='JPG')) as session:
        session.show_channel(LED0)
        (still,) = _capture(session, false_color_on=True)
        assert image_utils.load_pixels(still.file)[0].ndim == 3, 'the JPG is colour'
        with TestClient(build_app(session)) as client:
            shown = _jpeg(client.get(f'/api/v1/previews/{_name(session, still)}?max_width=200'))

    assert shown.ndim == 2


def test_a_name_that_is_no_file_or_no_image_is_refused(summed):
    session, client, _ = summed
    (pathlib.Path(session.get_setting('live_folder')) / 'notes.txt').write_bytes(b'not an image')

    missing = client.get('/api/v1/previews/nothing_here.tiff')
    unreadable = client.get('/api/v1/previews/notes.txt')
    outside = client.get('/api/v1/previews/..%2Fescape.tiff')

    assert missing.status_code == 404
    assert unreadable.json()['reason'] == 'unreadable', unreadable.json()
    assert outside.json()['reason'] == 'outside_live_folder', outside.json()
