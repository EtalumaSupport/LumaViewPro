# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every frame the API hands out, and every file it saves, has row 0 at the top of the picture.

The camera drivers deliver that order (at the sensor where it can reverse
its rows, else in the copy they already make) and nothing above them turns
a frame. Before, every driver delivered row 0 at the bottom; the saver and
the recorder each flipped it, and every other client -- ``POST
/api/v1/still``, the REST live view, the SDK, Micro-Manager -- got it upside
down against LumaViewPro's window and its files.

On the simulator: one field of the simulated specimen, whose top and bottom
differ, is the same way up in ``capture_still().image``, the TIFF a manual
capture saves, ``/still``'s pixels and a decoded ``/live.jpg``. Which way up
the frame itself is, against the stage, is
``test_a_stage_move_shifts_the_image_by_the_distance_over_the_pixel_size_against_the_stage``
(a move to higher Y carries the sample toward the bottom); the simulated
FX2 and the simulated Basler deliver one picture the same way up
(``test_the_classic_and_the_ls850_image_one_sample``).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tifffile

from _lumascope.drivers.simulated_camera import SimulatedCamera, SimulatedReverseNode
from tests.camera_fakes import grab_a_frame_made_after_now
from tests.test_a_simulated_frame_shows_the_light_and_the_field_it_was_made_under import (
    _lit_frame,
    _session,
)
from tests.test_the_live_view_shows_the_newest_frame import _decoded, _serving

# Small enough that the simulator's one-pixel drift and the JPEG's loss leave
# a picture's likeness to itself near 1, and to itself upside down near 0.
_COMPARED = (48, 48)


def _likeness(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation of two pictures at one small size: 1 the same, about 0 unrelated."""
    a, b = (
        cv2.resize(p.astype(np.float32), _COMPARED, interpolation=cv2.INTER_AREA) for p in (a, b)
    )
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def _same_way_up(output: np.ndarray, frame: np.ndarray) -> str | None:
    """None when ``output`` shows ``frame`` the way up it is; else why not."""
    upright, turned = _likeness(output, frame), _likeness(output, frame[::-1])
    if upright > 0.8 and turned < 0.5:
        return None
    return f'likeness {upright:.2f} upright, {turned:.2f} upside down'


@pytest.mark.slow
def test_the_api_s_frame_the_saved_file_and_the_rest_outputs_are_one_way_up():
    with _session('LS850', homed=True) as session:
        # The sample brightfield-lit, as the stage-move test lights it.
        frame = _lit_frame(session.scope)
        # BF is shown, and its stored false colour is off: the file is grey.
        (still,) = session.manual_capture.capture().result(timeout=30)
        outputs = {'the saved TIFF': tifffile.imread(still.file)}
        with _serving(session) as (_app, _server, client):
            answer = client.post('/api/v1/still')
            outputs['/live.jpg'] = _decoded(client.get('/api/v1/live.jpg').content)
        length = int(answer.headers['X-Record-Length'])
        shape = tuple(int(n) for n in answer.headers['X-Image-Shape'].split(','))
        outputs['/still'] = np.frombuffer(
            answer.content[length:], dtype=np.dtype(answer.headers['X-Image-Dtype'])
        ).reshape(shape)

    wrong = {name: why for name, image in outputs.items() if (why := _same_way_up(image, frame))}
    assert not wrong, wrong


def test_a_body_that_owes_a_reverse_delivers_the_frame_a_body_that_takes_it_does(monkeypatch):
    """The copy turns what the sensor did not, once: a read-only reverse that
    already reverses is not turned again."""
    bodies = {
        'owes ReverseY': {'ReverseY': None, 'ReverseX': SimulatedReverseNode()},
        'owes ReverseX': {'ReverseY': SimulatedReverseNode(), 'ReverseX': None},
        'a read-only ReverseY that reverses': {
            'ReverseY': SimulatedReverseNode(writable=False, value=True),
            'ReverseX': SimulatedReverseNode(),
        },
    }
    taking = _a_frame()
    wrong = {}
    for name, nodes in bodies.items():
        monkeypatch.setattr(SimulatedCamera, 'REVERSE_NODES', nodes)
        frame = _a_frame()
        upright, mirrored = _likeness(frame, taking), _likeness(frame, taking[:, ::-1])
        turned = _likeness(frame, taking[::-1])
        if not (upright > 0.8 and mirrored < 0.5 and turned < 0.5):
            wrong[name] = f'{upright:.2f} as taken, {turned:.2f} turned, {mirrored:.2f} mirrored'
    assert not wrong, wrong


def _a_frame() -> np.ndarray:
    camera = SimulatedCamera(width=192, height=128)
    try:
        camera.set_outside_light(40.0)
        camera.start_grabbing()
        return grab_a_frame_made_after_now(camera).image.copy()
    finally:
        camera.disconnect()
