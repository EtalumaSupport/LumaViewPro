# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A setting a camera restore puts back answers as its setter answers for the same value.

``restore_camera_state`` writes a snapshot's gain, exposure, black level and
frame size through the camera's private bodies, while the setters judge a
caller's value first (``_held_to_range``: a finite number of the right sign,
then held to the camera's range, the clamp told). The restore judged by a
rule of its own: a gain or exposure that was no number, or of the wrong sign,
was skipped with a warning; one past the range went to the camera as given,
so the camera model decided (the simulator refused it); a frame size was not
judged at all (0x0 raised the driver's bare ``ValueError``, 99999x99999 was
clamped with nothing told). Now a value its setter refuses is refused before
the restore writes anything, and one past the range is put back at the
range's end and told once, as unasked-for: the restore's callers are a run's
cleanup, an autofocus and a characterization, so the notice is muted with an
unattended run as the restore's refusals are.
"""

import math

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError, ValueClampedNotice


def _camera_state(cam):
    return {
        'gain_db': cam.get_gain(),
        'exposure_ms': cam.get_exposure_t(),
        'black_level': cam.get_black_level(),
        'binning': cam.get_binning_size(),
        'frame_size': dict(cam.get_frame_size()),
    }


def _clamps(centre_posts):
    return [post for post in centre_posts if post.reason == ValueClampedNotice.reason]


# Each snapshot also carries a binning the camera does not hold, so a refusal
# that came after the geometry was written would show as a changed binning.
_REFUSED = [
    ('gain_db', math.nan, 'not_a_number', 'gain_db'),
    ('gain_db', -3.0, 'out_of_range', 'gain_db'),
    ('exposure_ms', 0.0, 'out_of_range', 'exposure_ms'),
    ('black_level', -1.0, 'out_of_range', 'black_level'),
    ('frame_size', {'width': 0, 'height': 0}, 'out_of_range', 'width'),
    ('frame_size', {'width': 800, 'height': math.inf}, 'not_a_number', 'height'),
]


def test_a_value_its_setter_refuses_is_refused_before_anything_is_written(sim_camera, centre_posts):
    camera, cam = sim_camera
    before = _camera_state(cam)
    other_binning = 2 if before['binning'] != 2 else 1

    wrong = []
    for key, value, reason, argument in _REFUSED:
        snapshot = {'tag': 'test', 'binning': other_binning, key: value}
        try:
            camera.restore_camera_state(snapshot)
        except ArgumentRefusedError as refused:
            if (refused.reason, refused.argument) != (reason, argument):
                wrong.append(f'{key}={value!r}: refused as {refused.reason}/{refused.argument}')
        else:
            wrong.append(f'{key}={value!r}: not refused')
        if _camera_state(cam) != before:
            wrong.append(f'{key}={value!r}: the camera changed')
            camera.restore_camera_state(
                {'tag': 'reset', 'binning': before['binning'], 'frame_size': before['frame_size']}
            )

    assert wrong == []
    assert _clamps(centre_posts) == []


def test_a_gain_past_the_range_is_restored_at_its_end_and_told_unasked(sim_camera, centre_posts):
    camera, cam = sim_camera
    maximum = camera.setting_limits.gain_max_db

    camera.restore_camera_state({'tag': 'test', 'gain_db': maximum + 500.0})

    assert cam.get_gain() == pytest.approx(maximum)
    clamps = _clamps(centre_posts)
    assert [post.message for post in clamps] == [
        str(ValueClampedNotice([('camera', 'gain_db', maximum + 500.0, maximum)]))
    ]
    assert [post.solicited for post in clamps] == [False]


def test_a_frame_past_the_range_is_restored_at_its_end_and_told_unasked(
    sim_scope, sim_camera, centre_posts
):
    camera, cam = sim_camera
    maximum = sim_scope.capabilities.camera_limits.frame_max
    factor = cam.get_binning_size()

    camera.restore_camera_state({'tag': 'test', 'frame_size': {'width': 99999, 'height': 99999}})

    held = {'width': maximum.width // factor, 'height': maximum.height // factor}
    assert dict(cam.get_frame_size()) == held
    clamps = _clamps(centre_posts)
    assert [post.message for post in clamps] == [
        str(
            ValueClampedNotice(
                [
                    ('camera', 'frame_width', 99999.0, held['width']),
                    ('camera', 'frame_height', 99999.0, held['height']),
                ]
            )
        )
    ]
    assert [post.solicited for post in clamps] == [False]


def test_a_black_level_past_the_range_is_told_unasked(sim_camera, centre_posts):
    camera = sim_camera[0]

    camera.restore_camera_state({'tag': 'test', 'black_level': 1e9})

    clamps = _clamps(centre_posts)
    assert len(clamps) == 1 and 'black_level' in clamps[0].message
    assert [post.solicited for post in clamps] == [False]
