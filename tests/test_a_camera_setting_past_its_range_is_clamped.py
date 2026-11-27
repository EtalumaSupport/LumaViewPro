# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A gain, exposure or black level written past the camera's range is set at
the range's end, answered, and told once; a value no range makes sense of is
refused.

Before, the camera's own setters refused a value past the range while a
channel's stored value past it was capped at apply, so one number met two
rules by the door it came in, and 3000 ms typed for a 1000 ms camera ran at
1000 with only a log line. Now every raw write holds its value to the range
the camera declares (``setting_limits``) before anything is sent, the same on
every camera, answers the value in effect, and reports one
``ValueClampedNotice`` naming each clamped value once the write has landed.
The sign is judged first: the camera's floor would otherwise turn an
exposure of no time or a negative gain into a value nobody asked for. A
limit the camera does not declare narrows nothing.
"""

import math

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError, ValueClampedNotice
from lumascope import CameraSettingsInEffect
from tests.scope_fakes import set_camera_limits


def _clamp_notices(centre_posts):
    return [post.message for post in centre_posts if post.reason == ValueClampedNotice.reason]


def test_a_gain_above_the_maximum_is_set_at_it_and_told(sim_camera, centre_posts):
    camera, cam = sim_camera
    maximum = camera.setting_limits.gain_max_db

    assert camera.set_gain_db(maximum + 5.0) == pytest.approx(maximum)

    assert cam.get_gain() == pytest.approx(maximum)
    assert camera.gain_db == pytest.approx(maximum)
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('camera', 'gain_db', maximum + 5.0, maximum)]))
    ]


def test_a_gain_below_a_declared_floor_is_raised_to_it_and_told(
    sim_camera, centre_posts, monkeypatch
):
    camera, cam = sim_camera
    set_camera_limits(monkeypatch, camera._scope, gain_min_db=2.0)

    assert camera.set_gain_db(1.0) == pytest.approx(2.0)

    assert cam.get_gain() == pytest.approx(2.0)
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('camera', 'gain_db', 1.0, 2.0)]))
    ]


def test_an_exposure_above_the_maximum_is_set_at_it_and_told(sim_camera, centre_posts):
    camera, cam = sim_camera
    maximum = camera.setting_limits.exposure_max_ms

    assert camera.set_exposure_ms(maximum * 3) == pytest.approx(maximum)

    assert cam.get_exposure_t() == pytest.approx(maximum)
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('camera', 'exposure_ms', maximum * 3, maximum)]))
    ]


def test_an_exposure_under_the_floor_is_raised_to_it_and_told(sim_camera, centre_posts):
    camera, cam = sim_camera
    floor = camera.setting_limits.exposure_min_ms
    assert floor is not None and floor > 0.001, 'the test needs a camera with a floor'

    assert camera.set_exposure_ms(0.001) == pytest.approx(floor)

    assert cam.get_exposure_t() == pytest.approx(floor)
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('camera', 'exposure_ms', 0.001, floor)]))
    ]


def test_an_undeclared_floor_narrows_nothing(sim_camera, centre_posts, monkeypatch):
    """A camera that declares no floor: the camera takes what it is given,
    and nothing is told."""
    camera, cam = sim_camera
    monkeypatch.setattr(cam, '_EXPOSURE_FLOOR_MS', dict.fromkeys(cam._EXPOSURE_FLOOR_MS))
    set_camera_limits(monkeypatch, camera._scope, exposure_min_ms=None)

    camera.set_exposure_ms(0.001)

    assert cam.get_exposure_t() == 0.001
    assert _clamp_notices(centre_posts) == []


def test_a_value_in_range_is_applied_and_nothing_is_told(sim_camera, centre_posts):
    camera, _cam = sim_camera
    maximum = camera.setting_limits.gain_max_db

    assert camera.set_gain_db(maximum) == pytest.approx(maximum)

    assert _clamp_notices(centre_posts) == []


def test_the_layer_apply_clamps_both_values_in_one_notice(sim_scope, centre_posts):
    """``apply`` is a public door to the same gain and exposure, so it holds
    them to the same range, answers the values in effect as its record, and
    tells both in the one notice of its call."""
    camera = sim_scope.camera
    limits = camera.setting_limits

    in_effect = camera.apply(limits.gain_max_db + 5.0, limits.exposure_max_ms * 3, channel='BF')

    assert in_effect == CameraSettingsInEffect(
        gain_db=pytest.approx(limits.gain_max_db),
        exposure_ms=pytest.approx(limits.exposure_max_ms),
    )
    assert _clamp_notices(centre_posts) == [
        str(
            ValueClampedNotice(
                [
                    ('camera', 'gain_db', limits.gain_max_db + 5.0, limits.gain_max_db),
                    ('camera', 'exposure_ms', limits.exposure_max_ms * 3, limits.exposure_max_ms),
                ]
            )
        )
    ]


def test_a_black_level_above_the_range_is_set_at_it_and_told(sim_scope, centre_posts):
    camera = sim_scope.camera
    _low, high = camera.read_black_level_range()

    assert camera.set_black_level(high + 10.0) == high

    assert camera.read_black_level() == high
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('camera', 'black_level', high + 10.0, high)]))
    ]


def test_a_value_no_range_makes_sense_of_is_refused_and_nothing_is_sent(
    sim_scope, centre_posts, monkeypatch
):
    """Each raw door refuses a value that is not a finite number, an exposure
    of zero or below, and a gain or black level below zero, before the
    camera is asked; the camera's floor would otherwise raise each to a
    value never asked for."""
    camera = sim_scope.camera
    driver = sim_scope._camera_driver
    sent = []
    for node in ('gain', 'exposure_t', 'set_black_level'):
        monkeypatch.setattr(driver, node, lambda v, node=node: sent.append((node, v)) or v)
    cases = (
        ('set_exposure_ms', lambda: camera.set_exposure_ms(0.0), 'out_of_range'),
        ('set_exposure_ms', lambda: camera.set_exposure_ms(-5.0), 'out_of_range'),
        ('set_exposure_ms', lambda: camera.set_exposure_ms(math.nan), 'not_a_number'),
        ('set_gain_db', lambda: camera.set_gain_db(-1.0), 'out_of_range'),
        ('set_gain_db', lambda: camera.set_gain_db(math.inf), 'not_a_number'),
        ('apply', lambda: camera.apply(5.0, 0.0, channel='BF'), 'out_of_range'),
        ('apply', lambda: camera.apply(-1.0, 20.0, channel='BF'), 'out_of_range'),
        ('set_black_level', lambda: camera.set_black_level(-1.0), 'out_of_range'),
    )
    wrong = []
    for member, write, reason in cases:
        try:
            write()
        except ArgumentRefusedError as refused:
            if refused.reason != reason:
                wrong.append((member, refused.reason))
        else:
            wrong.append((member, 'not refused'))

    assert wrong == []
    assert sent == [], 'a refused value reaches no camera'
    assert _clamp_notices(centre_posts) == []


@pytest.mark.parametrize(
    ('node', 'setter', 'past'),
    [('gain', 'set_gain_db', 5.0), ('exposure', 'set_exposure_ms', 3.0)],
    ids=['gain', 'exposure'],
)
def test_a_clamped_value_the_camera_took_but_could_not_read_back_is_told(
    sim_scope, centre_posts, node, setter, past
):
    # parametrize-justified: each fault is armed on its own camera node.
    # The write lands and answers nothing, and the read after it fails, so
    # the value in effect is not known: the clamp was still sent, and is told.
    camera = sim_scope.camera
    limits = camera.setting_limits
    maximum = limits.gain_max_db if node == 'gain' else limits.exposure_max_ms
    asked = maximum + past if node == 'gain' else maximum * past
    sim_scope.simulation.refuse_camera_write(node, 'unreported')
    sim_scope.simulation.refuse_camera_read(node, 'failed')

    assert getattr(camera, setter)(asked) is None

    setting = 'gain_db' if node == 'gain' else 'exposure_ms'
    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([('camera', setting, asked, maximum)]))
    ]
