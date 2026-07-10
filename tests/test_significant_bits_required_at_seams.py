"""Depth-drop seams must REQUIRE significant_bits, not default it.

Each function below could formerly be called without a payload depth: it
either defaulted the value (12 / 16) or re-derived it from live camera state
at save time. A defaulted depth draws a scale bar at the wrong full-scale,
downconverts against the wrong range, or stamps a file with a depth read from
the camera's CURRENT format rather than the captured frame's. Making the
parameter required makes the depth-less call unconstructible -- the caller is
forced to hand down the depth it captured the frame at.

The re-derivation helper (_default_significant_bits) is deleted outright: a
save must never resolve depth from the camera's live state.
"""

import inspect

import numpy as np
import pytest

from modules import image_save, image_utils
from modules.cell_count import CellCount
from modules.post_processing import PostProcessing


def _significant_bits_param(func):
    return inspect.signature(func).parameters['significant_bits']


SEAM_FUNCS = [
    image_utils.add_scale_bar,
    image_utils._compute_scale_bar_overlay,
    PostProcessing.preview_cell_count,
    CellCount.process_image,
    image_save.prepare_image_for_saving,
    image_save.save_image,
]


@pytest.mark.parametrize('func', SEAM_FUNCS, ids=lambda f: f.__qualname__)
def test_significant_bits_has_no_default(func):
    param = _significant_bits_param(func)
    assert param.default is inspect.Parameter.empty, (
        f'{func.__qualname__} must require significant_bits, not default it -- '
        'a defaulted depth silently mis-scales the frame'
    )


def test_default_significant_bits_helper_is_gone():
    assert not hasattr(image_save, '_default_significant_bits'), (
        'a save must not re-derive depth from live camera state; the caller '
        'hands down the depth captured with the frame'
    )


def test_add_scale_bar_rejects_missing_depth():
    with pytest.raises(TypeError):
        image_utils.add_scale_bar(np.zeros((200, 200), dtype=np.uint16), {}, 1)


def test_save_image_rejects_missing_depth():
    with pytest.raises(TypeError):
        image_save.save_image(None, np.zeros((4, 4), dtype=np.uint8), save_encoding='8bit')


def test_save_live_image_returns_none_on_capture_failure(monkeypatch, tmp_path):
    """capture_and_wait returns None (never False) when the camera is inactive
    or the drain fails; save_live_image must surface that as its documented
    None return, not raise from save_image on a None array."""
    from types import SimpleNamespace

    scope = SimpleNamespace(
        imaging=SimpleNamespace(
            capture_and_wait=lambda **kw: None,
            capture_frame_depth=lambda array, sum_count=1: 8,
        ),
        illumination=SimpleNamespace(leds_off=lambda: None),
    )

    def _fail_save(*args, **kwargs):
        raise AssertionError('save_image must not be called after a failed capture')

    monkeypatch.setattr(image_save, 'save_image', _fail_save)

    out = image_save.save_live_image(scope, save_folder=str(tmp_path), save_encoding='8bit')
    assert out is None
