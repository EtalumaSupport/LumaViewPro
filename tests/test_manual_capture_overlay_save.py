"""Regression: a manual capture saves what the operator can actually see.

The overlaid copy of a manual capture used to be written only when the app was
built in engineering mode, but the crosshairs toggle that produces the overlay
is available to everyone -- it is gated on a protocol being in progress, not on
a build mode. So an operator could switch crosshairs on, watch them render over
the live image, press capture, and receive a file with no overlay in it and no
message saying anything had been dropped.

The mode check also stood in front of the question that actually matters --
"is an overlay switched on?" -- which the very next line already asked, so the
two arms of the capture held the same fifteen-argument save call twice over.

Removing the mode check exposed a second defect underneath it. The overlay arm
grabs its own frame rather than going through save_live_image, and that grab
was passing no frame count, so switching an overlay on silently reduced a
summed capture to a single frame and stamped the file at the unsummed depth.
Only the luminescence layer offers a summing control today, which is the sole
reason this had not been noticed.
"""

import pathlib
import sys
import types
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ui.composite_capture is a Kivy widget module; conftest mocks `kivy` but not
# the uix submodules, and CompositeCapture subclasses FloatLayout (a bare
# MagicMock cannot be subclassed). A real minimal base for that one, and
# permissive mocks for the rest.


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_floatlayout = ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)

import modules.app_context as _app_ctx


LAYER = 'Lumi'


def _layer_config(sum_count=1):
    return {LAYER: {'exposure_ms': 100, 'sum': sum_count, 'illumination_ma': 0}}


@pytest.fixture
def capture_ctx(tmp_path):
    """Stand up the app context a manual capture reads, with no overlay set.

    Individual tests switch the overlays on. The layer carries no LED current,
    which keeps the dark-frame guard out of the way -- it is not what these
    tests are about.
    """
    ctx = MagicMock()
    ctx.settings = {
        'live_folder': str(tmp_path),
        'separate_folder_per_channel': False,
        'image_output_format': {'live': 'TIFF'},
        'jpg_quality': 90,
    }
    ctx.scope.runtime_state.get_well_label.return_value = 'A1'
    ctx.image_settings.accordion_item_lookup.return_value = types.SimpleNamespace(collapse=False)
    ctx.image_settings.layer_lookup.return_value = types.SimpleNamespace(
        ids={'false_color': types.SimpleNamespace(active=False)}
    )
    ctx.scope_display.use_bullseye = False
    ctx.scope_display.use_crosshairs = False
    ctx.scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
    ctx.scope.imaging.last_significant_bits = 8
    ctx.scope.imaging.capture_frame_depth.return_value = 8
    ctx.scope_display.add_crosshairs.side_effect = lambda img: img
    ctx.scope_display.transform_to_bullseye.side_effect = lambda img: img

    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    try:
        yield ctx
    finally:
        _app_ctx.ctx = original


def _run_capture(sum_count=1):
    """Drive the real capture path and hand back its save calls.

    ``_live_capture_impl`` never touches ``self``, so it runs unbound rather
    than requiring a realised Kivy widget tree.
    """
    from ui.composite_capture import CompositeCapture

    capture_config = types.SimpleNamespace(capture_depth=8, save_encoding='raw')

    with (
        patch('ui.composite_capture.save_live_image') as save_live,
        patch('ui.composite_capture.save_image') as save_one,
        patch('ui.composite_capture.set_last_save_folder'),
        patch('ui.composite_capture.common_utils.get_layers', return_value=[LAYER]),
        patch('ui.composite_capture.common_utils.get_layers_with_led', return_value=[]),
        patch(
            'modules.config_ui_getters.get_layer_configs',
            return_value=_layer_config(sum_count),
        ),
        patch(
            'modules.config_ui_getters.get_image_capture_config_from_ui',
            return_value=capture_config,
        ),
    ):
        CompositeCapture._live_capture_impl(object())
        return save_live, save_one


class TestOverlayIsSavedRegardlessOfBuildMode:
    def test_crosshairs_on_writes_the_overlay_copy(self, capture_ctx):
        """The defect itself: the operator sees crosshairs, so a copy carrying
        them must reach disk even though this is not an engineering build."""
        capture_ctx.engineering_mode = False
        capture_ctx.scope_display.use_crosshairs = True

        _, save_one = _run_capture()

        appends = [call.kwargs['append'] for call in save_one.call_args_list]
        assert len(appends) == 2, f'expected the clean image and an overlay copy, got {appends}'
        assert appends[1].endswith('_overlay')
        assert not appends[0].endswith('_overlay')

    def test_no_overlay_still_takes_the_single_plain_save(self, capture_ctx):
        """Collapsing the two duplicate arms must not change the no-overlay
        case: one file, written by save_live_image, carrying the frame count."""
        capture_ctx.engineering_mode = False

        save_live, save_one = _run_capture(sum_count=3)

        assert save_live.call_count == 1
        assert save_one.call_count == 0
        assert save_live.call_args.kwargs['sum_count'] == 3

    def test_engineering_mode_no_longer_decides_what_is_written(self):
        """Structural: the mode gate cannot be reintroduced quietly.

        A behavioural test only covers the modes it is run in; this pins that
        the capture path has no opinion about the build mode at all.
        """
        source = (
            pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'composite_capture.py'
        ).read_text()
        assert 'engineering_mode' not in source, (
            'ui/composite_capture.py consults engineering_mode again. What a '
            'capture writes follows from which overlays are switched on, not '
            'from how the application was built.'
        )


class TestSummingSurvivesAnOverlay:
    def test_overlay_capture_keeps_the_configured_frame_count(self, capture_ctx):
        """Switching on a display overlay must not change the exposure."""
        capture_ctx.engineering_mode = False
        capture_ctx.scope_display.use_crosshairs = True

        _run_capture(sum_count=3)

        kwargs = capture_ctx.scope.imaging.capture_and_wait.call_args.kwargs
        assert kwargs['sum_count'] == 3, (
            'the overlay path grabbed its frame without the configured frame '
            'count, so an overlay silently reduced a summed capture to one frame'
        )
        assert kwargs['sum_delay_s'] == pytest.approx(0.1)

    def test_saved_depth_accounts_for_the_summed_range(self, capture_ctx):
        """Summing widens the real range, and the file has to say so."""
        capture_ctx.engineering_mode = False
        capture_ctx.scope_display.use_crosshairs = True

        _run_capture(sum_count=3)

        depth_calls = capture_ctx.scope.imaging.capture_frame_depth.call_args_list
        assert any(len(call.args) == 2 and call.args[1] == 3 for call in depth_calls), (
            'the clean image was stamped at its unsummed depth; save_live_image '
            'resolves depth with the frame count and this path must match it'
        )
