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
grabbed its own frame rather than going through the plain capture, and that
grab was passing no frame count, so switching an overlay on silently reduced a
summed capture to a single frame and stamped the file at the unsummed depth.
Only the luminescence layer offers a summing control today, which is the sole
reason this had not been noticed.
"""

import ast
import pathlib
import types
from unittest.mock import patch

import numpy as np
import pytest

from tests.ast_seams import parse_module
from tests.scope_fakes import spec_scope

LAYER = 'Lumi'


@pytest.fixture
def capture_ctx(tmp_path):
    """A scope that answers what a manual capture reads, with no overlay set.

    Individual tests switch the overlays on and set the engineering mode. The
    layer carries no LED current, which keeps the dark-frame guard out of the
    way -- it is not what these tests are about. The camera lane runs the body
    on the calling thread, as it does for a scope with no executors.
    """
    scope = spec_scope()
    scope.runtime_state.get_well_label.return_value = 'A1'
    # The objective the frame is taken with, read at capture.
    scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
    scope.illumination.get_led_states.return_value = {}
    scope.imaging._capture_and_wait_impl.return_value = np.zeros((4, 4), dtype=np.uint8)
    scope.imaging.capture_frame_depth.return_value = 8
    scope.imaging._dispatch_camera.side_effect = (
        lambda impl, name, args=(), kwargs=None, *, timeout_s: impl(*args, **(kwargs or {}))
    )
    return types.SimpleNamespace(
        scope=scope,
        tmp_path=tmp_path,
        engineering_mode=False,
        use_bullseye=False,
        use_crosshairs=False,
    )


def _run_capture(ctx, sum_count=1):
    """Drive the one manual-capture path and hand back its save calls."""
    from modules.manual_capture import ManualCaptureController
    from tests.settings_fixtures import complete_settings

    settings = complete_settings(
        live_folder=str(ctx.tmp_path), separate_folder_per_channel=False, image_mode='8bit'
    )
    settings[LAYER].update({'exposure_ms': 100, 'sum': sum_count, 'illumination_ma': 0})
    capture = ManualCaptureController(
        scope=ctx.scope, settings_snapshot=lambda: settings, engineering_mode=ctx.engineering_mode
    )

    def _saved(scope, array, *, file_root, append, **kwargs):
        return pathlib.Path(ctx.tmp_path) / f'{file_root}{append}.tiff'

    with patch('modules.manual_capture.save_image', side_effect=_saved) as save_one:
        capture.capture(
            layer=LAYER,
            false_color_on=False,
            bullseye=ctx.use_bullseye,
            crosshairs=ctx.use_crosshairs,
        ).result(timeout=10)
    return save_one


class TestOverlayIsSavedRegardlessOfBuildMode:
    def test_crosshairs_on_writes_the_overlay_copy(self, capture_ctx):
        """The defect itself: the operator sees crosshairs, so a copy carrying
        them must reach disk even though this is not an engineering build."""
        capture_ctx.engineering_mode = False
        capture_ctx.use_crosshairs = True

        save_one = _run_capture(capture_ctx)

        appends = [call.kwargs['append'] for call in save_one.call_args_list]
        assert len(appends) == 2, f'expected the clean image and an overlay copy, got {appends}'
        assert appends[1].endswith('_overlay')
        assert not appends[0].endswith('_overlay')

    def test_no_overlay_still_takes_the_single_plain_save(self, capture_ctx):
        """The no-overlay case is one file, grabbed with the frame count."""
        capture_ctx.engineering_mode = False

        save_one = _run_capture(capture_ctx, sum_count=3)

        assert save_one.call_count == 1
        kwargs = capture_ctx.scope.imaging._capture_and_wait_impl.call_args.kwargs
        assert kwargs['sum_count'] == 3

    def test_engineering_mode_no_longer_decides_what_is_written(self):
        """Structural: the mode gate cannot be reintroduced quietly.

        A behavioural test only covers the modes it is run in; this pins that
        no save call sits under a branch on the build mode. The capture may
        still READ the flag -- it names a manual capture's turret position
        under it -- but WHICH files get written follows from the overlays that are
        switched on, never from how the application was built. Walks the
        AST rather than the text, so a mention in a comment is not a failure
        and a branch cannot hide behind reformatting.
        """

        def _reads_the_mode(expr):
            # A plain attribute read, or the ``getattr(ctx, 'engineering_mode', ...)``
            # form the engine's own retired reads used.
            return any(
                (isinstance(n, ast.Attribute) and n.attr == 'engineering_mode')
                or (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == 'getattr'
                    and len(n.args) >= 2
                    and isinstance(n.args[1], ast.Constant)
                    and n.args[1].value == 'engineering_mode'
                )
                for n in ast.walk(expr)
            )

        def _save_calls(nodes):
            for node in nodes:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        func = inner.func
                        name = (
                            func.attr
                            if isinstance(func, ast.Attribute)
                            else getattr(func, 'id', None)
                        )
                        if name == 'save_image':
                            yield inner.lineno

        tree = parse_module('modules/manual_capture.py')
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _reads_the_mode(node.test):
                offenders.extend(_save_calls(node.body + node.orelse))
            elif isinstance(node, ast.IfExp) and _reads_the_mode(node.test):
                offenders.extend(_save_calls([node.body, node.orelse]))
        assert not offenders, (
            f'modules/manual_capture.py chooses a save call on engineering_mode at '
            f'line(s) {offenders}. What a capture writes follows from which '
            'overlays are switched on, not from how the application was built.'
        )


class TestSummingSurvivesAnOverlay:
    def test_overlay_capture_keeps_the_configured_frame_count(self, capture_ctx):
        """Switching on a display overlay must not change the exposure."""
        capture_ctx.engineering_mode = False
        capture_ctx.use_crosshairs = True

        _run_capture(capture_ctx, sum_count=3)

        kwargs = capture_ctx.scope.imaging._capture_and_wait_impl.call_args.kwargs
        assert kwargs['sum_count'] == 3, (
            'the overlay path grabbed its frame without the configured frame '
            'count, so an overlay silently reduced a summed capture to one frame'
        )
        assert kwargs['sum_delay_s'] == pytest.approx(0.1)

    def test_saved_depth_accounts_for_the_summed_range(self, capture_ctx):
        """Summing widens the real range, and the file has to say so."""
        capture_ctx.engineering_mode = False
        capture_ctx.use_crosshairs = True

        _run_capture(capture_ctx, sum_count=3)

        depth_calls = capture_ctx.scope.imaging.capture_frame_depth.call_args_list
        assert any(len(call.args) == 2 and call.args[1] == 3 for call in depth_calls), (
            'the clean image was stamped at its unsummed depth; the depth is '
            'resolved with the frame count'
        )


class TestEngineeringModeNamesTheTurretPosition:
    """An engineering-mode manual capture keeps its turret position in the
    filename, in the one spelling the filename reader recognises.

    The token used to be appended by the free save-path function as ``_T<n>``,
    read off the application context two hops below this capture. That
    spelling is one no reader parses and shares its prefix with the tile
    vocabulary, and on a protocol run it duplicated the writer's own
    ``Turret<n>``. The capture now composes its name through the same renderer
    the writer uses, so a manual capture and a protocol step spell the
    position the same way.
    """

    def test_engineering_mode_writes_the_canonical_turret_token(self, capture_ctx):
        capture_ctx.engineering_mode = True
        capture_ctx.scope.motion.get_turret_slot.return_value = 2

        save_one = _run_capture(capture_ctx)

        append = save_one.call_args_list[0].kwargs['append']
        assert append == 'A1_Lumi_Turret2', append

    def test_the_legacy_spelling_is_gone(self, capture_ctx):
        capture_ctx.engineering_mode = True
        capture_ctx.scope.motion.get_turret_slot.return_value = 2
        capture_ctx.use_crosshairs = True

        save_one = _run_capture(capture_ctx)

        appends = [call.kwargs['append'] for call in save_one.call_args_list]
        assert appends, 'no save reached disk'
        for append in appends:
            assert '_T2' not in append.replace('_Turret2', ''), append
            assert append.count('Turret2') == 1, append

    def test_an_unknown_turret_position_adds_no_token(self, capture_ctx):
        """A scope that has not reported a turret position yet names the file
        exactly as production mode does; nothing is invented."""
        capture_ctx.engineering_mode = True
        capture_ctx.scope.motion.get_turret_slot.return_value = None

        save_one = _run_capture(capture_ctx)

        assert save_one.call_args_list[0].kwargs['append'] == 'A1_Lumi'

    def test_production_mode_names_no_turret_position(self, capture_ctx):
        """Behaviour preserved on both sides of the change."""
        capture_ctx.engineering_mode = False
        capture_ctx.scope.motion.get_turret_slot.return_value = 2

        save_one = _run_capture(capture_ctx)

        assert save_one.call_args_list[0].kwargs['append'] == 'A1_Lumi'
