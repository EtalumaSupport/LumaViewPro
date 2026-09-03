"""The live-display hold is a run callback the GUI hands in, not a context read.

After every saved protocol frame the writer holds that frame on screen for a
moment, so the user sees what was saved before the live preview overwrites
it. The writer used to find the display by reading the GUI's application
context from inside the engine, guarded so a process with no display skipped
it. That was the last GUI read inside the protocol image writer, and the only
run callback that did not travel on the callbacks object every other UI hook
rides on.

Now the hold is a field on the callbacks object. Every GUI run starter spreads
one helper that supplies it, late-bound so a run started before the display
exists degrades inside the writer's own guard exactly as the direct read did.
The dead ``update_scope_display`` field, which nothing read, goes with it.
"""

import ast
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import modules.app_context as _app_ctx
from modules.image_mode import ImageCaptureConfig
from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_image_writer import ProtocolImageWriter
from tests.ast_seams import iter_package_modules, parse_module
from tests.scope_fakes import spec_scope

# ui.ui_helpers imports Kivy's clock and a widget; conftest mocks ``kivy`` but
# not its submodules, so the two it needs are stubbed the way the other GUI
# module tests stub theirs.
for _name in ('kivy.clock', 'kivy.uix', 'kivy.uix.scrollview'):
    sys.modules.setdefault(_name, MagicMock())


def _writer(callbacks):
    writer = ProtocolImageWriter(
        scope=spec_scope(),
        callbacks=callbacks,
        aborted=threading.Event(),
        file_io_executor=MagicMock(),
        abort_fn=lambda: None,
        fatal_abort_event=threading.Event(),
        execution_record=None,
        leds_off_fn=lambda: None,
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        timestamp_overlay=True,
        video_max_fps=0,
        engineering_mode=False,
    )
    scope = writer._scope
    scope.capabilities.has_turret = False
    scope.led_connected = False
    scope.imaging._capture_and_wait_impl.return_value = np.zeros((4, 4), dtype=np.uint8)
    scope.imaging.capture_frame_depth.return_value = 8
    return writer


def _capture_one_still(writer):
    protocol = MagicMock()
    protocol.capture_root.return_value = ''
    writer.capture(
        save_folder='/tmp',
        step={
            'Name': 'stepA',
            'Label': '',
            'Acquire': 'image',
            'Auto_Gain': False,
            'Color': 'BF',
            'Gain': 2.0,
            'Exposure': 10.0,
            'Objective': '4x',
            'Well': 'A1',
            'Z-Slice': 0,
            'Tile': '',
            'Illumination': 50.0,
            'False_Color': False,
        },
        output_format='TIFF',
        protocol=protocol,
        enable_image_saving=True,
    )


@pytest.fixture
def no_context(monkeypatch):
    monkeypatch.setattr(_app_ctx, 'ctx', None)


class TestTheHoldFiresThroughTheCallback:
    def test_a_saved_frame_reaches_the_hold_with_no_context_in_the_process(self, no_context):
        """The first test this hold has ever had: the writer hands the frame
        and its depth to the callback it was given, with nothing in the
        process for it to read a display from."""
        holds = []
        writer = _writer(
            ProtocolCallbacks(
                hold_protocol_saved_image=lambda image, bits: holds.append((image, bits))
            )
        )

        _capture_one_still(writer)

        assert len(holds) == 1, holds
        image, bits = holds[0]
        assert image.shape == (4, 4)
        assert bits == 8

    def test_no_callback_no_hold_and_no_error(self, no_context):
        """A headless run supplies no display hook; the writer saves and
        moves on."""
        writer = _writer(ProtocolCallbacks())

        _capture_one_still(writer)

        assert writer._file_io_executor.protocol_put_wait.called, 'the save itself must still run'


class TestTheGuiHelperIsLateBound:
    def test_the_helper_builds_before_the_display_exists(self, monkeypatch):
        """A starter that runs before the display is built must not raise
        while assembling its callbacks; the degradation happens later, inside
        the writer's own guard, exactly where the direct read degraded."""
        from ui.ui_helpers import live_display_callbacks

        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace())

        callbacks = live_display_callbacks()

        assert set(callbacks) == {'hold_protocol_saved_image'}
        assert callable(callbacks['hold_protocol_saved_image'])

    def test_a_missing_display_degrades_at_debug_inside_the_writer(self, monkeypatch):
        from ui.ui_helpers import live_display_callbacks

        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace())
        debug_lines = []

        def quiet(*args, **kwargs):
            return None

        monkeypatch.setattr(
            'modules.protocol_image_writer.logger',
            SimpleNamespace(
                isEnabledFor=lambda level: False,
                debug=lambda msg, *a, **k: debug_lines.append(str(msg)),
                info=quiet,
                warning=quiet,
                error=quiet,
                exception=quiet,
                critical=quiet,
            ),
        )
        writer = _writer(ProtocolCallbacks(**live_display_callbacks()))

        _capture_one_still(writer)

        assert any('hold_protocol_saved_image' in line for line in debug_lines), debug_lines
        assert writer._file_io_executor.protocol_put_wait.called, 'the save itself must still run'

    def test_the_helper_reaches_the_live_display(self, monkeypatch):
        from ui.ui_helpers import live_display_callbacks

        display = MagicMock()
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope_display=display))
        image = np.zeros((2, 2), dtype=np.uint8)

        live_display_callbacks()['hold_protocol_saved_image'](image, 12)

        display.hold_protocol_saved_image.assert_called_once_with(image, 12)


def _spreads_the_helper(node) -> bool:
    return any(
        isinstance(n, ast.Dict)
        and any(
            key is None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == 'live_display_callbacks'
            for key, value in zip(n.keys, n.values, strict=True)
        )
        for n in ast.walk(node)
    )


def _passes_callbacks_to_a_run(node) -> bool:
    for n in ast.walk(node):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        if n.func.attr in ('prepare', 'start_composite') and any(
            kw.arg == 'callbacks' for kw in n.keywords
        ):
            return True
    return False


class TestEveryGuiRunStarterSpreadsTheHelper:
    def test_every_ui_method_that_hands_callbacks_to_a_run_spreads_the_helper(self):
        """Source-level: a GUI run started without the helper would silently
        lose the hold. Every method under ui/ that passes callbacks into
        prepare or start_composite must spread the helper somewhere in its
        body (directly, or through the dict it builds and hands down)."""
        offenders = []
        for rel, tree in iter_package_modules(('ui',)):
            # Outermost scopes only: a starter builds its dict in the method
            # and hands it down to a nested closure that calls prepare, so the
            # method is the unit that must carry the spread.
            outer = [n for n in tree.body if isinstance(n, ast.FunctionDef)] + [
                m
                for c in tree.body
                if isinstance(c, ast.ClassDef)
                for m in c.body
                if isinstance(m, ast.FunctionDef)
            ]
            for node in outer:
                if _passes_callbacks_to_a_run(node) and not _spreads_the_helper(node):
                    offenders.append(f'{rel}:{node.lineno} {node.name}')
        assert not offenders, (
            'these GUI run starters hand callbacks to a run without spreading '
            f'live_display_callbacks(): {offenders}'
        )

    def test_the_dead_display_key_is_gone_everywhere(self):
        """``update_scope_display`` was a field nothing read and three no-op
        lambdas feeding it; none of the four may come back, as a name, an
        attribute, a dict key or a keyword."""

        def _names_it(tree) -> bool:
            for n in ast.walk(tree):
                if isinstance(n, ast.Name) and n.id == 'update_scope_display':
                    return True
                if isinstance(n, ast.Attribute) and n.attr == 'update_scope_display':
                    return True
                if isinstance(n, ast.Constant) and n.value == 'update_scope_display':
                    return True
                if isinstance(n, ast.keyword) and n.arg == 'update_scope_display':
                    return True
            return False

        hits = [rel for rel, tree in iter_package_modules(('modules', 'ui')) if _names_it(tree)]
        if _names_it(parse_module('lumaviewpro.py')):
            hits.append('lumaviewpro.py')
        assert not hits, f'update_scope_display survives in {hits}'

    def test_the_writer_reads_no_context(self):
        """Structural: the display hold was the writer's last context read."""
        tree = parse_module('modules/protocol_image_writer.py')
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(a.name == 'modules.app_context' for a in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == 'modules.app_context'
                    or (
                        node.module == 'modules'
                        and any(a.name == 'app_context' for a in node.names)
                    )
                )
            )
        ]
        assert not offenders, (
            f'modules/protocol_image_writer.py imports the application context at {offenders}'
        )
