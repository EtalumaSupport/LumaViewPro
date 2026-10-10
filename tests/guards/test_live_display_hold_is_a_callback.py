"""A saved frame reaches the display through the run's ``frame_captured`` event.

After every saved protocol frame the GUI holds that frame on screen for a
moment, so the user sees what was saved before the live preview overwrites
it. The writer used to find the display by reading the GUI's application
context from inside the engine; then it called a hold hook the GUI handed in
by name. Now the run publishes ``frame_captured(image, frames_summed,
frame_significant_bits)`` once the frame's write is the batch's, and every GUI
run starter subscribes ``show_captured_frame``, which renders the frame and
holds it. The engine names no display. The dead ``update_scope_display``
field, which nothing read, stays gone.
"""

import ast
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.protocol_drives import lent_run_claim
from tests.frame_records import frame_record, plate
from modules.protocol_image_writer import RunWriteBatch
import modules.app_context as _app_ctx
from modules.image_mode import ImageCaptureConfig
from modules.protocol_image_writer import ProtocolImageWriter
from modules.run_events import RunEvents
from tests.ast_seams import iter_package_modules, parse_module
from tests.scope_fakes import spec_scope

# ui.ui_helpers imports Kivy's clock and a widget; conftest mocks ``kivy`` but
# not its submodules, so the two it needs are stubbed the way the other GUI
# module tests stub theirs.
for _name in ('kivy.clock', 'kivy.uix', 'kivy.uix.scrollview'):
    sys.modules.setdefault(_name, MagicMock())


from modules.run_outcome import EndingLatch


def _writer(events):
    writer = ProtocolImageWriter(
        scope=spec_scope(),
        events=events,
        aborted=threading.Event(),
        write_batch=RunWriteBatch(MagicMock()),
        abort_fn=lambda: None,
        fatal_abort_event=threading.Event(),
        ending=EndingLatch(),
        execution_record=None,
        leds_off_fn=lambda: None,
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        timestamp_overlay=True,
        video_max_fps=0,
        engineering_mode=False,
        run_claim=lent_run_claim(),
        labware=plate(),
        to_plate=None,
        captures_asked=1,
    )
    scope = writer._scope
    # The objective the frame is taken with, read at capture.
    scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
    scope.capabilities.has_turret = False
    scope.led_connected = False
    scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
    scope.imaging.capture_frame_depth.return_value = 8
    scope.imaging.capture_frame_full_scale.return_value = 255
    scope.imaging.last_capture_info = {'frame_record': frame_record()}
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


class TestFrameCapturedCarriesTheSavedFrame:
    def test_a_saved_frame_reaches_the_event_with_no_context_in_the_process(self, no_context):
        """The writer hands the frame, its sum count and its depth to the
        event, after the frame's write was submitted, with nothing in the
        process for it to read a display from."""
        seen = []
        writer = _writer(None)

        def _captured(image, frames_summed, bits):
            seen.append((image, frames_summed, bits, writer._write_batch._executor.put.called))

        writer._events = RunEvents(frame_captured=_captured)

        _capture_one_still(writer)

        assert len(seen) == 1, seen
        image, frames_summed, bits, submitted = seen[0]
        assert image.shape == (4, 4)
        assert (frames_summed, bits) == (1, 8)
        assert submitted, 'frame_captured must follow the write being handed to the batch'

    def test_the_frame_cannot_be_changed_through_a_view_or_its_base(self, no_context):
        """The handler holds the array the file is written from; neither it,
        a view of it, nor its base can be made writeable again."""
        seen = []
        writer = _writer(RunEvents(frame_captured=lambda image, *_: seen.append(image)))

        _capture_one_still(writer)

        (image,) = seen
        assert not image.flags.writeable
        with pytest.raises(ValueError):
            image.view().flags.writeable = True
        base = image.base if image.base is not None else image
        assert not base.flags.writeable

    def test_no_handler_no_event_and_no_error(self, no_context):
        """A headless run subscribes nothing; the writer saves and moves on."""
        writer = _writer(RunEvents())

        _capture_one_still(writer)

        assert writer._write_batch._executor.put.called, 'the save itself must still run'

    def test_a_failed_handler_is_reported_once_as_itself_and_the_save_goes_on(
        self, no_context, monkeypatch
    ):
        """A frame the handler cannot show -- here one deeper than its declared
        depth -- is reported once, unasked, as the exception it is, under the
        event's name, and the capture's write is still submitted."""
        from modules.exceptions import FrameDepthError
        from modules.notification_center import notifications

        reported = []
        monkeypatch.setattr(notifications, 'report_outcome', lambda e, **k: reported.append((e, k)))
        failure = FrameDepthError(4095, 8)

        def _captured(image, frames_summed, bits):
            raise failure

        writer = _writer(RunEvents(frame_captured=_captured))

        _capture_one_still(writer)

        assert writer._write_batch._executor.put.called, 'the save itself must still run'
        assert reported == [(failure, {'solicited': False, 'category': 'frame_captured'})], reported


class TestTheGuiHandlerLooksTheDisplayUpLate:
    def test_a_missing_display_is_reported_by_the_delivery(self, monkeypatch):
        """A run that starts before the display is built: the handler's late
        read fails inside the event's delivery, which reports it once,
        unasked, and the save goes on."""
        from modules.notification_center import notifications
        from ui.ui_helpers import show_captured_frame

        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace())
        reported = []
        monkeypatch.setattr(notifications, 'report_outcome', lambda e, **k: reported.append((e, k)))
        writer = _writer(RunEvents(frame_captured=show_captured_frame))

        _capture_one_still(writer)

        holds = [(e, k) for e, k in reported if isinstance(e, AttributeError)]
        assert len(holds) == 1, reported
        assert holds[0][1]['solicited'] is False
        assert writer._write_batch._executor.put.called, 'the save itself must still run'

    def test_the_handler_reaches_the_live_display(self, monkeypatch):
        from ui.ui_helpers import show_captured_frame

        display = MagicMock()
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope_display=display))
        image = np.full((2, 2), 200, dtype=np.uint8)

        show_captured_frame(image, 1, 8)

        ((held, bits), _kw) = display.hold_protocol_saved_image.call_args
        assert bits == 8 and held.dtype == np.uint8
        assert np.array_equal(held, image)


def _subscribes_the_handler(call: ast.Call) -> bool:
    return any(
        kw.arg == 'frame_captured'
        and isinstance(kw.value, ast.Name)
        and kw.value.id == 'show_captured_frame'
        for kw in call.keywords
    )


class TestEveryGuiRunStarterSubscribesTheHandler:
    def test_every_run_events_the_gui_builds_subscribes_show_captured_frame(self):
        """Source-level: a GUI run started without the handler would silently
        lose the hold. Every ``RunEvents(...)`` built under ui/ subscribes
        ``show_captured_frame`` as its ``frame_captured``."""
        built = []
        offenders = []
        for rel, tree in iter_package_modules(('ui',)):
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == 'RunEvents'
                ):
                    built.append(f'{rel}:{n.lineno}')
                    if not _subscribes_the_handler(n):
                        offenders.append(f'{rel}:{n.lineno}')
        assert built, 'no GUI run starter builds RunEvents; the guard would be vacuous'
        assert not offenders, (
            f'these GUI run starters build RunEvents without frame_captured=show_captured_frame: '
            f'{offenders}'
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


class TestTheHoldShowsASumAsItIsShown:
    def test_a_sum_reaches_the_display_rendered_against_one_frames_white(self, monkeypatch):
        from ui.ui_helpers import show_captured_frame

        display = MagicMock()
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope_display=display))
        writer = _writer(RunEvents(frame_captured=show_captured_frame))
        imaging = writer._scope.imaging
        imaging.capture_and_wait.return_value = np.full((4, 4), 255, dtype=np.uint16)
        imaging.capture_frame_depth.return_value = 10
        imaging.capture_frame_full_scale.return_value = 765
        imaging.last_capture_info = {
            'frame_record': frame_record(frames_summed=3, frame_significant_bits=8)
        }

        _capture_one_still(writer)

        ((image, bits), _kw) = display.hold_protocol_saved_image.call_args
        assert image.dtype == np.uint8 and bits == 8
        assert int(image.min()) == 255
