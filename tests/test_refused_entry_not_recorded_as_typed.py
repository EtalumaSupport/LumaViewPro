"""A refused entry must never be recorded as the value the user typed.

Five handlers wrote the app's own value back into the widget whose event had
invoked them. Where the widget re-dispatches on that write, the second pass
reads the app's value, finds it valid, and emits it as the user's -- and the
bundle asserts the user chose something they were actually refused.

The two halves of that split on whether the write dispatches:

- A TEXT box does not. Assigning ``.text`` rebuilds the lines and the cursor
  and never touches ``focus``, so the ``on_focus`` binding these handlers hang
  from cannot fire from an app write. The three text boxes are driven through
  their REAL handlers here, one commit per test, and what is pinned is that the
  attempt and the correction are BOTH recorded and the reverted value is never
  reported as typed.
- A SPINNER does: assigning a different ``.text`` dispatches its ``on_text``
  pick handler synchronously. The cold start's restores are driven through
  the real ``MicroscopeSettings.load_settings`` against the simulated session,
  with spinners that dispatch their kv handler on a write, and what is pinned
  is the record that comes out.
"""

import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest

import modules.app_context as _app_ctx
import ui.microscope_settings as ms
import ui.protocol_settings as ps
from modules import gui_logger
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


class _Widget:
    def __init__(self, text=''):
        self.text = text


@pytest.fixture
def emitted(monkeypatch):
    lines = []
    monkeypatch.setattr(
        gui_logger, 'text_input', lambda name, value: lines.append((name, str(value)))
    )
    return lines


def _advanced_panel(monkeypatch, widget_id, widget, stored):
    """A stand-in AdvancedSettings carrying just what these handlers touch."""
    from ui import advanced_settings

    from tests.settings_fixtures import settings_writer

    class _Panel:
        ids: ClassVar[dict] = {widget_id: widget}
        _commit_video_limit = staticmethod(advanced_settings.AdvancedSettings._commit_video_limit)

    context = type(
        'C', (), {'settings': stored, 'update_settings': staticmethod(settings_writer(stored))}
    )()
    monkeypatch.setattr(advanced_settings._app_ctx, 'ctx', context)
    return _Panel()


def test_a_refused_fps_limit_records_the_attempt_not_the_revert(emitted, monkeypatch):
    """Type 500 into a box that caps at 200: the bundle must not claim 30."""
    from ui.advanced_settings import AdvancedSettings

    widget = _Widget('500')
    stored = {'video': {'max_fps': 30}}
    panel = _advanced_panel(monkeypatch, 'video_max_fps_input', widget, stored)

    AdvancedSettings.update_video_max_fps(panel)  # one commit -- refused

    assert ('VIDEO_MAX_FPS', '500') in emitted, (
        f'the refused entry left no record of what was typed: {emitted}'
    )
    assert ('VIDEO_MAX_FPS_APPLIED', '30') in emitted, (
        f'the revert was not reported as the correction: {emitted}'
    )
    assert ('VIDEO_MAX_FPS', '30') not in emitted, (
        f'the reverted value was recorded as the one the user typed: {emitted}'
    )


def test_a_refused_duration_records_the_attempt_not_the_revert(emitted, monkeypatch):
    from ui.advanced_settings import AdvancedSettings

    widget = _Widget('99999')
    stored = {'video': {'max_duration_seconds': 300}}
    panel = _advanced_panel(monkeypatch, 'video_max_duration_input', widget, stored)

    AdvancedSettings.update_video_max_duration(panel)

    assert ('VIDEO_MAX_DURATION_S', '99999') in emitted, (
        f'the refused entry left no record of what was typed: {emitted}'
    )
    assert ('VIDEO_MAX_DURATION_S_APPLIED', '300') in emitted, (
        f'the revert was not reported as the correction: {emitted}'
    )
    assert ('VIDEO_MAX_DURATION_S', '300') not in emitted, (
        f'the reverted value was recorded as the one the user typed: {emitted}'
    )


def test_a_capture_root_records_and_stores_what_was_typed(emitted, monkeypatch):
    """A path-illegal root is recorded, and stored, as typed; the filename
    prefix made of it is the protocol's (capture_prefix), not the field's."""
    from unittest.mock import MagicMock

    from ui.protocol_settings import ProtocolSettings

    widget = _Widget('my/run:1')

    class _Panel:
        ids: ClassVar[dict] = {'capture_root': widget}
        _protocol = MagicMock()

    panel = _Panel()
    ProtocolSettings.update_capture_root(panel, widget.text)

    typed = [v for n, v in emitted if n == 'CAPTURE_ROOT']
    assert typed == ['my/run:1'], f'what the user typed is recorded once: {emitted}'
    panel._protocol.modify_capture_root.assert_called_once_with(capture_root='my/run:1')
    assert widget.text == 'my/run:1', 'the field shows what was typed'


class _Spinner:
    """A Kivy Spinner's text: a write of a different text dispatches ``on_text``,
    as the kv binding does, synchronously and whoever wrote it."""

    def __init__(self, text, on_text):
        self._text = text
        self._on_text = on_text
        self.values = []

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        if value != self._text:
            self._text = value
            self._on_text()


class _Labware(ps.ProtocolSettings):
    """The real protocol panel, holding only the labware spinner (kv text '')."""

    def __init__(self):
        # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
        self.ids = {'labware_spinner': _Spinner('', self.select_labware)}
        self._protocol = None
        self.picks = 0

    def select_labware(self):
        self.picks += 1
        super().select_labware()


class _Microscope(ms.MicroscopeSettings):
    """The real panel; its spinners start at their kv text and dispatch their kv handlers."""

    def __init__(self):
        # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
        self.ids = {
            'image_mode_spinner': _Spinner('Select', self.select_image_mode),
            'live_image_output_format_spinner': _Spinner(
                'Select', self.select_live_image_output_format
            ),
            'sequenced_image_output_format_spinner': _Spinner(
                'Select', self.select_sequenced_image_output_format
            ),
            'video_recording_format_spinner': _Spinner(
                'Select', self.select_video_recording_format
            ),
            'binning_spinner': _Spinner('Select', self.select_binning_size),
            'jpg_quality_slider': SimpleNamespace(value=0),
            'jpg_quality_value_label': SimpleNamespace(text=''),
            'frame_width_id': SimpleNamespace(text='', focus=False),
            'frame_height_id': SimpleNamespace(text='', focus=False),
            'enable_scale_bar_btn': SimpleNamespace(state='normal'),
            'show_tooltips_btn': SimpleNamespace(state='normal'),
        }
        self.binning_picks = 0

    def select_binning_size(self):
        self.binning_picks += 1
        super().select_binning_size()

    # The drawing that needs a window: the scope's controls, the stimulation
    # toggles on each layer, the field-of-view readout.
    def reconfigure_for_scope(self):
        pass

    def set_ui_features_for_scope(self):
        pass

    def apply_stimulation_support(self):
        pass

    def refresh_fov_labels(self):
        pass


@pytest.fixture
def cold_start(tmp_path, monkeypatch, caplog):
    """Run the panel's settings load as the app's start does, and return
    the GUI record it wrote with the two panels it filled."""
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    labware = _Labware()
    layer_widget = SimpleNamespace(
        ids={'gain_slider': SimpleNamespace(max=0), 'exp_slider': SimpleNamespace(max=0)},
        sync_widgets_from_settings=lambda: None,
    )
    zstack = SimpleNamespace(
        ids={
            name: SimpleNamespace(text='')
            for name in (
                'zstack_spinner',
                'zstack_stepsize_id',
                'zstack_range_id',
                'zstack_steps_id',
            )
        }
    )
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    ctx = SimpleNamespace(
        session=session,
        settings=session.settings,
        update_settings=session.update_settings,
        lumaview=SimpleNamespace(scope=session.scope),
        wellplate_loader=session.wellplate_loader,
        initializing=True,
        camera_executor=None,
        source_path=str(tmp_path),
        scope_display=SimpleNamespace(image_mode=None),
        stage=SimpleNamespace(full_redraw=lambda: None, show_protocol_steps=lambda enable: None),
        motion_settings=SimpleNamespace(
            ids={
                'protocol_settings_id': labware,
                'verticalcontrol_id': SimpleNamespace(
                    ids={'zstack_id': zstack}, show_turret_state=lambda prompt: None
                ),
            }
        ),
        image_settings=SimpleNamespace(
            layer_lookup=lambda layer: layer_widget,
            reconcile_layers_to_camera_caps=lambda: None,
        ),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    monkeypatch.setattr(gui_logger, '_write_backs', {})
    microscope = _Microscope()
    try:
        with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
            microscope.load_settings()
        recorded = [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']
        yield SimpleNamespace(
            recorded=recorded, microscope=microscope, labware=labware, settings=session.settings
        )
    finally:
        session.shutdown()


def test_restoring_labware_at_startup_is_not_recorded_as_a_selection(cold_start):
    """The restore writes the spinner, whose write runs select_labware, and
    then calls it again: two records of the stored plate, neither a pick."""
    plate = cold_start.settings['protocol']['labware']
    assert cold_start.labware.ids['labware_spinner'].text == plate
    assert cold_start.labware.picks == 2, 'the write and the call each ran the handler'
    assert not [line for line in cold_start.recorded if line.startswith('SELECT LABWARE')], (
        cold_start.recorded
    )


def test_restoring_binning_at_startup_is_not_recorded_as_a_selection(cold_start):
    """The twin of the labware restore: the spinner write and the explicit
    select_binning_size() each run the handler; neither is a pick."""
    label = cold_start.settings['binning']['size']
    assert cold_start.microscope.ids['binning_spinner'].text == label
    assert cold_start.microscope.binning_picks == 2, 'the write and the call each ran the handler'
    assert not [line for line in cold_start.recorded if line.startswith('SELECT BINNING')], (
        cold_start.recorded
    )
