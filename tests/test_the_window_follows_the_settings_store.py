# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The window follows the settings store: a change announced is redrawn, and a redraw writes nothing.

With REST served beside it, a client's write left the window showing the
old value. The listener bridge now takes the Session's ``SettingsChanged``,
gathers a frame's paths, and on the UI thread runs each redraw the
changed paths route to (``ui.listener_bridge.SHOWN_SETTINGS``), once per
frame however many paths it gathered.

A redraw writes nothing back. Six widgets' handlers ran when a redraw set
them to the stored value and wrote it again -- the image mode and the
binning re-drove the camera, the labware, the z-stack position, the JPG
quality and the video format each wrote the setting -- and under a client's
control session each write would be refused and shown to the operator as a
refusal of a change they never made. Each now takes the stored value as no
pick.

Kivy is stubbed in the test process: each handler is driven on a stand-in
shaped like its panel, against a simulated session holding a client's
control session.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.microscope_settings as microscope_settings
from modules import kivy_utils
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from ui.listener_bridge import UIListenerBridge
from ui.microscope_settings import MicroscopeSettings
from ui.protocol_settings import ProtocolSettings
from ui.zstack import ZSTACK_REFERENCE_LABELS, ZStack


@pytest.fixture
def session(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    s = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield s
    s.shutdown()


@pytest.fixture
def held(session, monkeypatch, centre_posts):
    """A client holds the scope; every write the window makes would be refused and shown."""
    session.take_control('client')
    stored = []
    session.add_settings_listener(stored.append)
    dispatched = []
    # a stand-in by design: catches a camera write that must not be sent;
    # sent, it would be refused on the lane and reported there.
    monkeypatch.setattr(
        microscope_settings, 'submit_reported', lambda *args, **kwargs: dispatched.append(args)
    )
    redrawn = []
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=session,
            settings=session.settings,
            scope=session.scope,
            update_settings=session.update_settings,
            initializing=False,
            camera_executor=None,
            stage=SimpleNamespace(full_redraw=lambda: redrawn.append('stage')),
        ),
    )
    yield SimpleNamespace(stored=stored, dispatched=dispatched, posts=centre_posts)
    session.remove_settings_listener(stored.append)


def _box(text):
    return SimpleNamespace(text=text, values=[])


def test_each_widget_a_redraw_sets_to_the_stored_value_writes_nothing(session, held):
    settings = session.settings
    panel = SimpleNamespace(
        ids={
            'image_mode_spinner': _box(
                microscope_settings.IMAGE_MODE_LABELS[settings['image_mode']]
            ),
            'binning_spinner': _box(settings['binning']['size']),
            'jpg_quality_value_label': _box(''),
            'video_recording_format_spinner': _box(
                'Frames' if settings['video_as_frames'] else 'mp4'
            ),
        },
        _redraw_image_mode=lambda: None,
        _redraw_framing=lambda: None,
    )
    MicroscopeSettings.select_image_mode(panel)
    MicroscopeSettings.select_binning_size(panel)
    MicroscopeSettings.update_jpg_quality(panel, settings['jpg_quality'])
    MicroscopeSettings.select_video_recording_format(panel)
    ZStack.set_position(
        SimpleNamespace(
            ids={'zstack_spinner': _box(ZSTACK_REFERENCE_LABELS[settings['zstack']['position']])}
        )
    )
    ProtocolSettings.select_labware(
        SimpleNamespace(
            ids={'labware_spinner': _box(settings['protocol']['labware'])}, _protocol=None
        )
    )

    assert held.stored == []
    assert held.dispatched == []
    assert [p.title for p in held.posts] == []


class _Frame:
    """The UI thread's next frame, run when the test says: what was scheduled meanwhile."""

    def __init__(self):
        self.scheduled = []
        self.thread = threading.current_thread()

    def schedule(self, func, timeout):
        self.scheduled.append(func)

    def run_all(self):
        scheduled, self.scheduled = self.scheduled, []
        for func in scheduled:
            func(0)


def test_a_frames_changes_are_redrawn_once_each_on_the_ui_thread():
    """Every path routed, each redraw run once for the frame, whatever the burst."""
    ui = _Frame()
    drawn = []

    class _Layer:
        def __init__(self, layer):
            self.layer = layer

        def sync_widgets_from_settings(self):
            drawn.append(f'layer {self.layer}')

    # a stand-in by design: Kivy is stubbed in the test process; the subject is the routing
    ctx = SimpleNamespace(
        ready=True,
        settings={'live_view_fps': 12, 'show_tooltips': True},
        image_settings=SimpleNamespace(layer_lookup=lambda layer: _Layer(layer)),
        motion_settings=SimpleNamespace(
            ids={
                'microscope_settings_id': SimpleNamespace(
                    show_capture_settings=lambda: drawn.append('capture'),
                    advanced_settings_shown=lambda: None,
                ),
                'verticalcontrol_id': SimpleNamespace(
                    ids={'zstack_id': SimpleNamespace(show_settings=lambda: drawn.append('zstack'))}
                ),
            }
        ),
    )
    bridge = UIListenerBridge(scope=None, ctx=ctx, stage=None)
    ScopeSession.set_ui_dispatcher(kivy_utils.UiDispatcher(schedule=ui.schedule, thread=ui.thread))
    try:
        bridge._on_settings_changed(('BF.gain_db', 'jpg_quality'))
        bridge._on_settings_changed(('BF.exposure_ms', 'zstack.range', 'scale_bar.enabled'))
        assert drawn == []
        ui.run_all()
    finally:
        ScopeSession.set_ui_dispatcher(None)

    assert sorted(drawn) == ['capture', 'layer BF', 'zstack']


def test_a_redraw_leaves_what_the_operator_is_editing_and_follows_the_rest():
    """A box being typed in keeps the edit, which its focus loss commits; a slider mid-drag keeps the finger's value.

    A focused box still showing what was last drawn holds no edit: it takes
    the new value, so its focus loss commits nothing stale over another
    writer's change.
    """
    from ui.ui_helpers import show_on_slider, show_text

    typing = SimpleNamespace(text='5', focus=True, shown_text='3')
    focused_untouched = SimpleNamespace(text='3', focus=True, shown_text='3')
    idle = SimpleNamespace(text='3', focus=False, shown_text='3')
    for box in (typing, focused_untouched, idle):
        show_text(box, '7')
    dragged = SimpleNamespace(value=10.0, user_interacting=True)
    at_rest = SimpleNamespace(value=10.0, user_interacting=False)
    for slider in (dragged, at_rest):
        show_on_slider(slider, 20.0)

    assert (typing.text, focused_untouched.text, idle.text) == ('5', '7', '7')
    assert (dragged.value, at_rest.value) == (10.0, 20.0)
