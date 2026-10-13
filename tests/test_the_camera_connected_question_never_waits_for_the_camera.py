# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether the camera is connected, and the 1:1 view's scale, never wait on the camera.

A Basler grab stop holds the SDK's camera lock for as long as the camera takes
to stop, which has been 11 s on one camera (#847). The UI thread asks
``camera_connected`` when a layer's settings are applied and when a binning is
picked, and the 1:1 button scales the view by the frame size; any of these that
calls into the SDK waits out the whole stop, and the window stops answering
for it. So the connected question answers from the driver's recorded state,
and the 1:1 scale from the cached frame size.

The SDK is a stand-in by design: the simulated camera does not go through
pylon, so it cannot hold pylon's lock. Each SDK call these paths could make is
made to block until the test releases it, standing in for a stop in progress.

Because the connected question no longer asks the SDK, the removal handler is
the signal for a camera unplugged while idle, so a connect whose handler cannot
be registered is refused.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from tests.camera_fakes import bare_pylon_camera
from tests.scope_fakes import give_stub_lanes

# Long enough that an answer which waits on the held SDK is plainly late; it
# only bounds a failing run.
_ANSWER_BOUND_S = 2.0


@pytest.fixture
def sdk_and_hold():
    """A PylonCamera, and the call that starts a stop holding its SDK.

    Before the hold its SDK answers at once, so a scope can be built over it
    the way bring-up builds one; once held, each call blocks until the test
    ends, standing in for a stop in progress.
    """
    cam = bare_pylon_camera()
    cam._device_removed = False
    held = threading.Event()
    released = threading.Event()

    def _answers(answer):
        def _call(*_args, **_kwargs):
            if held.is_set():
                released.wait(10)
            return answer

        return _call

    cam.active.IsCameraDeviceRemoved.side_effect = _answers(False)
    cam.active.Width.GetValue.side_effect = _answers(1900)
    cam.active.Height.GetValue.side_effect = _answers(1900)
    try:
        yield cam, held.set
    finally:
        released.set()


@pytest.fixture
def held_sdk(sdk_and_hold):
    """A PylonCamera whose SDK calls block until the test ends."""
    cam, hold = sdk_and_hold
    hold()
    return cam


def _answers_while_held(ask):
    """Run ``ask`` on another thread; return (answered in time, its answer)."""
    result = {}
    answered = threading.Event()

    def _run():
        result['value'] = ask()
        answered.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    in_time = answered.wait(_ANSWER_BOUND_S)
    return in_time, result.get('value')


class TestTheConnectedQuestion:
    def test_the_driver_answers_while_the_sdk_is_held(self, held_sdk):
        in_time, connected = _answers_while_held(held_sdk.is_connected)

        assert in_time, 'is_connected() waited on the SDK'
        assert connected is True

    def test_the_api_answers_while_the_sdk_is_held(self, held_sdk):
        # a stand-in by design: a held pylon lock is a fault the simulator cannot yet produce (the simulated camera does not go through pylon)
        scope = Lumascope.__new__(Lumascope)
        scope._camera_driver = held_sdk

        in_time, connected = _answers_while_held(lambda: scope.camera_connected)

        assert in_time, 'Lumascope.camera_connected waited on the SDK'
        assert connected is True

    def test_a_removed_camera_reads_disconnected(self, held_sdk):
        held_sdk._device_removed = True

        in_time, connected = _answers_while_held(held_sdk.is_connected)

        assert in_time
        assert connected is False


class TestTheOneToOneScale:
    @staticmethod
    def _press_one_to_one(sdk_and_hold, monkeypatch, frame_size):
        """Press 1:1, during a held stop, over a camera whose cache holds ``frame_size``."""
        import ui.main_display as main_display

        cam, hold = sdk_and_hold
        # a stand-in by design: a held pylon lock is a fault the simulator cannot yet produce (the simulated camera does not go through pylon)
        scope = Lumascope.__new__(Lumascope)
        scope._camera_driver = cam
        give_stub_lanes(scope)
        scope.imaging = ImagingAPI(scope, cam)
        scope.imaging._camera_cache['active'] = True
        scope.imaging._camera_cache['frame_size'] = frame_size
        hold()
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=scope))
        viewer = SimpleNamespace(size=(950, 950), scale=1.0, pos=(0, 0))
        display = SimpleNamespace(scope=scope, ids={'viewer_id': viewer}, status_line_height=22)
        in_time, _ = _answers_while_held(lambda: main_display.MainDisplay.one2one_image(display))
        return in_time, viewer

    def test_the_scale_is_set_while_the_sdk_is_held(self, sdk_and_hold, monkeypatch):
        in_time, viewer = self._press_one_to_one(
            sdk_and_hold, monkeypatch, {'width': 1900, 'height': 1900}
        )

        assert in_time, 'the 1:1 button waited on the SDK'
        assert viewer.scale == 2.0


class TestTheRemovalHandlerIsRequired:
    def test_a_connect_whose_removal_handler_cannot_register_is_refused(self, monkeypatch):
        from drivers import pyloncamera

        cam = bare_pylon_camera()
        # a stand-in by design: pylon refusing a registration is a fault the simulator cannot yet produce (it has no pylon)
        sdk_camera = MagicMock()

        def _register(configuration, *_args):
            if isinstance(configuration, pyloncamera._CameraRemovalHandler):
                raise RuntimeError('registration failed')

        sdk_camera.RegisterConfiguration.side_effect = _register
        monkeypatch.setattr(pyloncamera.pylon, 'InstantCamera', lambda _device: sdk_camera)

        assert cam.connect() is False
        assert cam.active is None
        sdk_camera.Open.assert_not_called()
