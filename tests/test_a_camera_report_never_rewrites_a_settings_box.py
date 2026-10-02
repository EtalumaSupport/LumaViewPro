# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera's report of what it applied never rewrites a layer's boxes.

Bug shape: a camera that quantizes (the FX2 sets exposure in whole sensor
rows) answers a 1000 ms request with 1000.0057 ms. The GUI's camera
listener wrote that answer over the exposure box, so the box read
"1000.01" after the slider had set 1000.0 -- and the box commits what it
shows on focus loss, so a click in and out stored the camera's number
over the user's. The box shows the stored setting; what the camera
applied is the API's to answer (``set_exposure_ms`` returns it,
``get_exposure_ms`` reads it) and the saved frame's to record.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.lumascope_api.imaging import AppliedCameraSetting
from tests.scope_fakes import spec_scope
from ui.listener_bridge import UIListenerBridge


def _bridge_with_an_open_blue_layer():
    """A real bridge over a fake scope, Blue's drawer open at 1000 ms / 6 dB."""
    camera_listeners = []
    scope = spec_scope()
    scope.imaging.add_camera_listener.side_effect = camera_listeners.append

    boxes = {
        'exp_text': SimpleNamespace(text='1000.0'),
        'gain_text': SimpleNamespace(text='6.0'),
    }
    blue = SimpleNamespace(ids=boxes, _initializing=False)
    imaging = MagicMock()
    imaging.applied_exposure_ms_for.side_effect = lambda v: AppliedCameraSetting(
        stored=v, applied=v, capped=False
    )
    imaging.applied_gain_db_for.side_effect = lambda v: AppliedCameraSetting(
        stored=v, applied=v, capped=False
    )
    ctx = SimpleNamespace(
        ready=True,
        session=SimpleNamespace(is_protocol_running=False),
        settings={'Blue': {'exposure_ms': 1000.0, 'gain_db': 6.0}},
        image_settings=SimpleNamespace(
            layer_lookup=lambda layer: blue,
            accordion_item_lookup=lambda layer: SimpleNamespace(collapse=layer != 'Blue'),
        ),
        lumaview=SimpleNamespace(scope=SimpleNamespace(imaging=imaging)),
    )
    bridge = UIListenerBridge(
        scope=scope,
        ctx=ctx,
        stage=MagicMock(),
        ui_dispatcher=lambda callback, dt: callback(dt),
    )
    bridge.register_all()
    return camera_listeners, boxes


def test_a_quantized_exposure_leaves_the_box_at_what_was_set():
    listeners, boxes = _bridge_with_an_open_blue_layer()

    for listener in listeners:
        listener('exposure', 1000.0057)

    assert boxes['exp_text'].text == '1000.0'


def test_a_quantized_gain_leaves_the_box_at_what_was_set():
    listeners, boxes = _bridge_with_an_open_blue_layer()

    for listener in listeners:
        listener('gain', 6.06)

    assert boxes['gain_text'].text == '6.0'
