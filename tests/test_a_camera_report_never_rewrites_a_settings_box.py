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

from tests.scope_fakes import spec_scope
from ui.listener_bridge import UIListenerBridge


def test_the_window_hears_no_camera_report():
    # The boxes cannot be rewritten by what the camera applied when the window
    # subscribes to neither camera event. A test that fed a quantized value to
    # the bridge's camera listeners passed with none registered, the loop over
    # them running zero times; the subscription itself is the contract.
    # The subject is the bridge's subscriptions, which a spec'd scope records.
    scope = spec_scope()  # a stand-in by design: the subject is what the bridge subscribes to
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            add_shown_channel_listener=lambda listener: None,
            add_settings_listener=lambda listener: None,
        ),
    )
    UIListenerBridge(scope=scope, ctx=ctx, stage=MagicMock()).register_all()

    scope.camera.add_gain_listener.assert_not_called()
    scope.camera.add_exposure_listener.assert_not_called()
