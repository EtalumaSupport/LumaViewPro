# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus run is told which LED to sweep under and how bright.

``AutofocusRunner.run`` once defaulted ``led_illumination`` to 0, so a caller
that forgot it swept in the dark and reported a focus found on noise. The
channel and the current are required keywords, as the LED lease is.
"""

import inspect

from modules.autofocus_runner import AutofocusRunner


def test_the_led_channel_and_current_are_required_keywords():
    params = inspect.signature(AutofocusRunner.run).parameters
    for name in ('led_color', 'led_illumination'):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params[name].default is inspect.Parameter.empty, name
