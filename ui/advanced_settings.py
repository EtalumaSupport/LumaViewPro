# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Advanced Settings modal.

A single popup that houses power-user / rarely-touched settings (sensor
low-noise toggles, video limits, motion tuning, output options, scope
identity) so the main Microscope Settings panel stays focused on
everyday controls. Each row reads and writes the shared settings store
and applies through the imaging / motion APIs -- this component owns the
rows, rather than the popup being driven from the panel.
"""

from kivy.lang import Builder
from kivy.uix.popup import Popup

from lvp_logger import logger


class AdvancedSettings(Popup):
    """Modal container for the advanced settings rows.

    Opened from a button in the Microscope Settings panel. Rows are added
    to the ``advanced_rows`` container; the container scrolls so the box
    holds an arbitrary number of settings without resizing the window.
    """

    def close(self):
        logger.debug('[Advanced ] AdvancedSettings closed')
        self.dismiss()


kv = Builder.load_string(
    """
<AdvancedSettings>:
    size_hint: .55, .85
    auto_dismiss: True
    title: 'Advanced Settings'

    BoxLayout:
        orientation: 'vertical'
        spacing: dp(6)

        ScrollView:
            scroll_type: ['bars']
            do_scroll_x: False
            bar_width: dp(8)
            BoxLayout:
                id: advanced_rows
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: 0, 0, dp(8), 0
                spacing: dp(5)

        Button:
            text: 'Close'
            size_hint_y: None
            height: '36dp'
            on_release: root.close()
"""
)
