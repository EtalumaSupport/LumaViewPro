# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Advanced Settings modal.

A single popup that houses power-user / rarely-touched settings (sensor
low-noise toggles, video limits, motion tuning, output options, scope
identity) so the main Microscope Settings panel stays focused on
everyday controls. Each row reads and writes the shared settings store
and applies through the imaging / motion APIs -- this component owns the
rows, rather than the popup being driven from the panel.

The popup is created fresh each time it opens, so ``on_open`` is the load
path: it populates every row from the settings store. There is no
persistent startup-load for these rows.
"""

from kivy.lang import Builder
from kivy.properties import BooleanProperty
from kivy.uix.popup import Popup

import modules.app_context as _app_ctx
from lvp_logger import logger
from modules import gui_logger
from modules.config_helpers import get_manual_video_max_duration
from modules.sequential_io_executor import IOTask


class AdvancedSettings(Popup):
    """Modal container for the advanced settings rows.

    Opened from a button in the Microscope Settings panel. Rows live in the
    ``advanced_rows`` container, which scrolls so the box holds an arbitrary
    number of settings without resizing the window.
    """

    # Mirror the camera low-noise capabilities so each toggle hides when the
    # camera lacks the node (Pylon Bsl features). Set on open from
    # scope.capabilities; default hidden so a toggle never flashes before the
    # probe result is read.
    conversion_gain_supported = BooleanProperty(False)
    line_noise_reduction_supported = BooleanProperty(False)

    def on_open(self):
        """Populate every row from the settings store when the modal opens."""
        ctx = _app_ctx.ctx
        settings = ctx.settings

        caps = ctx.lumaview.scope.capabilities
        self.conversion_gain_supported = caps.camera_supports_conversion_gain_mode
        self.line_noise_reduction_supported = caps.camera_supports_line_noise_reduction
        camera_settings = settings.setdefault('camera', {})
        self.ids['high_conversion_gain'].active = bool(
            self.conversion_gain_supported and camera_settings.get('high_conversion_gain', False)
        )
        self.ids['line_noise_reduction'].active = bool(
            self.line_noise_reduction_supported
            and camera_settings.get('line_noise_reduction', False)
        )

        manual_video = settings.get('manual_video', {})
        self.ids['manual_video_max_fps_input'].text = str(manual_video.get('max_fps', 0))
        self.ids['manual_video_max_duration_input'].text = str(
            get_manual_video_max_duration(settings)
        )

    def update_high_conversion_gain(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        state = self.ids['high_conversion_gain'].active
        gui_logger.select('HIGH_CONVERSION_GAIN', state)
        settings.setdefault('camera', {})['high_conversion_gain'] = state
        mode = 'High' if state else 'Low'

        def _set_conversion_gain():
            ctx.lumaview.scope.imaging.set_conversion_gain_mode(mode)

        ctx.camera_executor.put(IOTask(action=_set_conversion_gain))

    def update_line_noise_reduction(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        state = self.ids['line_noise_reduction'].active
        gui_logger.select('LINE_NOISE_REDUCTION', state)
        settings.setdefault('camera', {})['line_noise_reduction'] = state

        def _set_line_noise():
            ctx.lumaview.scope.imaging.set_line_noise_reduction(state)

        ctx.camera_executor.put(IOTask(action=_set_line_noise))

    def update_manual_video_max_fps(self):
        # 0 = no limit (camera free-run rate). record_init keys
        # _user_requested_fps_limit on this; non-zero requests the
        # camera-side rate cap.
        settings = _app_ctx.ctx.settings
        widget = self.ids['manual_video_max_fps_input']
        try:
            value = int(widget.text)
        except (ValueError, TypeError):
            value = -1
        if value < 0 or value > 200:
            from modules.notification_center import notifications

            notifications.warning(
                'Settings',
                'Invalid FPS limit',
                'Manual Video Max FPS must be between 0 and 200 (0 = no limit). '
                'Reverting to previous value.',
            )
            settings.setdefault('manual_video', {})
            widget.text = str(settings['manual_video'].get('max_fps', 0))
            return
        settings.setdefault('manual_video', {})
        settings['manual_video']['max_fps'] = value
        gui_logger.text_input_debounced('MANUAL_VIDEO_MAX_FPS', value)

    def update_manual_video_max_duration(self):
        # Memmap allocates max_fps * duration frames; the disk-space
        # pre-flight in record_init catches infeasible sizes.
        settings = _app_ctx.ctx.settings
        widget = self.ids['manual_video_max_duration_input']
        try:
            value = int(widget.text)
        except (ValueError, TypeError):
            value = 0
        if value < 1 or value > 3600:
            from modules.notification_center import notifications

            notifications.warning(
                'Settings',
                'Invalid time limit',
                'Video Time Limit must be between 1 and 3600 seconds. Reverting to previous value.',
            )
            settings.setdefault('manual_video', {})
            widget.text = str(get_manual_video_max_duration(settings))
            return
        settings.setdefault('manual_video', {})
        settings['manual_video']['max_duration_seconds'] = value
        gui_logger.text_input_debounced('MANUAL_VIDEO_MAX_DURATION_S', value)

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

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp' if root.conversion_gain_supported else 0
                    opacity: 1 if root.conversion_gain_supported else 0
                    disabled: not root.conversion_gain_supported
                    Label:
                        font_size: '12sp'
                        text: 'High Conversion Gain'
                        tooltip_text: 'Sensor low-noise mode: lower read noise for dim/fluorescence imaging,\\nat the cost of dynamic range'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                    CheckBox:
                        id: high_conversion_gain
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '25dp'
                        active: False
                        on_release: root.update_high_conversion_gain()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp' if root.line_noise_reduction_supported else 0
                    opacity: 1 if root.line_noise_reduction_supported else 0
                    disabled: not root.line_noise_reduction_supported
                    Label:
                        font_size: '12sp'
                        text: 'Line Noise Reduction'
                        tooltip_text: 'Camera filter that smooths horizontal stripe artifacts in the sensor readout'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                    CheckBox:
                        id: line_noise_reduction
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '25dp'
                        active: False
                        on_release: root.update_line_noise_reduction()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Manual Video Max FPS'
                        tooltip_text: 'Maximum frames per second for manual recording.\\n0 = no limit (camera free-run rate).\\nAt low FPS the camera is rate-limited so live preview slows too;\\nrestored to free-run when recording stops.'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    TextInput:
                        id: manual_video_max_fps_input
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '45dp'
                        multiline: False
                        font_size: '12sp'
                        padding: ['4dp', (self.height-self.line_height)/2]
                        halign: 'right'
                        input_filter: 'int'
                        text: '0'
                        on_text_validate: root.update_manual_video_max_fps()
                        on_focus: if not self.focus: root.update_manual_video_max_fps()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Video Time Limit (s)'
                        tooltip_text: 'Global maximum video length, in seconds.\\nManual recording auto-stops at this limit (press Record\\nagain to stop sooner); a protocol video Step longer than\\nthis is flagged. Memmap is sized for max_fps * limit frames.'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    TextInput:
                        id: manual_video_max_duration_input
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '45dp'
                        multiline: False
                        font_size: '12sp'
                        padding: ['4dp', (self.height-self.line_height)/2]
                        halign: 'right'
                        input_filter: 'int'
                        text: '30'
                        on_text_validate: root.update_manual_video_max_duration()
                        on_focus: if not self.focus: root.update_manual_video_max_duration()

        Button:
            text: 'Close'
            size_hint_y: None
            height: '36dp'
            on_release: root.close()
"""
)
