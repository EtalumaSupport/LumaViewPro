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

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import BooleanProperty
from kivy.uix.popup import Popup

import modules.app_context as _app_ctx
from lvp_logger import logger
from modules import gui_logger
from modules.config_helpers import get_manual_video_max_duration
from modules.config_ui_getters import firmware_stim_supported
from modules.sequential_io_executor import IOTask
from modules.tiling_config import TilingConfig


class AdvancedSettings(Popup):
    """Modal container for the advanced settings rows.

    Opened from a button in the Microscope Settings panel. Rows are grouped
    under Camera / Protocol / General headers; each group is a two-column
    sub-grid inside the vertical ``advanced_sections`` box. The popup
    auto-sizes its height to that box so it grows as settings are added, with
    no scroll bar.
    """

    # Mirror the camera low-noise capabilities so each toggle hides when the
    # camera lacks the node (Pylon Bsl features). Set on open from
    # scope.capabilities; default hidden so a toggle never flashes before the
    # probe result is read.
    conversion_gain_supported = BooleanProperty(False)
    line_noise_reduction_supported = BooleanProperty(False)

    # The acceleration limit applies to the X-Y stage, so the row shows only
    # when an X-Y stage is actually present. Gated on the capabilities probe
    # (the single source of truth for present hardware), not the scope-model
    # config -- the config only says the model can ship with a stage, not that
    # one is connected. Set on open; default hidden so the row never flashes
    # before the capability is read.
    xy_stage_supported = BooleanProperty(False)

    # Mirrors the LED firmware's stim capability so the row hides when the
    # firmware cannot drive stimulation. Set on open from
    # firmware_stim_supported(); default hidden so the row never flashes
    # before the capability is read.
    stim_supported = BooleanProperty(False)

    def on_open(self):
        """Populate every row from the settings store when the modal opens."""
        ctx = _app_ctx.ctx
        settings = ctx.settings

        caps = ctx.lumaview.scope.capabilities
        self.conversion_gain_supported = caps.camera_supports_conversion_gain_mode
        self.line_noise_reduction_supported = caps.camera_supports_line_noise_reduction
        self.xy_stage_supported = caps.has_xy_stage
        camera_settings = settings.setdefault('camera', {})
        self.ids['high_conversion_gain'].active = bool(
            self.conversion_gain_supported and camera_settings.get('high_conversion_gain', False)
        )
        self.ids['line_noise_reduction'].active = bool(
            self.line_noise_reduction_supported
            and camera_settings.get('line_noise_reduction', False)
        )

        video_settings = settings.get('video', {})
        self.ids['video_max_fps_input'].text = str(video_settings.get('max_fps', 0))
        self.ids['video_max_duration_input'].text = str(get_manual_video_max_duration(settings))
        self.ids['video_timestamp_overlay_id'].active = video_settings.get(
            'timestamp_overlay', True
        )

        self.ids['separate_folder_per_channel_id'].state = (
            'down' if settings.get('separate_folder_per_channel') else 'normal'
        )

        # The runtime fps mirror (ctx.live_view_fps) and scope_display are
        # initialized at app startup; the slider just reflects the stored value.
        # fps=0 means uncapped, which maps to the slider's top position.
        live_fps = settings.get('live_view_fps', 30)
        self.ids['live_view_fps_slider'].value = 65 if live_fps == 0 else live_fps

        self.ids['protocol_led_on_btn'].state = (
            'down' if settings.get('protocol_led_on') else 'normal'
        )

        self.ids['keep_led_between_steps_btn'].state = (
            'down' if settings.get('keep_led_between_steps') else 'normal'
        )

        # Setting the slider value drives the text via the kv binding
        # (text: format(acceleration_pct_slider.value)).
        self.ids['acceleration_pct_slider'].value = settings['motion']['acceleration_max_pct']

        # Populate the dropdown values BEFORE setting the text. The spinner
        # keeps text_autoupdate at its default (False); were it True,
        # reassigning .values would reset .text to the alphabetically-first
        # entry and fire on_text with the wrong scope. Setting text here fires
        # select_scope, which no-ops because it matches the stored model.
        self.load_scopes()
        self.ids['scope_spinner'].text = settings['microscope']

        self.stim_supported = firmware_stim_supported()
        self.ids['stimulation_settings_btn'].state = (
            'down' if settings['stimulation_enabled'] else 'normal'
        )

        # Populating the spinner fires on_text -> update_tiling_overlap, which
        # early-returns because the value already matches the setting (so the
        # load does not log a phantom user selection).
        self.ids['tiling_overlap_spinner'].text = f'{int(settings["tiling_overlap_percent"])}%'

        # Setting .active does not fire on_release (a user-press event), so this
        # populate does not re-run the handler -- no phantom toggle on open.
        self.ids['show_step_locations_id'].active = settings['show_step_locations']

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

    def update_video_max_fps(self):
        # 0 = no limit: the recording rate is then bounded only by
        # exposure and the delivery constant; non-zero is the user's
        # explicit cap on the recording cadence.
        settings = _app_ctx.ctx.settings
        widget = self.ids['video_max_fps_input']
        try:
            value = int(widget.text)
        except (ValueError, TypeError):
            value = -1
        if value < 0 or value > 200:
            from modules.notification_center import notifications

            notifications.warning(
                'Settings',
                'Invalid FPS limit',
                'Video max FPS must be between 0 and 200 (0 = no limit). '
                'Reverting to previous value.',
            )
            settings.setdefault('video', {})
            widget.text = str(settings['video'].get('max_fps', 0))
            return
        settings.setdefault('video', {})
        settings['video']['max_fps'] = value
        gui_logger.text_input_debounced('VIDEO_MAX_FPS', value)

    def update_video_max_duration(self):
        # Bounds the recording's frame budget (fps * duration); the
        # record start's disk floor check guards feasibility.
        settings = _app_ctx.ctx.settings
        widget = self.ids['video_max_duration_input']
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
            settings.setdefault('video', {})
            widget.text = str(get_manual_video_max_duration(settings))
            return
        settings.setdefault('video', {})
        settings['video']['max_duration_seconds'] = value
        gui_logger.text_input_debounced('VIDEO_MAX_DURATION_S', value)

    def update_video_timestamp_overlay(self):
        settings = _app_ctx.ctx.settings
        state = self.ids['video_timestamp_overlay_id'].active
        gui_logger.toggle('VIDEO_TIMESTAMP_OVERLAY', state)
        settings.setdefault('video', {})['timestamp_overlay'] = state

    def update_separate_folders_per_channel(self):
        settings = _app_ctx.ctx.settings
        state = self.ids['separate_folder_per_channel_id'].state == 'down'
        gui_logger.toggle('SEPARATE_FOLDERS', state)
        settings['separate_folder_per_channel'] = state

    def live_view_fps_slider(self):
        ctx = _app_ctx.ctx
        fps_val = int(self.ids['live_view_fps_slider'].value)
        gui_logger.slider('FPS', fps_val)
        # Values above 60 mean "uncapped" -- store 0 as sentinel.
        if fps_val > 60:
            fps_val = 0
        ctx.live_view_fps = fps_val
        with ctx.settings_lock:
            ctx.settings['live_view_fps'] = fps_val
        logger.info(
            f'[LVP Main  ] Live view FPS set to {"Max (uncapped)" if fps_val == 0 else fps_val}'
        )

        scope_display = ctx.scope_display
        if scope_display is not None:
            scope_display.stop()
            scope_display.start(fps=fps_val)

    def update_protocol_led_on(self):
        settings = _app_ctx.ctx.settings
        enabled = self.ids['protocol_led_on_btn'].state == 'down'
        gui_logger.toggle('PROTOCOL_LED_ON', enabled)
        settings['protocol_led_on'] = enabled

    def update_keep_led_between_steps(self):
        settings = _app_ctx.ctx.settings
        enabled = self.ids['keep_led_between_steps_btn'].state == 'down'
        gui_logger.toggle('KEEP_LED_BETWEEN_STEPS', enabled)
        settings['keep_led_between_steps'] = enabled

    def update_stimulation_settings(self):
        ctx = _app_ctx.ctx
        enabled = self.ids['stimulation_settings_btn'].state == 'down'
        gui_logger.toggle('STIMULATION_ENABLED', enabled)
        ctx.settings['stimulation_enabled'] = enabled
        # The microscope panel owns the per-layer stim sync; the startup load
        # re-uses the same owner, so the toggle and load stay in lockstep.
        ctx.motion_settings.ids['microscope_settings_id'].apply_stimulation_support()

    def update_tiling_overlap(self):
        ctx = _app_ctx.ctx
        overlap = TilingConfig.validate_overlap_percent(
            self.ids['tiling_overlap_spinner'].text.strip().rstrip('%')
        )
        # on_open populates the spinner with the stored value; that programmatic
        # write is not a user change, so skip the action log and redundant write.
        if overlap == ctx.settings['tiling_overlap_percent']:
            return
        gui_logger.select('TILING_OVERLAP', overlap)
        ctx.settings['tiling_overlap_percent'] = overlap

    def update_show_step_locations(self):
        ctx = _app_ctx.ctx
        enabled = bool(self.ids['show_step_locations_id'].active)
        gui_logger.toggle('SHOW_STEP_LOCATIONS', enabled)
        ctx.settings['show_step_locations'] = enabled
        ctx.stage.show_protocol_steps(enable=enabled)

    def load_scopes(self):
        scopes = _app_ctx.ctx.motion_settings.ids['microscope_settings_id'].scopes
        self.ids['scope_spinner'].values = list(scopes.keys())

    def select_scope(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        new_model = self.ids['scope_spinner'].text
        # on_open sets the spinner to the current model; selecting the value
        # already in effect is not a scope change, so skip the reconfigure.
        if new_model == settings['microscope']:
            return
        gui_logger.select('SCOPE', new_model)
        settings['microscope'] = new_model
        # Reconfigure the panel for the new scope through its single owner
        # (control visibility + read-only model label + stage redraw); the
        # startup path uses the same method so both reconfigure identically.
        ctx.motion_settings.ids['microscope_settings_id'].reconfigure_for_scope()

    def acceleration_pct_slider(self):
        acc_val = self.ids['acceleration_pct_slider'].value
        gui_logger.slider('ACCELERATION', acc_val)
        self.set_acceleration_limit(val_pct=acc_val)

    def acceleration_pct_text(self):
        acc_min = self.ids['acceleration_pct_slider'].min
        acc_max = self.ids['acceleration_pct_slider'].max
        try:
            acc_val = int(self.ids['acceleration_pct_text'].text)
        except (ValueError, TypeError):
            logger.debug(
                f'[Advanced ] Invalid acceleration input: '
                f'{self.ids["acceleration_pct_text"].text!r}'
            )
            return

        # The slider's [min, max] is the valid domain for the typed value. A
        # Kivy input_filter can't enforce a minimum on partial input (typing
        # "10" must allow the intermediate "1"), so clamp the validated value.
        acc_val = int(max(acc_min, min(acc_max, acc_val)))
        self.ids['acceleration_pct_slider'].value = acc_val
        self.ids['acceleration_pct_text'].text = str(acc_val)
        self.set_acceleration_limit(val_pct=acc_val)

    _ACCELERATION_DEBOUNCE_S = 0.10
    _acceleration_dispatch_trigger = None
    _pending_acceleration_pct = None

    def set_acceleration_limit(self, val_pct):
        """Apply acceleration limit (writes settings + dispatches motor command).

        The motor serial write goes through ``io_executor`` instead of running
        synchronously on MainThread. The slider's ``on_value`` event can fire at
        up to 60 Hz on a smooth drag -- without the executor route, every tick
        blocks the UI on a serial write. The settings dict is still updated
        synchronously so other UI code reading the slider sees the committed
        value immediately.

        The 100 ms ``Clock.create_trigger`` debounce coalesces rapid slider
        ticks into one motor write per debounce window. Final settle of the
        slider always lands on the last value the user picked.
        """
        ctx = _app_ctx.ctx
        with ctx.settings_lock:
            ctx.settings['motion']['acceleration_max_pct'] = val_pct
        # Stash the most recent value; the trigger reads it when it fires.
        self._pending_acceleration_pct = int(val_pct)
        if self._acceleration_dispatch_trigger is None:
            self._acceleration_dispatch_trigger = Clock.create_trigger(
                lambda dt: self._dispatch_acceleration_to_motor(),
                self._ACCELERATION_DEBOUNCE_S,
            )
        self._acceleration_dispatch_trigger()

    def _dispatch_acceleration_to_motor(self):
        """Send the most-recent acceleration value to the motor on IO_WORKER.

        Reads ``self._pending_acceleration_pct`` (latest stash from
        ``set_acceleration_limit``) and submits an IOTask through
        ``io_executor``. If the slider moved again while the trigger was
        pending, only the latest value reaches the motor -- no queued command
        burst.
        """
        ctx = _app_ctx.ctx
        if ctx is None or self._pending_acceleration_pct is None:
            return
        val_pct = self._pending_acceleration_pct
        scope = ctx.lumaview.scope if ctx.lumaview else None
        if scope is None:
            return
        ctx.io_executor.put(
            IOTask(
                action=scope.motion.set_acceleration_limit,
                kwargs={'val_pct': val_pct},
            )
        )

    def close(self):
        logger.debug('[Advanced ] AdvancedSettings closed')
        self.dismiss()


kv = Builder.load_string(
    """
<SectionHeader@Label>:
    font_size: '13sp'
    bold: True
    color: 0.55, 0.78, 1, 1
    halign: 'left'
    valign: 'bottom'
    text_size: self.width, None
    size_hint_y: None
    height: '24dp'
    padding: dp(4), 0

<AdvancedSettings>:
    size_hint_x: .35
    size_hint_y: None
    height: advanced_sections.minimum_height + dp(108)
    auto_dismiss: True
    title: 'Advanced Settings'

    BoxLayout:
        orientation: 'vertical'
        spacing: dp(6)

        # Sectioned settings: a vertical stack of Camera / Protocol / General
        # groups, each a full-width header over its own two-column sub-grid (no
        # full-width controls inside a grid -- Kivy GridLayout has no colspan,
        # so the header lives outside the grid). No ScrollView: the bars read as
        # nearly invisible in this theme, so the popup just grows with content.
        # Within a sub-grid, like-gated rows are kept in the same grid row so a
        # hidden pair collapses its whole row cleanly (Camera: the two
        # camera-gated toggles; Protocol: the two stage-gated rows).
        BoxLayout:
            id: advanced_sections
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            spacing: dp(4)

            SectionHeader:
                text: 'General'
            GridLayout:
                cols: 2
                size_hint_y: None
                height: self.minimum_height
                padding: dp(6), dp(2)
                spacing: dp(16), dp(6)

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Lumascope Model'
                        tooltip_text: 'Lumi, LS820, LS850, and LS850T'
                        size_hint_x: None
                        width: '110dp'
                        font_size: '12sp'
                    Spinner:
                        id: scope_spinner
                        disabled: app.protocol_running
                        sync_height: True
                        text: 'Select'
                        font_size: '12sp'
                        option_cls: 'SpinnerOption0'
                        on_release: root.load_scopes()
                        on_text: root.select_scope()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp' if root.xy_stage_supported else 0
                    opacity: 1 if root.xy_stage_supported else 0
                    disabled: not root.xy_stage_supported
                    Label:
                        id: acceleration_pct_label
                        text: 'Acceleration Max (%)'
                        tooltip_text: 'Maximum acceleration percentage for X-Y stage'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    ModSlider:
                        id: acceleration_pct_slider
                        disabled: app.protocol_running
                        min: 1
                        max: 100
                        value: 100
                        step: 1
                        cursor_size: '20dp','20dp'
                        cursor_image: './data/icons/slider_cursor.png'
                        track_width: dp(5)
                        value_track: True
                        value_track_width: dp(5)
                        on_release: root.acceleration_pct_slider()
                    TextInput:
                        id: acceleration_pct_text
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '40dp'
                        multiline: False
                        font_size: '12sp'
                        padding: ['4dp', (self.height-self.line_height)/2]
                        halign: 'right'
                        input_filter: 'int'
                        text: format(acceleration_pct_slider.value)
                        on_text_validate: root.acceleration_pct_text()
                        on_focus: if not self.focus: root.acceleration_pct_text()

                # Hidden when firmware lacks stim. The toggle's OWN height
                # collapses with the row, so the invisible button cannot overlap
                # a neighbor and, as a disabled ButtonBehavior, swallow that
                # neighbor's click.
                BoxLayout:
                    id: stim_settings_box
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp' if root.stim_supported else 0
                    opacity: 1 if root.stim_supported else 0
                    disabled: not root.stim_supported
                    Label:
                        font_size: '12sp'
                        text: 'Stimulation Settings'
                        tooltip_text: 'Enable stimulation features globally'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                    ToggleButton:
                        id: stimulation_settings_btn
                        disabled: app.protocol_running
                        size_hint: None, None
                        tooltip_text: 'Enable stimulation features globally'
                        width: '45dp'
                        height: '30dp' if root.stim_supported else 0
                        border: 0, 0, 0, 0
                        valign: 'middle'
                        background_normal: './data/icons/ToggleL.png'
                        background_down: './data/icons/ToggleRW.png'
                        on_release: root.update_stimulation_settings()
                        state: 'normal'

            SectionHeader:
                text: 'Camera'
            GridLayout:
                cols: 2
                size_hint_y: None
                height: self.minimum_height
                padding: dp(6), dp(2)
                spacing: dp(16), dp(6)

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
                        text: 'Video max FPS'
                        tooltip_text: 'Maximum frames per second for manual recording.\\n0 = no limit (camera free-run rate).\\nAt low FPS the camera is rate-limited so live preview slows too;\\nrestored to free-run when recording stops.'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    TextInput:
                        id: video_max_fps_input
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '45dp'
                        multiline: False
                        font_size: '12sp'
                        padding: ['4dp', (self.height-self.line_height)/2]
                        halign: 'right'
                        input_filter: 'int'
                        text: '0'
                        on_text_validate: root.update_video_max_fps()
                        on_focus: if not self.focus: root.update_video_max_fps()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Video Time Limit (s)'
                        tooltip_text: 'Global maximum video length, in seconds.\\nManual recording auto-stops at this limit (press Record\\nagain to stop sooner); a protocol video Step longer than\\nthis is flagged.'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    TextInput:
                        id: video_max_duration_input
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '45dp'
                        multiline: False
                        font_size: '12sp'
                        padding: ['4dp', (self.height-self.line_height)/2]
                        halign: 'right'
                        input_filter: 'int'
                        text: '30'
                        on_text_validate: root.update_video_max_duration()
                        on_focus: if not self.focus: root.update_video_max_duration()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Video Timestamp Overlay'
                        tooltip_text: 'Burn the capture timestamp into each video frame.\\nApplies to manual recordings and protocol video steps.'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    CheckBox:
                        id: video_timestamp_overlay_id
                        disabled: app.protocol_running
                        size_hint_x: None
                        width: '25dp'
                        active: True
                        on_release: root.update_video_timestamp_overlay()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Live View FPS'
                        tooltip_text: 'Maximum frames per second for live camera display'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    ModSlider:
                        id: live_view_fps_slider
                        disabled: app.protocol_running
                        min: 5
                        max: 65
                        value: 30
                        step: 5
                        cursor_size: '20dp','20dp'
                        cursor_image: './data/icons/slider_cursor.png'
                        track_width: dp(5)
                        value_track: True
                        value_track_width: dp(5)
                        on_release: root.live_view_fps_slider()
                    TextInput:
                        size_hint_x: None
                        width: '40dp'
                        multiline: False
                        font_size: '12sp'
                        padding: ['4dp', (self.height-self.line_height)/2]
                        halign: 'right'
                        text: 'Max' if int(live_view_fps_slider.value) > 60 else format(int(live_view_fps_slider.value))
                        readonly: True

            SectionHeader:
                text: 'Protocol'
            GridLayout:
                cols: 2
                size_hint_y: None
                height: self.minimum_height
                padding: dp(6), dp(2)
                spacing: dp(16), dp(6)

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        font_size: '12sp'
                        text: 'Preview LED when stepping'
                        tooltip_text: "Keep the step's LED on while manually navigating through protocol steps so you can preview the illumination. Does not affect LED behavior during a protocol scan."
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                    ToggleButton:
                        id: protocol_led_on_btn
                        disabled: app.protocol_running
                        size_hint: None, None
                        tooltip_text: "Keep the step's LED on while manually navigating through protocol steps so you can preview the illumination. Does not affect LED behavior during a protocol scan."
                        size: '45dp', '30dp'
                        border: 0, 0, 0, 0
                        valign: 'middle'
                        background_normal: './data/icons/ToggleL.png'
                        background_down: './data/icons/ToggleRW.png'
                        on_release: root.update_protocol_led_on()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        font_size: '12sp'
                        text: 'Keep LED on across moves'
                        tooltip_text: "During a protocol scan, keep the LED on while the stage moves between steps instead of switching it off and back on. Speeds up brightfield scans; off by default."
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                    ToggleButton:
                        id: keep_led_between_steps_btn
                        disabled: app.protocol_running
                        size_hint: None, None
                        tooltip_text: "During a protocol scan, keep the LED on while the stage moves between steps instead of switching it off and back on. Speeds up brightfield scans; off by default."
                        size: '45dp', '30dp'
                        border: 0, 0, 0, 0
                        valign: 'middle'
                        background_normal: './data/icons/ToggleL.png'
                        background_down: './data/icons/ToggleRW.png'
                        on_release: root.update_keep_led_between_steps()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp' if root.xy_stage_supported else 0
                    opacity: 1 if root.xy_stage_supported else 0
                    disabled: not root.xy_stage_supported
                    Label:
                        text: 'Tiling Overlap'
                        tooltip_text: 'Tile overlap percentage for acquisition tiling'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    Spinner:
                        id: tiling_overlap_spinner
                        disabled: app.protocol_running
                        sync_height: True
                        text: '0%'
                        font_size: '12sp'
                        size_hint_y: None
                        height: '30dp' if root.xy_stage_supported else 0
                        size_hint_x: None
                        width: '65dp'
                        option_cls: 'SpinnerOption0'
                        text_autoupdate: True
                        values: ('0%', '10%', '15%', '20%')
                        on_text: root.update_tiling_overlap()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp' if root.xy_stage_supported else 0
                    opacity: 1 if root.xy_stage_supported else 0
                    disabled: not root.xy_stage_supported
                    Label:
                        text: 'Show step locations'
                        tooltip_text: 'Display yellow cross for each Step'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                        font_size: '12sp'
                    CheckBox:
                        id: show_step_locations_id
                        size_hint_x: None
                        width: '30dp'
                        active: False
                        tooltip_text: 'Display yellow cross for each Step'
                        on_release: root.update_show_step_locations()

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: '30dp'
                    Label:
                        text: 'Save channels in separate folders'
                        tooltip_text: "Save each channels' images in separate folders"
                        font_size: '12sp'
                        halign: 'left'
                        valign: 'middle'
                        text_size: self.size
                    ToggleButton:
                        id: separate_folder_per_channel_id
                        disabled: app.protocol_running
                        size_hint: None, None
                        tooltip_text: "Save each channels' images in separate folders"
                        size: '45dp', '30dp'
                        border: 0, 0, 0, 0
                        valign: 'middle'
                        background_normal: './data/icons/ToggleL.png'
                        background_down: './data/icons/ToggleRW.png'
                        on_release: root.update_separate_folders_per_channel()
                        state: 'normal'

        Button:
            text: 'Close'
            size_hint_y: None
            height: '36dp'
            on_release: root.close()
"""
)
